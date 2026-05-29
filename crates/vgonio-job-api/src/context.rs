//! Execution-time context handed to a capability handler.
//!
//! A *capability handler* (also called a *service*) is the pure-compute
//! implementation of one named operation: `FittingService` runs
//! [`crate::ids::CapabilityId::fit`], `MeasurementService` runs
//! [`crate::ids::CapabilityId::measure_bsdf`] / `_ndf` / `_msf` / `_sdf`. The
//! handler is *not* the executor (the dispatcher); see the crate-root docs
//! for the executor-vs-handler split. This module is the executor-side
//! abstraction the handler talks to: progress, cancellation, artifacts, and
//! identity, packaged so the same handler body works under `LocalExecutor`
//! and `RemoteExecutor` without changes.
//!
//! When the executor accepts a [`crate::envelope::JobEnvelope`] and picks a
//! handler, it builds a [`JobContext`] and passes it by value to the handler.
//! Everything the handler needs to *do work and report on it* lives here:
//!
//! - identity ([`JobContext::job_id`], [`JobContext::idempotency_key`]),
//! - the progress channel ([`ProgressSender`]),
//! - cooperative cancellation ([`CancellationToken`]),
//! - the artifact-store interface ([`ArtifactStore`]),
//! - the inherited trace context, and
//! - an optional wall-clock deadline.
//!
//! # Lifecycle
//!
//! ```text
//! Executor                        Capability handler
//! --------                        ------------------
//! build JobContext { ... }
//! handler.run(ctx) ------------->
//!                                 ctx.progress.send(Started { ... })
//!                                 ... work ...
//!                                 ctx.cancel.check()?           // periodic
//!                                 ctx.progress.send(Progress { ... })
//!                                 let h = ctx.artifacts.publish(...)?;
//!                                 ctx.progress.send(Completed { ... })
//! <-- result --------------------
//! drop JobContext
//! ```
//!
//! [`JobContext`] is [`Clone`], cheaply so: every nested handle is either an
//! [`Arc`] or a [`std::sync::mpsc::Sender`]. Handlers may clone the context
//! into helper threads (rayon scope, dedicated reader, etc.) without having to
//! re-plumb the progress channel or cancellation flag.
//!
//! # Async-ness
//!
//! Phase 1 capability handlers run *synchronously* on rayon thread pools, so
//! everything here is sync. The plan calls out the migration path: when Phase 3
//! introduces async transport, the [`ProgressSender`] stays sync (handlers
//! don't change) and the executor-side consumer learns to bridge into async.
//! Picking [`std::sync::mpsc`] over `tokio::sync::mpsc` is deliberate for this
//! reason.

use std::{
    fmt::Debug,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    time::Instant,
};

use crate::{
    artifact::{ArtifactKind, ArtifactRef},
    envelope::TraceContext,
    error::{JobError, JobErrorCode},
    ids::{IdempotencyKey, JobId},
    progress::{Activity, JobEvent, Lifecycle, PhaseInstanceId},
};

/// Execution-time context handed to a capability handler.
///
/// Cheap to clone (every nested handle is an [`Arc`] or a channel sender);
/// handlers may pass clones into helper threads.
///
/// All fields are public to keep construction ergonomic on the executor side
/// (every executor implementation needs to mint these from scratch); handlers
/// should treat them as read-only and interact through the methods on each
/// sub-handle ([`ProgressSender::send`], [`CancellationToken::check`], etc.).
#[derive(Clone)]
pub struct JobContext {
    /// Identifies this job attempt. See [`crate::ids::JobId`] for the
    /// `JobId` versus [`IdempotencyKey`] distinction.
    pub job_id: JobId,
    /// The logical-request key the submitter chose. Handlers rarely need it
    /// directly, but it's threaded through so per-job state stores (caches,
    /// resumable checkpoints) can be keyed by it instead of `job_id`.
    pub idempotency_key: IdempotencyKey,
    /// Channel the handler emits [`JobEvent`]s on. Drops events silently
    /// if no one is listening; see [`ProgressSender`]. Handlers should
    /// only emit [`Activity`] variants via `progress.activity(...)`;
    /// [`Lifecycle`] is owned by the executor.
    pub progress: ProgressSender,
    /// Monotonic counter for minting [`PhaseInstanceId`]s within this
    /// job. Shared across clones so helper threads emitting their own
    /// phases get unique ids. The executor allocates this from zero.
    pub phase_counter: Arc<AtomicU32>,
    /// Cooperative cancellation. Handlers should call
    /// [`CancellationToken::check`] at natural boundaries (between iterations,
    /// before expensive I/O) and propagate the resulting error.
    pub cancel: CancellationToken,
    /// Handler-facing view of the artifact store. The executor injects a
    /// concrete implementation (local-fs in Phase 1, HTTP-backed in Phase 3);
    /// the trait keeps the handler decoupled from transport.
    pub artifacts: Arc<dyn ArtifactStore>,
    /// Distributed-tracing context inherited from the envelope. When the
    /// handler creates spans, it should chain them off this context.
    pub trace: TraceContext,
    /// Wall-clock instant after which the executor will cancel the job.
    /// `None` for jobs submitted without
    /// [`crate::resources::ResourceHints::max_runtime_secs`]. Use to short-
    /// circuit long iterations rather than relying solely on the executor
    /// firing the cancellation token at the deadline.
    pub deadline: Option<Instant>,
}

impl JobContext {
    /// Mints a fresh [`PhaseInstanceId`] from this context's shared
    /// counter. Each call returns a value greater than the previous one
    /// across all clones of this context, so helper threads that emit
    /// their own phases get distinct ids.
    pub fn next_phase_id(&self) -> PhaseInstanceId {
        PhaseInstanceId(self.phase_counter.fetch_add(1, Ordering::Relaxed))
    }
}

/// Handler-facing artifact store.
///
/// The executor injects a concrete implementation; handlers never see the
/// concrete type. Object-safe (`&dyn ArtifactStore` works) so the
/// [`JobContext::artifacts`] field can stay a trait object.
///
/// `Send + Sync + Debug` because:
/// - `Send + Sync` so an [`Arc<dyn ArtifactStore>`] can be cloned into helper threads.
/// - `Debug` so a future `#[derive(Debug)]` on [`JobContext`] doesn't require manual impl
///   gymnastics.
pub trait ArtifactStore: Send + Sync + Debug {
    /// Resolves an input artifact reference. Phase 1 returns the resolved ref
    /// after verifying the blob is present; Phase 3 may return a re-issued
    /// reference whose `origin` is updated to a local path after the worker
    /// pulled the bytes from the remote store.
    fn resolve(&self, id: &ArtifactRef) -> Result<ArtifactHandle, JobError>;
    /// Publishes a freshly-produced output blob and returns the ref the
    /// handler should include in its result. The store assigns the
    /// [`crate::ids::ArtifactId`] and computes the checksum.
    fn publish(&self, kind: ArtifactKind, bytes: bytes::Bytes) -> Result<ArtifactRef, JobError>;
}

/// Read handle for an artifact's bytes.
///
/// Two flavours so the consumer can pick the cheapest access pattern:
/// `Path` lets a handler `mmap` or stream from disk without copying;
/// `Bytes` lets it work over an already-loaded in-memory blob (cheap clone,
/// reference-counted under the hood). The store decides which to hand back.
#[derive(Clone)]
pub enum ArtifactHandle {
    /// Filesystem-backed; consumer may mmap or stream from the path.
    Path(std::path::PathBuf),
    /// In-memory bytes (small artifacts); consumer can read directly.
    Bytes(bytes::Bytes),
}

/// Send side of the job-event channel.
///
/// Backed by a [`std::sync::mpsc::Sender`]; cheap to clone (one `Arc` bump).
/// Send is best-effort: if the receiver is gone, the event is dropped on
/// the floor. Capability code is *not* expected to handle send errors;
/// the progress stream is observability, not the source of truth for
/// results.
///
/// Two helpers, one channel: [`Self::activity`] wraps an [`Activity`] in
/// [`JobEvent::Activity`] for the handler; [`Self::lifecycle`] wraps a
/// [`Lifecycle`] in [`JobEvent::Lifecycle`] for the executor. Handlers
/// MUST NOT call [`Self::lifecycle`]; the executor owns those transitions.
#[derive(Clone)]
pub struct ProgressSender {
    tx: std::sync::mpsc::Sender<JobEvent>,
}

impl ProgressSender {
    /// Wraps an existing mpsc sender. The executor builds the channel and
    /// owns the receiver side; this is the only constructor handlers need.
    pub fn new(tx: std::sync::mpsc::Sender<JobEvent>) -> Self { Self { tx } }

    /// Raw best-effort send. Drops the event silently if the receiver is
    /// gone. Prefer [`Self::activity`] / [`Self::lifecycle`] which wrap
    /// the inner enum for you.
    pub fn send(&self, event: JobEvent) { let _ = self.tx.send(event); }

    /// Emit an [`Activity`] event. Handler-facing API.
    pub fn activity(&self, event: Activity) { self.send(JobEvent::Activity(event)); }

    /// Emit a [`Lifecycle`] event. Executor-facing API; handlers MUST NOT
    /// call this.
    pub fn lifecycle(&self, event: Lifecycle) { self.send(JobEvent::Lifecycle(event)); }
}

/// Cooperative cancellation flag.
///
/// Cancellation is *cooperative*: setting the flag merely signals the handler
/// that it should stop at the next checkpoint. The handler must opt in by
/// calling [`Self::check`] (or [`Self::is_cancelled`]) at boundaries; nothing
/// in this type interrupts a running computation.
///
/// All clones share one [`AtomicBool`] under an [`Arc`], so cancelling one
/// clone is visible to every other handle into the same job.
#[derive(Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Fresh, un-cancelled token.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Sets the cancellation flag. Idempotent: calling twice has the same
    /// effect as calling once. Cannot be un-set; mint a new token for a fresh
    /// attempt.
    pub fn cancel(&self) { self.flag.store(true, Ordering::SeqCst); }

    /// Returns `true` if [`Self::cancel`] has been called on any clone of
    /// this token.
    pub fn is_cancelled(&self) -> bool { self.flag.load(Ordering::SeqCst) }

    /// Helper for capability code: returns `Err(JobError { code: Cancelled, ... })`
    /// if the flag is set, `Ok(())` otherwise. Call at boundaries and `?`
    /// through.
    ///
    /// The returned [`JobError`] uses [`JobErrorCode::Cancelled`] with
    /// `retriable: false`; clients decide whether to re-submit.
    pub fn check(&self) -> Result<(), JobError> {
        if self.is_cancelled() {
            Err(JobError {
                code: JobErrorCode::Cancelled,
                message: "job was cancelled".into(),
                retriable: false,
                details: None,
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{ArtifactKind, ArtifactOrigin, Checksum},
        ids::ArtifactId,
    };
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    // ----- CancellationToken -----

    #[test]
    fn cancellation_token_starts_uncancelled() {
        let t = CancellationToken::new();
        assert!(!t.is_cancelled());
        assert!(t.check().is_ok());
    }

    #[test]
    fn cancel_flips_is_cancelled() {
        let t = CancellationToken::new();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let t = CancellationToken::new();
        t.cancel();
        t.cancel();
        t.cancel();
        // Two cancels must still leave the token cancelled with the same
        // observable state. We can't observe "called twice" directly, but we
        // can confirm no panic and the flag stays set.
        assert!(t.is_cancelled());
    }

    #[test]
    fn check_returns_cancelled_error_after_cancel() {
        let t = CancellationToken::new();
        t.cancel();
        let err = t.check().unwrap_err();
        assert_eq!(err.code, JobErrorCode::Cancelled);
        assert!(!err.retriable, "cancellation must not be retriable");
        assert!(err.details.is_none());
    }

    #[test]
    fn cancellation_clones_share_the_flag() {
        // The clone-share semantic is the whole point of `Arc<AtomicBool>`:
        // a cancel on any clone must be visible from any other clone, so a
        // handler can hand a clone to a helper thread.
        let original = CancellationToken::new();
        let cloned = original.clone();

        // Confirm both see the same state before cancel.
        assert!(!original.is_cancelled());
        assert!(!cloned.is_cancelled());

        // Cancel via one clone, observe via the other.
        cloned.cancel();
        assert!(
            original.is_cancelled(),
            "cancel on clone must be visible on original"
        );
    }

    #[test]
    fn cancellation_is_observable_across_threads() {
        // Real-world usage: handler spawns a helper thread that polls the
        // token. Lock that in.
        let t = CancellationToken::new();
        let t_helper = t.clone();
        let handle = thread::spawn(move || {
            // Spin briefly waiting for cancellation. Bounded so the test
            // can't hang if the contract regresses.
            let deadline = Instant::now() + Duration::from_secs(2);
            while !t_helper.is_cancelled() {
                if Instant::now() > deadline {
                    return false;
                }
                thread::sleep(Duration::from_millis(1));
            }
            true
        });
        t.cancel();
        assert!(
            handle.join().unwrap(),
            "helper thread did not observe cancellation"
        );
    }

    // ----- ProgressSender -----

    fn ts() -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc)
    }

    #[test]
    fn progress_sender_delivers_lifecycle_event() {
        let (tx, rx) = mpsc::channel();
        let sender = ProgressSender::new(tx);
        sender.lifecycle(Lifecycle::Started { at: ts() });
        let received = rx.recv().expect("receiver got nothing");
        assert!(matches!(
            received,
            JobEvent::Lifecycle(Lifecycle::Started { .. })
        ));
    }

    #[test]
    fn progress_sender_delivers_activity_event() {
        let (tx, rx) = mpsc::channel();
        let sender = ProgressSender::new(tx);
        sender.activity(Activity::Message {
            phase: PhaseInstanceId(7),
            level: 0,
            text: "hi".into(),
        });
        let received = rx.recv().expect("receiver got nothing");
        assert!(matches!(
            received,
            JobEvent::Activity(Activity::Message { .. })
        ));
    }

    #[test]
    fn progress_sender_preserves_order() {
        let (tx, rx) = mpsc::channel();
        let sender = ProgressSender::new(tx);
        sender.lifecycle(Lifecycle::Queued { at: ts() });
        sender.lifecycle(Lifecycle::Started { at: ts() });
        sender.lifecycle(Lifecycle::Completed { at: ts() });
        assert!(matches!(
            rx.recv().unwrap(),
            JobEvent::Lifecycle(Lifecycle::Queued { .. })
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            JobEvent::Lifecycle(Lifecycle::Started { .. })
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            JobEvent::Lifecycle(Lifecycle::Completed { .. })
        ));
    }

    #[test]
    fn send_after_receiver_dropped_does_not_panic() {
        // The contract says send is best-effort: the event is dropped on the
        // floor if no one's listening. A panic here would force handlers to
        // wrap every `send` call.
        let (tx, rx) = mpsc::channel();
        let sender = ProgressSender::new(tx);
        drop(rx);
        sender.lifecycle(Lifecycle::Completed { at: ts() });
        // Reaching this line is the assertion.
    }

    #[test]
    fn progress_sender_clones_share_underlying_channel() {
        // The handler is expected to clone the sender into helper threads;
        // every clone must deliver to the same receiver.
        let (tx, rx) = mpsc::channel();
        let sender = ProgressSender::new(tx);
        let helper = sender.clone();
        sender.lifecycle(Lifecycle::Queued { at: ts() });
        helper.lifecycle(Lifecycle::Completed { at: ts() });
        assert!(matches!(
            rx.recv().unwrap(),
            JobEvent::Lifecycle(Lifecycle::Queued { .. })
        ));
        assert!(matches!(
            rx.recv().unwrap(),
            JobEvent::Lifecycle(Lifecycle::Completed { .. })
        ));
    }

    #[test]
    fn next_phase_id_monotonic_across_clones() {
        // Helper threads must mint distinct ids; the counter lives on the
        // context (Arc'd) so clones share it.
        let counter = Arc::new(AtomicU32::new(0));
        let (tx, _rx) = mpsc::channel();
        let ctx = JobContext {
            job_id: JobId::new(),
            idempotency_key: IdempotencyKey("test".into()),
            progress: ProgressSender::new(tx),
            cancel: CancellationToken::new(),
            artifacts: Arc::new(MockStore),
            trace: TraceContext::default(),
            deadline: None,
            phase_counter: counter,
        };
        let a = ctx.next_phase_id();
        let b = ctx.clone().next_phase_id();
        let c = ctx.next_phase_id();
        assert_eq!(a, PhaseInstanceId(0));
        assert_eq!(b, PhaseInstanceId(1));
        assert_eq!(c, PhaseInstanceId(2));
    }

    // ----- ArtifactHandle -----

    #[test]
    fn artifact_handle_path_variant_constructible() {
        let h = ArtifactHandle::Path(std::path::PathBuf::from("/tmp/blob.vgms"));
        match h {
            ArtifactHandle::Path(p) => assert_eq!(p.to_str(), Some("/tmp/blob.vgms")),
            ArtifactHandle::Bytes(_) => panic!("unexpected Bytes"),
        }
    }

    #[test]
    fn artifact_handle_bytes_variant_constructible() {
        let h = ArtifactHandle::Bytes(bytes::Bytes::from_static(b"hello"));
        match h {
            ArtifactHandle::Bytes(b) => assert_eq!(&b[..], b"hello"),
            ArtifactHandle::Path(_) => panic!("unexpected Path"),
        }
    }

    #[test]
    fn artifact_handle_is_clone() {
        // Compile-time check via a generic function; if the derive ever gets
        // dropped this test stops compiling.
        fn assert_clone<T: Clone>(_: &T) {}
        let h = ArtifactHandle::Bytes(bytes::Bytes::from_static(b"x"));
        assert_clone(&h);
        let _h2 = h.clone();
    }

    // ----- JobContext + ArtifactStore -----

    /// Trivial in-memory store so we can build a `JobContext` for the
    /// compile-time + clone-shape checks below. Not exposed as a public
    /// helper; tests own it.
    #[derive(Debug, Default)]
    struct MockStore;

    impl ArtifactStore for MockStore {
        fn resolve(&self, _r: &ArtifactRef) -> Result<ArtifactHandle, JobError> {
            Ok(ArtifactHandle::Bytes(bytes::Bytes::from_static(b"")))
        }

        fn publish(
            &self,
            kind: ArtifactKind,
            bytes: bytes::Bytes,
        ) -> Result<ArtifactRef, JobError> {
            Ok(ArtifactRef {
                id: ArtifactId::new(),
                kind,
                origin: ArtifactOrigin::Inline,
                checksum: Checksum::sha256_hex("00"),
                size_bytes: bytes.len() as u64,
            })
        }
    }

    fn sample_context() -> (JobContext, mpsc::Receiver<JobEvent>) {
        let (tx, rx) = mpsc::channel();
        let ctx = JobContext {
            job_id: JobId::new(),
            idempotency_key: IdempotencyKey::from_hash(b"seed"),
            progress: ProgressSender::new(tx),
            cancel: CancellationToken::new(),
            artifacts: Arc::new(MockStore),
            trace: TraceContext::default(),
            deadline: None,
            phase_counter: Arc::new(AtomicU32::new(0)),
        };
        (ctx, rx)
    }

    #[test]
    fn job_context_builds() {
        let (ctx, _rx) = sample_context();
        assert!(!ctx.cancel.is_cancelled());
        assert!(ctx.deadline.is_none());
    }

    #[test]
    fn job_context_is_clone_and_clones_share_progress_and_cancel() {
        // The whole reason JobContext is Clone: a handler can clone into
        // helper threads and still cancel / report progress through them.
        let (ctx, rx) = sample_context();
        let cloned = ctx.clone();

        // Cancellation via one clone must be visible via the other (Arc
        // semantics on the inner flag).
        cloned.cancel.cancel();
        assert!(ctx.cancel.is_cancelled());

        // A progress event sent on one clone must reach the original
        // receiver (sender clones share the channel).
        cloned.progress.lifecycle(Lifecycle::Started { at: ts() });
        assert!(matches!(
            rx.recv().unwrap(),
            JobEvent::Lifecycle(Lifecycle::Started { .. })
        ));
    }

    #[test]
    fn job_context_is_send_and_sync() {
        // Compile-time: ensure the trait-object field doesn't accidentally
        // make the context !Send or !Sync. Handlers need both for rayon
        // scopes; if a new field regresses this, the test breaks loudly.
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let (ctx, _rx) = sample_context();
        assert_send_sync(&ctx);
    }

    #[test]
    fn artifact_store_is_object_safe() {
        // The contract: JobContext stores Arc<dyn ArtifactStore>, so the
        // trait must be object-safe. If a future method adds a `Self` return
        // or a generic, this stops compiling.
        let _arc: Arc<dyn ArtifactStore> = Arc::new(MockStore);
    }

    #[test]
    fn artifact_store_publish_round_trip() {
        // Smoke test that the mock store's publish returns a ref whose
        // origin/kind match the input. Validates the trait contract is
        // implementable end-to-end.
        let store = MockStore;
        let r = store
            .publish(ArtifactKind::Vgbsdf, bytes::Bytes::from_static(b"fake-zip"))
            .unwrap();
        assert!(matches!(r.kind, ArtifactKind::Vgbsdf));
        assert_eq!(r.size_bytes, b"fake-zip".len() as u64);
    }
}

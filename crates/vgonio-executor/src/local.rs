//! In-process implementation of [`crate::Executor`].
//!
//! [`LocalExecutor`] dispatches each accepted envelope to a registered
//! capability handler on a freshly spawned OS thread, sets up the progress /
//! result channels the caller observes through [`crate::JobHandle`], and
//! brackets the run with [`Lifecycle::Started`] / [`Lifecycle::Completed`]
//! (or [`Lifecycle::Failed`]) events wrapped in [`JobEvent`]s.
//!
//! # What this implementation deliberately does NOT do (yet)
//!
//! - **No thread pool.** One OS thread per submitted job. Phase 1 is sized for one or two
//!   concurrent jobs; Phase 4 introduces a bounded thread pool to support N parallel local jobs.
//! - **No async / `tokio` integration.** Handlers are synchronous; the channels are
//!   [`std::sync::mpsc`] so they work without a runtime.
//! - **No artifact aggregation.** [`crate::JobOutcome::artifacts`] is always empty; handlers that
//!   publish artifacts do so via [`vgn_job_api::context::ArtifactStore::publish`] and embed the
//!   resulting refs inside their result payload.
//! - **No deadline enforcement.** [`vgn_job_api::context::JobContext::deadline`] is wired through
//!   but the executor itself never fires the cancel token; only an explicit
//!   [`crate::Executor::cancel`] call does. Deadline-driven cancellation is a Phase 4 concern.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{mpsc, Arc, RwLock},
    thread,
};

use std::sync::atomic::AtomicU32;
use vgn_artifact::LocalFsStore;

use vgn_job_api::{
    context::{ArtifactStore, CancellationToken, JobContext, ProgressSender},
    envelope::JobEnvelope,
    error::JobError,
    ids::JobId,
    progress::{JobEvent, Lifecycle},
    PROTOCOL_VERSION,
};

use crate::{
    executor::{Executor, ExecutorError, JobHandle, JobOutcome},
    registry::CapabilityRegistry,
};

/// In-process job executor.
///
/// Owns:
///
/// - a [`CapabilityRegistry`] guarded by an [`RwLock`] (so handlers can in principle be added after
///   construction, though Phase 1 uses startup-only registration);
/// - an [`Arc`] of the artifact store the executor injects into every [`JobContext`];
/// - a `JobId → CancellationToken` table for in-flight jobs, so [`Self::cancel`] can flip the right
///   token.
///
/// Cheap to clone behind an `Arc` (all internal handles are `Arc`/`RwLock`).
/// The struct itself isn't `Clone`; wrap it in `Arc<LocalExecutor>` when
/// multiple owners are needed.
pub struct LocalExecutor {
    /// Registered capability handlers. The lock is held only briefly during
    /// dispatch to look up + clone the matching `Arc<Handler>`.
    registry: Arc<RwLock<CapabilityRegistry>>,
    /// Handler-facing artifact store, shared into every [`JobContext`] this
    /// executor builds.
    artifact_store: Arc<dyn ArtifactStore>,
    /// Live jobs' cancellation tokens, keyed by [`JobId`]. Entries are
    /// inserted at submit-time. Phase 1 never evicts (the table grows for
    /// the executor's lifetime); the leak is bounded by the workload and is
    /// addressed in Phase 4 along with thread-pool work.
    cancels: Arc<RwLock<HashMap<JobId, CancellationToken>>>,
}

impl LocalExecutor {
    /// Builds a `LocalExecutor` from a populated registry and a
    /// caller-provided artifact store.
    ///
    /// Use this constructor when you need to inject a custom store (a
    /// test double, an in-memory store, a remote-backed one). For the
    /// common "I want a local-fs store under some cache dir" path, use
    /// [`Self::from_cache_dir`] instead.
    pub fn new(registry: CapabilityRegistry, artifact_store: Arc<dyn ArtifactStore>) -> Self {
        Self {
            registry: Arc::new(RwLock::new(registry)),
            artifact_store,
            cancels: Arc::default(),
        }
    }

    /// Builds a `LocalExecutor` whose artifact store is a [`LocalFsStore`]
    /// rooted at `{cache_dir}/artifacts`.
    ///
    /// Creates the `artifacts/` subdirectory (and its `objects/` child)
    /// eagerly so submission can't fail later for missing directories.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`std::io::Error`] if the directories can't be
    /// created (typically a permission or filesystem-layout problem).
    pub fn from_cache_dir(
        registry: CapabilityRegistry,
        cache_dir: impl Into<PathBuf>,
    ) -> Result<Self, std::io::Error> {
        let artifact_store = LocalFsStore::new(cache_dir.into().join("artifacts"))?;
        Ok(Self::new(registry, artifact_store))
    }
}

impl Executor for LocalExecutor {
    fn submit(&self, envelope: JobEnvelope) -> Result<JobHandle, ExecutorError> {
        // 1. Protocol-version gate. Mismatch is a producer bug; reject hard.
        if envelope.protocol_version != PROTOCOL_VERSION {
            return Err(ExecutorError::ProtocolMismatch {
                got: envelope.protocol_version,
                supported: vec![PROTOCOL_VERSION],
            });
        }

        // 2. Look up the handler. Hold the read lock only long enough to clone the `Arc<Handler>`;
        //    never call the handler while holding it.
        let handler = {
            let registry = self.registry.read().unwrap();
            registry
                .get(&envelope.capability_id)
                .ok_or_else(|| ExecutorError::NoHandler(envelope.capability_id.clone()))?
                .clone()
        };

        // 3. Build the channels the caller sees through the JobHandle.
        let (event_tx, event_rx) = mpsc::channel::<JobEvent>();
        let (result_tx, result_rx) = mpsc::channel::<Result<JobOutcome, JobError>>();

        // 4. Register a cancellation token under this attempt's JobId so `cancel(job_id)` can flip
        //    it.
        let cancel = CancellationToken::new();
        self.cancels
            .write()
            .unwrap()
            .insert(envelope.job_id, cancel.clone());

        let job_id = envelope.job_id;
        let progress = ProgressSender::new(event_tx);
        let artifacts = self.artifact_store.clone();
        let trace = envelope.trace.clone();
        let envelope_owned = envelope;

        // 5. Spawn the worker thread. Phase 1 = one OS thread per job; see module docs for the
        //    thread-pool roadmap.
        thread::Builder::new()
            .name(format!("vgn-job-{job_id}"))
            .spawn(move || {
                let ctx = JobContext {
                    job_id,
                    idempotency_key: envelope_owned.idempotency_key.clone(),
                    progress: progress.clone(),
                    cancel,
                    artifacts,
                    trace,
                    deadline: None,
                    phase_counter: Arc::new(AtomicU32::new(0)),
                };

                // `Started` brackets the handler call; pair with the
                // terminal event below.
                progress.emit_lifecycle(Lifecycle::Started {
                    at: chrono::Utc::now(),
                });

                let out = handler(&envelope_owned, ctx);
                let outcome = match out {
                    Ok(payload) => {
                        progress.emit_lifecycle(Lifecycle::Completed {
                            at: chrono::Utc::now(),
                        });
                        // `artifacts` is always empty in Phase 1; capability
                        // handlers embed published refs inside their payload.
                        // See JobOutcome::artifacts docs.
                        Ok(JobOutcome {
                            payload,
                            artifacts: vec![],
                        })
                    },
                    Err(err) => {
                        // The progress stream is the *canonical* error record;
                        // sending it before the result so consumers that watch
                        // events also see the failure even if they drop the
                        // result receiver.
                        progress.emit_lifecycle(Lifecycle::Failed {
                            error: err.clone(),
                            at: chrono::Utc::now(),
                        });
                        Err(err)
                    },
                };
                // If the caller has already dropped the result receiver
                // (e.g. they're observing only progress events), the send
                // fails silently. The job still finished correctly.
                let _ = result_tx.send(outcome);
            })
            .map_err(|e| ExecutorError::Other(e.to_string()))?;

        Ok(JobHandle {
            job_id,
            events: event_rx,
            result: result_rx,
        })
    }

    fn cancel(&self, job_id: JobId) -> Result<(), ExecutorError> {
        let cancels = self.cancels.read().unwrap();
        let token = cancels
            .get(&job_id)
            .ok_or(ExecutorError::UnknownJob(job_id))?;
        token.cancel();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{mpsc as smpsc, Arc},
        thread,
        time::Duration,
    };

    use bytes::Bytes;
    use tempfile::tempdir;
    use vgn_job_api::{
        envelope::{JobEnvelope, PayloadEncoding, TraceContext},
        error::{JobError, JobErrorCode},
        ids::{CapabilityId, IdempotencyKey, JobId},
        progress::{Activity, JobEvent, Lifecycle, PhaseInstanceId, ProgressUnits},
        resources::ResourceHints,
    };

    use super::*;

    /// Minimal envelope addressed at `cap` with empty payload. Override
    /// individual fields after construction when a test needs to.
    fn smoke_envelope(cap: &str) -> JobEnvelope {
        JobEnvelope {
            job_id: JobId::new(),
            protocol_version: PROTOCOL_VERSION,
            capability_id: CapabilityId(cap.into()),
            capability_version: 1,
            payload_encoding: PayloadEncoding::Json,
            payload: Bytes::new(),
            resources: ResourceHints::default(),
            inputs: vec![],
            idempotency_key: IdempotencyKey("test".into()),
            trace: TraceContext::default(),
        }
    }

    /// Builds an executor backed by a fresh temp directory. Returns the
    /// `TempDir` alongside so the caller can keep the artifact directory
    /// alive for the duration of the test.
    fn make_executor() -> (LocalExecutor, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let exec = LocalExecutor::from_cache_dir(CapabilityRegistry::new(), dir.path()).unwrap();
        (exec, dir)
    }

    // ----- submit: happy path -----

    #[test]
    fn dispatches_to_registered_handler_and_echoes_payload() {
        // The handler echoes the envelope payload back; verifies that the
        // envelope (not a placeholder) reaches the handler unchanged.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("echo".into()),
            Arc::new(|env, _ctx| Ok(env.payload.clone())),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();

        let mut env = smoke_envelope("echo");
        env.payload = Bytes::from_static(b"hello");
        let handle = exec.submit(env).unwrap();
        let outcome = handle.result.recv().unwrap().unwrap();
        assert_eq!(&outcome.payload[..], b"hello");
    }

    #[test]
    fn submit_returns_job_id_matching_envelope() {
        // The JobHandle must surface the envelope's id verbatim; the cancel
        // table is keyed on that same id, so the round-trip is load-bearing.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("noop".into()),
            Arc::new(|_env, _ctx| Ok(Bytes::new())),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let env = smoke_envelope("noop");
        let envelope_id = env.job_id;
        let handle = exec.submit(env).unwrap();
        assert_eq!(handle.job_id, envelope_id);
        let _ = handle.result.recv().unwrap();
    }

    #[test]
    fn outcome_artifacts_is_empty_in_phase_1() {
        // Locks in the Phase 1 contract: handlers publish via
        // ctx.artifacts.publish and embed refs in the payload; the
        // out-of-band `artifacts` vec stays empty until Phase 4.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("noop".into()),
            Arc::new(|_env, _ctx| Ok(Bytes::new())),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("noop")).unwrap();
        let outcome = handle.result.recv().unwrap().unwrap();
        assert!(outcome.artifacts.is_empty());
    }

    // ----- submit: failure modes -----

    #[test]
    fn rejects_unknown_capability() {
        let (exec, _dir) = make_executor();
        let err = exec.submit(smoke_envelope("unknown")).unwrap_err();
        match err {
            ExecutorError::NoHandler(id) => assert_eq!(id.0, "unknown"),
            other => panic!("expected NoHandler, got {other:?}"),
        }
    }

    #[test]
    fn rejects_protocol_version_mismatch() {
        let (exec, _dir) = make_executor();
        let mut env = smoke_envelope("anything");
        env.protocol_version = PROTOCOL_VERSION + 99;
        let err = exec.submit(env).unwrap_err();
        match err {
            ExecutorError::ProtocolMismatch { got, supported } => {
                assert_eq!(got, PROTOCOL_VERSION + 99);
                assert_eq!(supported, vec![PROTOCOL_VERSION]);
            },
            other => panic!("expected ProtocolMismatch, got {other:?}"),
        }
    }

    #[test]
    fn handler_error_propagates_to_result_receiver() {
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("boom".into()),
            Arc::new(|_env, _ctx| {
                Err(JobError {
                    code: JobErrorCode::HandlerError,
                    message: "boom".into(),
                    retriable: false,
                    details: None,
                })
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("boom")).unwrap();
        let err = handle.result.recv().unwrap().unwrap_err();
        assert_eq!(err.code, JobErrorCode::HandlerError);
        assert_eq!(err.message, "boom");
    }

    // ----- progress event lifecycle -----

    #[test]
    fn emits_started_then_completed_on_success() {
        // The executor brackets the handler with Started / Completed.
        // Locking that in protects the bridge: a bridge that renders
        // "started"/"completed" relies on the executor actually sending
        // those events.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("ok".into()),
            Arc::new(|_env, _ctx| Ok(Bytes::new())),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("ok")).unwrap();
        // Wait for terminal result so the worker has finished sending.
        let _ = handle.result.recv().unwrap();
        let events: Vec<JobEvent> = handle.events.iter().collect();
        assert!(
            matches!(
                events.first(),
                Some(JobEvent::Lifecycle(Lifecycle::Started { .. }))
            ),
            "first event must be Lifecycle::Started, got {events:?}",
        );
        assert!(
            matches!(
                events.last(),
                Some(JobEvent::Lifecycle(Lifecycle::Completed { .. }))
            ),
            "last event must be Lifecycle::Completed, got {events:?}",
        );
    }

    #[test]
    fn emits_started_then_failed_on_handler_error() {
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("nope".into()),
            Arc::new(|_env, _ctx| {
                Err(JobError {
                    code: JobErrorCode::Other,
                    message: "nope".into(),
                    retriable: false,
                    details: None,
                })
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("nope")).unwrap();
        let _ = handle.result.recv().unwrap();
        let events: Vec<JobEvent> = handle.events.iter().collect();
        assert!(matches!(
            events.first(),
            Some(JobEvent::Lifecycle(Lifecycle::Started { .. }))
        ));
        match events.last() {
            Some(JobEvent::Lifecycle(Lifecycle::Failed { error, .. })) => {
                assert_eq!(error.message, "nope")
            },
            other => panic!("expected Lifecycle::Failed last, got {other:?}"),
        }
    }

    #[test]
    fn handler_can_emit_progress_through_context() {
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("ticky".into()),
            Arc::new(|_env, ctx| {
                ctx.progress.emit_activity(Activity::Progress {
                    phase: PhaseInstanceId(0),
                    fraction: 0.5,
                    units: ProgressUnits::Ratio,
                    label: Some("halfway".into()),
                });
                Ok(Bytes::new())
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("ticky")).unwrap();
        let _ = handle.result.recv().unwrap();
        let events: Vec<JobEvent> = handle.events.iter().collect();
        let saw_progress = events.iter().any(|e| {
            matches!(
                e,
                JobEvent::Activity(Activity::Progress { fraction, .. })
                    if (*fraction - 0.5).abs() < 1e-6
            )
        });
        assert!(saw_progress, "expected a Progress(0.5) event in {events:?}");
    }

    // ----- caller drops receivers -----

    #[test]
    fn job_still_runs_when_caller_drops_event_receiver() {
        // The plan calls this out as part of the best-effort contract on
        // ProgressSender: a caller that doesn't care about progress drops
        // the receiver; the handler must not panic, the result still lands.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("quiet".into()),
            Arc::new(|_env, ctx| {
                for _ in 0..16 {
                    ctx.progress.emit_activity(Activity::Message {
                        phase: PhaseInstanceId(0),
                        level: 7,
                        text: "spam".into(),
                    });
                }
                Ok(Bytes::from_static(b"done"))
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("quiet")).unwrap();
        drop(handle.events); // caller doesn't care about progress
        let outcome = handle.result.recv().unwrap().unwrap();
        assert_eq!(&outcome.payload[..], b"done");
    }

    // ----- cancellation -----

    #[test]
    fn cancel_propagates_to_running_handler() {
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("slow".into()),
            Arc::new(|_env, ctx| {
                // Tight cancellation polling loop: in worst case (cancel
                // arrives just after we entered the loop) we wait ~2s
                // before observing it. The test cancels after a fixed
                // 20ms, so we expect to observe well before the loop
                // ends naturally.
                for _ in 0..2000 {
                    ctx.cancel.check()?;
                    thread::sleep(Duration::from_millis(1));
                }
                Ok(Bytes::new())
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("slow")).unwrap();
        thread::sleep(Duration::from_millis(20));
        exec.cancel(handle.job_id).unwrap();
        let err = handle.result.recv().unwrap().unwrap_err();
        assert_eq!(err.code, JobErrorCode::Cancelled);
    }

    #[test]
    fn cancel_unknown_job_returns_unknown_job_error() {
        let (exec, _dir) = make_executor();
        let err = exec.cancel(JobId::new()).unwrap_err();
        assert!(matches!(err, ExecutorError::UnknownJob(_)));
    }

    #[test]
    fn cancel_is_idempotent_on_live_job() {
        // Calling cancel twice in a row is a real pattern (UI debounce,
        // user mashes a button); it must succeed both times.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("slow2".into()),
            Arc::new(|_env, ctx| {
                for _ in 0..2000 {
                    ctx.cancel.check()?;
                    thread::sleep(Duration::from_millis(1));
                }
                Ok(Bytes::new())
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("slow2")).unwrap();
        thread::sleep(Duration::from_millis(20));
        exec.cancel(handle.job_id).unwrap();
        // Second cancel: still in the table (we don't evict), so it
        // succeeds rather than UnknownJob.
        exec.cancel(handle.job_id).unwrap();
        let _ = handle.result.recv().unwrap(); // drain so the thread exits cleanly
    }

    // ----- concurrency -----

    #[test]
    fn multiple_jobs_run_in_parallel() {
        // Two long-running jobs submitted back-to-back must run on
        // separate threads (Phase 1 spawns per-job), so the total elapsed
        // time is closer to one job's runtime than two.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("sleepy".into()),
            Arc::new(|_env, _ctx| {
                thread::sleep(Duration::from_millis(100));
                Ok(Bytes::new())
            }),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();

        let start = std::time::Instant::now();
        let h1 = exec.submit(smoke_envelope("sleepy")).unwrap();
        let h2 = exec.submit(smoke_envelope("sleepy")).unwrap();
        let _ = h1.result.recv().unwrap();
        let _ = h2.result.recv().unwrap();
        let elapsed = start.elapsed();
        // Sequential would be ~200ms; parallel ~100ms. Leave 80ms
        // headroom for slow test runners.
        assert!(
            elapsed < Duration::from_millis(180),
            "two 100ms jobs took {elapsed:?}, expected near-parallel execution"
        );
    }

    // ----- channel contract -----

    #[test]
    fn event_channel_closes_after_terminal_event() {
        // Once the worker thread exits, the sender side of the events
        // channel drops, and the receiver returns Err. The bridge relies
        // on this to unblock its `for ev in events.iter()` loop.
        let dir = tempdir().unwrap();
        let mut reg = CapabilityRegistry::new();
        reg.register(
            CapabilityId("done".into()),
            Arc::new(|_env, _ctx| Ok(Bytes::new())),
        );
        let exec = LocalExecutor::from_cache_dir(reg, dir.path()).unwrap();
        let handle = exec.submit(smoke_envelope("done")).unwrap();
        let _ = handle.result.recv().unwrap();
        // Drain all events; the iterator stops cleanly when the sender drops.
        let _events: Vec<JobEvent> = handle.events.iter().collect();
        // A further recv must observe the closed channel.
        assert!(matches!(handle.events.recv(), Err(smpsc::RecvError)));
    }
}

//! The [`Executor`] trait and the supporting handle / outcome / error types.
//!
//! This module is intentionally transport-agnostic: nothing here knows whether
//! a job will run in-process, on a worker across the network, or in a future
//! hybrid scheduler. The concrete implementations live in [`crate::local`] and
//! `vgonio-transport-http`.

use std::sync::mpsc::Receiver;

use thiserror::Error;
use vgn_job_api::{
    artifact::ArtifactRef,
    envelope::JobEnvelope,
    error::JobError,
    ids::{CapabilityId, JobId},
    progress::JobEvent,
};

/// What a successfully completed job returns to its submitter.
///
/// The `payload` is the capability-specific result bytes, encoded under
/// whatever [`vgn_job_api::envelope::PayloadEncoding`] the envelope declared;
/// only the matching capability handler knows how to decode them.
///
/// `artifacts` lists every blob the handler published via
/// [`vgn_job_api::context::ArtifactStore::publish`] over the course of the run.
/// Currently always returns an empty vector; handlers publish through
/// `ctx.artifacts` and embed the resulting [`ArtifactRef`]s inside `payload`
/// instead. The field is reserved for later work that needs to surface
/// artifact lists out-of-band (e.g. progress UIs that link to outputs while
/// the job is still running).
#[derive(Debug, Clone)]
pub struct JobOutcome {
    /// Encoded result bytes. Decoder is chosen by the capability, not the
    /// executor.
    pub payload: bytes::Bytes,
    /// Output artifacts the handler explicitly surfaced through the result
    /// path. Always empty in Phase 1; see the struct docs for the rationale.
    pub artifacts: Vec<ArtifactRef>,
}

/// Caller-facing handle to one submitted job.
///
/// Returned by [`Executor::submit`]. Carries:
///
/// - the [`JobId`] minted for this attempt (use it with [`Executor::cancel`]),
/// - a receiver of [`JobEvent`]s, drained either directly or via [`crate::LocalStatusBridge`],
/// - a one-shot receiver of the final [`Result<JobOutcome, JobError>`].
///
/// # Channel lifecycle
///
/// - **`events`**: stays open for the lifetime of the job. The executor drops its send side when
///   the handler finishes (success or failure); after that, `recv()` returns [`Err`]
///   ([`std::sync::mpsc::RecvError`]). Dropping `events` early is fine; the executor's sends become
///   no-ops, the handler keeps running, and the final result still lands on `result`.
/// - **`result`**: receives exactly one value. `Ok` carries a [`JobOutcome`]; `Err` carries the
///   [`JobError`] that the handler returned. Disconnect (sender dropped without sending) means the
///   worker thread panicked.
///
/// The handle holds no [`Send`] cell that prevents you from passing it between
/// threads; both receivers are `Send`.
#[derive(Debug)]
pub struct JobHandle {
    /// The minted attempt ID. Pass back to [`Executor::cancel`] to abort.
    pub job_id: JobId,
    /// Stream of progress observations. See [`JobEvent`] for the lifecycle
    /// + activity contract.
    pub events: Receiver<JobEvent>,
    /// One-shot delivery of the terminal result.
    pub result: Receiver<Result<JobOutcome, JobError>>,
}

/// The dispatch seam between adapter code and capability handlers.
///
/// Same trait surface for local (in-process) and remote executors,
/// so adapter code never branches on the dispatch mode. Implementations are
/// expected to be cheap to clone behind an [`std::sync::Arc`] and safe to share
/// across threads (`Send + Sync`).
pub trait Executor: Send + Sync {
    /// Submits an envelope for execution.
    ///
    /// Returns immediately with a [`JobHandle`] once the executor has accepted
    /// the envelope (validated the protocol version, found a handler,
    /// reserved a slot). Actual execution runs in the background; the caller
    /// observes progress and the terminal result through the returned
    /// handle's receivers.
    ///
    /// # Errors
    ///
    /// - [`ExecutorError::ProtocolMismatch`] if `envelope.protocol_version` disagrees with the
    ///   versions this executor speaks.
    /// - [`ExecutorError::NoHandler`] if no handler is registered for the envelope's capability.
    /// - [`ExecutorError::Other`] for back-end-specific failures during admission (thread spawn,
    ///   slot allocation, …). Inspect the message for details.
    fn submit(&self, envelope: JobEnvelope) -> Result<JobHandle, ExecutorError>;

    /// Requests cooperative cancellation of a previously submitted job.
    ///
    /// Cancellation is cooperative: the executor flips the job's
    /// [`vgn_job_api::context::CancellationToken`], but the handler must call
    /// [`vgn_job_api::context::CancellationToken::check`] at boundaries for
    /// the cancellation to take effect. A handler that ignores the flag will
    /// run to completion; this method does **not** kill threads.
    ///
    /// Idempotent: cancelling an already-cancelled job, or a job that has
    /// already completed but whose entry is still in the executor's table, is
    /// a no-op success.
    ///
    /// # Errors
    ///
    /// [`ExecutorError::UnknownJob`] if no live job matches `job_id`. Note
    /// that a job's entry may be evicted from the cancel table once the
    /// terminal event is delivered, so "the job finished a moment ago" and
    /// "the job never existed" are indistinguishable to the caller.
    fn cancel(&self, job_id: JobId) -> Result<(), ExecutorError>;
}

/// Things that can go wrong during [`Executor::submit`] or
/// [`Executor::cancel`].
///
/// Distinct from [`JobError`]: this enum is the *executor-layer* failure
/// surface (admission, routing, lifecycle), whereas `JobError` is the
/// *handler-layer* failure surface (the job ran but produced an error). A
/// failed handler returns `Ok(JobHandle)` from `submit` and `Err(JobError)` on
/// the result receiver; only failures that prevent the job from running at all
/// land here.
#[derive(Debug, Error)]
pub enum ExecutorError {
    /// No capability handler is registered for `capability_id`. For
    /// `LocalExecutor`, register one via
    /// [`crate::CapabilityRegistry::register`]; for remote executors, route
    /// the envelope to a worker whose
    /// [`vgn_job_api::handshake::WorkerCapabilities`] advertises it.
    #[error("no handler registered for capability {0:?}")]
    NoHandler(CapabilityId),
    /// The envelope's `protocol_version` is not one this executor speaks.
    /// `got` is the value seen on the envelope; `supported` lists the
    /// versions this executor can accept (Phase 1 always has exactly one
    /// entry, [`vgn_job_api::PROTOCOL_VERSION`]).
    #[error("protocol mismatch: envelope={got:?}, supported={supported:?}")]
    ProtocolMismatch { got: u32, supported: Vec<u32> },
    /// `cancel` was called with a `JobId` that the executor has no record
    /// of. May mean the job never existed, or that it completed and the
    /// entry was evicted.
    #[error("job {0} not found")]
    UnknownJob(JobId),
    /// Implementation-specific failure (thread spawn, channel allocation,
    /// transport error, ...). The string is human-readable diagnostic
    /// detail and not machine-parseable.
    #[error("other: {0}")]
    Other(String),
}

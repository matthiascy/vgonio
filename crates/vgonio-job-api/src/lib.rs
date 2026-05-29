//! Versioned job envelope and execution contract for vgonio's distributed
//! executor.
//!
//! This crate defines the wire-stable types that flow between the CLI/adapter,
//! the executor, the transport, and worker capability handlers. It has **no
//! domain dependencies** (no `vgonio-bxdf`, no `vgonio-io`) and **no transport
//! dependencies** (no HTTP / gRPC / WebSocket); both sides of the boundary
//! import this crate.
//!
//! # Roles: executor vs. capability handler
//!
//! Two distinct things in the system both informally "run" a job, and the
//! types in this crate are the contract between them:
//!
//! - **Executor** — the *dispatcher*. Owns "given a [`envelope::JobEnvelope`], *where and how* does
//!   this run?" `LocalExecutor` looks the capability up in an in-process registry and calls the
//!   handler directly on a rayon thread; `RemoteExecutor` (Phase 3) serializes the envelope, ships
//!   it to a worker host, and streams progress back. Same trait surface in both cases. The executor
//!   also owns the cross-cutting middleware: timeouts, deadline enforcement, retries, bridging the
//!   [`progress::JobEvent`] stream to the caller, and the
//!   [`ids::IdempotencyKey`]-keyed dedup cache.
//! - **Capability handler** (also called the *service*) — the *pure-compute* implementation of one
//!   named operation. The canonical handlers are `FittingService` (capability id `"fit"`) and
//!   `MeasurementService` (`"measure-bsdf"` / `"measure-ndf"` / `"measure-msf"` /
//!   `"measure-sdf"`); see [`ids::CapabilityId`] for the full canonical set. A handler takes a typed
//!   request plus a [`context::JobContext`], emits [`progress::JobEvent`]s, calls
//!   [`context::CancellationToken::check`] at boundaries, publishes outputs via
//!   [`context::ArtifactStore::publish`], and returns a typed result. It does **not** know about
//!   transport, CLI printing, the local-vs-remote distinction, or worker process boundaries.
//!
//! The split is load-bearing: the handler code is *byte-identical* whether
//! the job runs in-process under `LocalExecutor` or on a CUDA box across the
//! network under `RemoteExecutor`. Only the [`context::JobContext::artifacts`]
//! implementation and the [`context::JobContext::progress`] transport change.
//! That is the whole reason [`context::JobContext`] is its own type instead of
//! "pass the executor in" — the handler talks to abstractions
//! ([`context::ProgressSender`], [`context::CancellationToken`],
//! `Arc<dyn `[`context::ArtifactStore`]`>`) that the executor wires up
//! differently per dispatch mode.
//!
//! This crate is the type vocabulary both sides share: envelopes,
//! [`ids::JobId`]s, progress events, errors, and the
//! [`context::JobContext`] handle. The executor lives in `vgonio-executor`;
//! the handlers live in `vgonio-fitting` / `vgonio-measurement` (and the
//! backend variants); neither knows about the other except through the types
//! defined here.
//!
//! # Module map
//!
//! | Module | What lives there |
//! |---|---|
//! | [`ids`] | Typed identifiers: `JobId`, `ArtifactId`, `WorkerId`, `WorkerSessionId`, `IdempotencyKey`, `CapabilityId`, `FeatureId`. |
//! | [`envelope`] | [`envelope::JobEnvelope`], the one struct that crosses the wire per submission. |
//! | [`resources`] | Submitter-side resource / feature hints. |
//! | [`handshake`] | Worker-side capability advertisement. |
//! | [`artifact`] | Wire references to input / output blobs (`.vgbsdf`, `.vgms`, `.ior.ron`). |
//! | [`progress`] | Structured progress events emitted by capability handlers. |
//! | [`error`] | Wire error types. |
//! | [`context`] | `JobContext`: what capability handlers receive at runtime. |
//!
//! # Versioning
//!
//! Two independent version numbers travel with every job; see
//! [`envelope::JobEnvelope`] for the rationale.
//!
//! - [`PROTOCOL_VERSION`]: bumped on any schema change to types in this crate.
//! - `capability_version` on each envelope: per-capability payload schema version, advertised by
//!   workers in [`handshake::CapabilityDescriptor::accepted_versions`].

pub mod artifact;
pub mod context;
pub mod envelope;
pub mod error;
pub mod handshake;
pub mod ids;
pub mod progress;
pub mod resources;

/// The wire protocol version implemented by this crate.
///
/// Bump on *any* schema change to types reachable from
/// [`envelope::JobEnvelope`]. Workers advertise this in
/// [`handshake::WorkerCapabilities::protocol_version`]; the scheduler rejects
/// envelopes whose value disagrees.
pub const PROTOCOL_VERSION: u32 = 1;

//! Versioned job envelope and execution contract for vgonio's distributed
//! executor.
//!
//! This crate defines the wire-stable types that flow between the CLI/adapter,
//! the executor, the transport, and worker capability handlers. It has **no
//! domain dependencies** (no `vgonio-bxdf`, no `vgonio-io`) and **no transport
//! dependencies** (no HTTP / gRPC / WebSocket); both sides of the boundary
//! import this crate.
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

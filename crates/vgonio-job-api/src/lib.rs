//! Versioned job envelope and execution contract for vgonio's distributed executor.
//!
//! This crate defines the wire-stable types that flow between the CLI/adapter,
//! the executor, the transport, and worker capability handlers. It has no
//! domain dependencies and no transport dependencies; both sides of the boundary
//! import this crate.

pub mod artifact;
pub mod context;
pub mod envelope;
pub mod error;
pub mod handshake;
pub mod ids;
pub mod progress;
pub mod resources;

/// The wire protocol version implemented by this crate.
pub const PROTOCOL_VERSION: u32 = 1;

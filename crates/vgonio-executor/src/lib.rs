//! Job-execution dispatcher for vgonio.
//!
//! This crate is the **execution layer** that sits between the adapter (CLI /
//! GUI) and the capability handlers (fitting, measurement, …). It owns the
//! [`Executor`] trait (the seam adapters call to submit jobs) and the local
//! in-process implementation ([`LocalExecutor`]). A remote implementation
//! lands in Phase 3 on top of the same trait surface.
//!
//! # Where this crate sits
//!
//! ```text
//! +-----------------+        +-----------------+
//! |  Adapter        |        |  Adapter        |
//! |  (CLI / GUI)    |        |  (CLI / GUI)    |
//! +-----------------+        +-----------------+
//!         | submit(envelope)          | submit(envelope)
//!         v                           v
//! +-----------------+        +-----------------+
//! |  LocalExecutor  |        | RemoteExecutor  |   <- this crate
//! |  (Phase 1)      |        | (Phase 3)       |
//! +-----------------+        +-----------------+
//!         |                           |
//!         | handler(&envelope, ctx)   | POST /v1/jobs ; SSE events
//!         v                           v
//! +-----------------+        +-----------------+
//! | Capability      |        |   Worker host   |
//! | handler in-proc |        | (capability     |
//! |                 |        |  handler there) |
//! +-----------------+        +-----------------+
//! ```
//!
//! Both executors expose the same [`Executor`] trait, so adapter code never
//! branches on local-vs-remote. Capability handlers (in
//! `vgonio-fitting` / `vgonio-measurement` / backend variants) are *byte
//! identical* across modes; only the [`vgn_job_api::context::JobContext`] they
//! receive is wired differently.
//!
//! # Module map
//!
//! | Module | What lives there |
//! |---|---|
//! | [`executor`] | [`Executor`] trait, [`JobHandle`], [`JobOutcome`], [`ExecutorError`]. |
//! | [`registry`] | [`CapabilityRegistry`] + the [`Handler`] closure type used by `LocalExecutor`. |
//! | [`local`] | [`LocalExecutor`]: in-process dispatch on a per-job OS thread. |
//! | [`bridge`] | [`LocalStatusBridge`]: renders a job's [`JobEvent`] stream to the CLI reporter. |
//!
//! [`JobEvent`]: vgn_job_api::progress::JobEvent
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use bytes::Bytes;
//! use vgn_executor::{CapabilityRegistry, Executor, JobOutcome, LocalExecutor};
//! use vgn_job_api::{
//!     envelope::{JobEnvelope, PayloadEncoding, TraceContext},
//!     ids::{CapabilityId, IdempotencyKey},
//!     resources::ResourceHints,
//! };
//!
//! // 1. Register an in-process handler for the "echo" capability. Handlers
//! //    return a `JobOutcome` (payload + any published artifact refs).
//! let mut registry = CapabilityRegistry::new();
//! registry.register(
//!     CapabilityId("echo".into()),
//!     Arc::new(|envelope, _ctx| {
//!         Ok(JobOutcome { payload: envelope.payload.clone(), artifacts: vec![] })
//!     }),
//! );
//!
//! // 2. Build a LocalExecutor backed by an on-disk artifact store.
//! let executor = LocalExecutor::from_cache_dir(registry, "/tmp/vgonio-cache").unwrap();
//!
//! // 3. Submit an envelope and consume the result.
//! let envelope = JobEnvelope::new(
//!     CapabilityId("echo".into()),
//!     1,
//!     PayloadEncoding::Json,
//!     Bytes::from_static(b"hello"),
//!     ResourceHints::default(),
//!     vec![],
//!     IdempotencyKey::from_hash(b"seed"),
//!     TraceContext::default(),
//! );
//! let handle = executor.submit(envelope).unwrap();
//! let outcome = handle.result.recv().unwrap().unwrap();
//! assert_eq!(&outcome.payload[..], b"hello");
//! ```

pub mod bridge;
pub mod executor;
pub mod local;
pub mod registry;

pub use bridge::LocalStatusBridge;
pub use executor::{Executor, ExecutorError, JobHandle, JobOutcome};
pub use local::LocalExecutor;
pub use registry::{CapabilityRegistry, Handler};

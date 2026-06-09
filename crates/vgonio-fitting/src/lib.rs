//! Fitting capability for vgonio: orchestration, request types, and
//! `register_handlers` for the executor registry. Heavy fitting algorithms
//! live in `vgn_bxdf::fitting`; this crate owns the orchestration that
//! drives them per `JobEnvelope`.

use std::sync::Arc;
use vgn_core::config::Config;
use vgn_executor::CapabilityRegistry;

pub mod orchestration;
pub mod request;

/// Registers the `fit` capability with the executor's registry.
/// Mirrors the closure currently in `vgonio-app::app::executor`.
pub fn register_handlers(_reg: &mut CapabilityRegistry, _config: Arc<Config>) {
    todo!("register_handlers for vgonio-fitting");
}

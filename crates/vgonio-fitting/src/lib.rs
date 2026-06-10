//! Fitting capability for vgonio: orchestration, request types, and
//! `register_handlers` for the executor registry. Heavy fitting algorithms
//! live in `vgn_bxdf::fitting`; this crate owns the orchestration that
//! drives them per `JobEnvelope`.
#![feature(adt_const_params)]

use bytes::Bytes;
use std::sync::Arc;
use vgn_core::config::Config;
use vgn_executor::{CapabilityRegistry, JobOutcome};
use vgn_job_api::{
    error::{JobError, JobErrorCode},
    ids::CapabilityId,
};

pub mod mfd;
pub mod orchestration;
pub mod plot;
pub mod request;

/// Registers the `fit` capability with the executor's registry.
/// Mirrors the closure currently in `vgonio-app::app::executor`.
pub fn register_handlers(reg: &mut CapabilityRegistry, config: Arc<Config>) {
    let config_for_fit = Arc::clone(&config);
    reg.register(
        CapabilityId::fit(),
        Arc::new(move |envelope, ctx| {
            let req: request::FitRequest =
                serde_json::from_slice(&envelope.payload).map_err(|e| JobError {
                    code: JobErrorCode::HandlerError,
                    message: format!("Failed to deserialize FitRequest: {e}"),
                    retriable: false,
                    details: None,
                })?;
            let cancel = ctx.cancel.clone();
            orchestration::run(req, Arc::clone(&config_for_fit), ctx)
                // Fit publishes no artifacts; when `--plot` was requested the
                // orchestration returns the plot data, which we serialize into
                // the result payload for the CLI to render client-side (the
                // capability is headless). Empty vec -> empty payload.
                .map(|plots| {
                    let payload = if plots.is_empty() {
                        Bytes::new()
                    } else {
                        serde_json::to_vec(&plots).map(Bytes::from).unwrap_or_else(|e| {
                            log::error!("Failed to serialize fit plot data: {e}");
                            Bytes::new()
                        })
                    };
                    JobOutcome {
                        payload,
                        artifacts: vec![],
                    }
                })
                .map_err(|e| {
                    if cancel.is_cancelled() {
                        JobError {
                            code: JobErrorCode::Cancelled,
                            message: "Job was cancelled".to_string(),
                            retriable: false,
                            details: None,
                        }
                    } else {
                        // The CLI adapter adds its own "Fit job failed:"
                        // context (via `e.message`); the handler stores the
                        // raw cause so the final message is single-wrapped.
                        JobError {
                            code: JobErrorCode::HandlerError,
                            message: e.to_string(),
                            retriable: false,
                            details: None,
                        }
                    }
                })
        }),
    );
}

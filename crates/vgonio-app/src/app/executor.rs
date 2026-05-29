use std::sync::Arc;

use bytes::Bytes;
use vgn_core::config::Config;
use vgn_executor::{CapabilityRegistry, LocalExecutor};
use vgn_job_api::{
    error::{JobError, JobErrorCode},
    ids::CapabilityId,
};

#[cfg(feature = "fitting")]
use crate::orchestration::fitting::FitRequest;
use crate::orchestration::measure::MeasureRequest;

/// Wire ID for the Phase 1 `vgonio measure` batch wrapper. The per-kind
/// `measure-{bsdf,ndf,msf,sdf}` IDs declared in [`CapabilityId`] are the
/// long-term protocol; Phase 1's CLI submits one batched envelope that
/// internally iterates over per-description dispatch. Phase 2 will split
/// per-kind and retire this string.
pub(crate) const MEASURE_BATCH_CAPABILITY: &str = "measure";

pub fn build_local_executor(config: Arc<Config>) -> LocalExecutor {
    // Extract before any handler closure consumes `config`; the closures use
    // `move` and would otherwise leave nothing for the registry-tail call.
    let cache_dir = config.cache_dir().to_path_buf();

    #[allow(unused_mut)]
    let mut reg = CapabilityRegistry::new();

    #[cfg(feature = "fitting")]
    {
        let config_for_fit = Arc::clone(&config);
        reg.register(
            CapabilityId::fit(),
            Arc::new(move |envelope, _ctx| {
                let req: FitRequest =
                    serde_json::from_slice(&envelope.payload).map_err(|e| JobError {
                        code: JobErrorCode::HandlerError,
                        message: format!("Failed to deserialize FitRequest: {e}"),
                        retriable: false,
                        details: None,
                    })?;
                crate::orchestration::fitting::run(req, Arc::clone(&config_for_fit))
                    .map(|_| Bytes::new())
                    .map_err(|e| JobError {
                        // The CLI adapter adds its own "Fit job failed:"
                        // context (via `e.message`); the handler stores the
                        // raw cause so the final message is single-wrapped.
                        code: JobErrorCode::HandlerError,
                        message: e.to_string(),
                        retriable: false,
                        details: None,
                    })
            }),
        );
    }

    {
        let config_for_measure = Arc::clone(&config);
        reg.register(
            CapabilityId(MEASURE_BATCH_CAPABILITY.into()),
            Arc::new(move |envelope, _ctx| {
                let req: MeasureRequest =
                    serde_json::from_slice(&envelope.payload).map_err(|e| JobError {
                        code: JobErrorCode::HandlerError,
                        message: format!("Failed to deserialize MeasureRequest: {e}"),
                        retriable: false,
                        details: None,
                    })?;
                // `LocalExecutor` runs each handler on a fresh OS thread that
                // has no rayon-pool affinity, so any `par_iter` inside the
                // orchestration falls back to the rayon global pool by
                // default. Honor the submitter's `cpu_cores` hint by
                // installing a pool here, around the orchestration call;
                // a missing hint keeps the global-pool default.
                let config_clone = Arc::clone(&config_for_measure);
                let run = move || {
                    crate::orchestration::measure::run(req, config_clone)
                        .map(|_| Bytes::new())
                        .map_err(|e| JobError {
                            code: JobErrorCode::HandlerError,
                            message: e.to_string(),
                            retriable: false,
                            details: None,
                        })
                };
                match envelope.resources.cpu_cores {
                    Some(cores) => {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(cores as usize)
                            .build()
                            .map_err(|e| JobError {
                                code: JobErrorCode::HandlerError,
                                message: format!(
                                    "Failed to create measurement thread pool with {} \
                                     threads: {}",
                                    cores, e
                                ),
                                retriable: false,
                                details: None,
                            })?;
                        pool.install(run)
                    },
                    None => run(),
                }
            }),
        );
    }

    LocalExecutor::from_cache_dir(reg, cache_dir).expect("Failed to create LocalExecutor")
}

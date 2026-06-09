use std::sync::Arc;

use vgn_core::config::Config;
use vgn_executor::{CapabilityRegistry, LocalExecutor};

#[cfg(feature = "fitting")]
use bytes::Bytes;
#[cfg(feature = "fitting")]
use crate::orchestration::fitting::FitRequest;
#[cfg(feature = "fitting")]
use vgn_job_api::{
    error::{JobError, JobErrorCode},
    ids::CapabilityId,
};

/// Wire ID for the `vgonio measure` batch capability. Defined in
/// `vgn_measurement` (alongside the handler) and re-exported here so the CLI
/// adapter (`cmd_measure`) can address the envelope via `executor::`.
pub(crate) use vgn_measurement::MEASURE_BATCH_CAPABILITY;

pub fn build_local_executor(config: Arc<Config>) -> LocalExecutor {
    let cache_dir = config.cache_dir().to_path_buf();

    #[allow(unused_mut)]
    let mut reg = CapabilityRegistry::new();

    #[cfg(feature = "fitting")]
    {
        let config_for_fit = Arc::clone(&config);
        reg.register(
            CapabilityId::fit(),
            Arc::new(move |envelope, ctx| {
                let req: FitRequest =
                    serde_json::from_slice(&envelope.payload).map_err(|e| JobError {
                        code: JobErrorCode::HandlerError,
                        message: format!("Failed to deserialize FitRequest: {e}"),
                        retriable: false,
                        details: None,
                    })?;
                crate::orchestration::fitting::run(req, Arc::clone(&config_for_fit), ctx)
                    .map(|_| Bytes::new())
                    .map_err(|e| JobError {
                        // The CLI adapter adds its own "Fit job failed:" context
                        // (via `e.message`); store the raw cause so the final
                        // message is single-wrapped.
                        code: JobErrorCode::HandlerError,
                        message: e.to_string(),
                        retriable: false,
                        details: None,
                    })
            }),
        );
    }

    // The measure handler + `MEASURE_BATCH_CAPABILITY` live in `vgn_measurement`
    // (DIST Task 2.6); the app just installs them on the registry.
    vgn_measurement::register_handlers(&mut reg, Arc::clone(&config));

    LocalExecutor::from_cache_dir(reg, cache_dir).expect("Failed to create LocalExecutor")
}

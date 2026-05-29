use std::sync::Arc;

#[cfg(feature = "fitting")]
use bytes::Bytes;
use vgn_core::config::Config;
use vgn_executor::{CapabilityRegistry, LocalExecutor};
#[cfg(feature = "fitting")]
use vgn_job_api::{
    error::{JobError, JobErrorCode},
    ids::CapabilityId,
};

#[cfg(feature = "fitting")]
use crate::orchestration::fitting::FitRequest;

pub fn build_local_executor(config: Arc<Config>) -> LocalExecutor {
    // Extract before any handler closure consumes `config`; the closures use
    // `move` and would otherwise leave nothing for the registry-tail call.
    let cache_dir = config.cache_dir().to_path_buf();

    #[allow(unused_mut)]
    let mut reg = CapabilityRegistry::new();

    #[cfg(feature = "fitting")]
    {
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
                crate::orchestration::fitting::run(req, Arc::clone(&config))
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

    // TODO: register measurement handler here once it's implemented. It will
    // also need to capture `Arc::clone(&config)` (cloned BEFORE the closure
    // is created if `config` is still owned here, or via a separate binding
    // if multiple handlers each take their own clone).
    // reg.register(
    //     CapabilityId::measure(),
    //     Arc::new(move |envelope, _ctx| {
    //         todo!(
    //             "Implement the measure capability handler. The handler should deserialize the \
    //              request, run the measurement, and serialize the response."
    //         )
    //     }),
    // );

    LocalExecutor::from_cache_dir(reg, cache_dir).expect("Failed to create LocalExecutor")
}

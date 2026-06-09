use std::sync::Arc;

use vgn_core::config::Config;
use vgn_executor::{CapabilityRegistry, LocalExecutor};
use vgn_measurement::backend::{BsdfRtBackend, GafBackend};

/// Wire ID for the `vgonio measure` batch capability. Defined in
/// `vgn_measurement` (alongside the handler) and re-exported here so the CLI
/// adapter (`cmd_measure`) can address the envelope via `executor::`.
pub(crate) use vgn_measurement::MEASURE_BATCH_CAPABILITY;

pub fn build_local_executor(config: Arc<Config>) -> LocalExecutor {
    let cache_dir = config.cache_dir().to_path_buf();
    let mut reg = CapabilityRegistry::new();

    // Backend selection is app-side per build features; a null
    // fallback is registered when the backend isn't compiled in (its first use
    // panics with an instructive message, caught by the executor).
    let bsdf_backend: Arc<dyn BsdfRtBackend> = {
        #[cfg(feature = "embree")]
        {
            vgn_measurement_embree::provide()
        }
        #[cfg(not(feature = "embree"))]
        {
            vgn_measurement::backend::null_bsdf_backend()
        }
    };
    let gaf_backend: Arc<dyn GafBackend> = {
        #[cfg(feature = "wgpu")]
        {
            vgn_measurement_wgpu::provide()
        }
        #[cfg(not(feature = "wgpu"))]
        {
            vgn_measurement::backend::null_gaf_backend()
        }
    };

    vgn_measurement::register_handlers(&mut reg, Arc::clone(&config), bsdf_backend, gaf_backend);
    #[cfg(feature = "fitting")]
    vgn_fitting::register_handlers(&mut reg, Arc::clone(&config));

    LocalExecutor::from_cache_dir(reg, cache_dir).expect("Failed to create LocalExecutor")
}

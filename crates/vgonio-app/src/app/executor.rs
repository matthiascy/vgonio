use std::sync::Arc;

use vgn_core::config::Config;
use vgn_executor::{CapabilityRegistry, LocalExecutor};

/// Wire ID for the `vgonio measure` batch capability. Defined in
/// `vgn_measurement` (alongside the handler) and re-exported here so the CLI
/// adapter (`cmd_measure`) can address the envelope via `executor::`.
pub(crate) use vgn_measurement::MEASURE_BATCH_CAPABILITY;

pub fn build_local_executor(config: Arc<Config>) -> LocalExecutor {
    let cache_dir = config.cache_dir().to_path_buf();

    let mut reg = CapabilityRegistry::new();

    // The capability handlers live in their respective capability crates; the app just installs
    // them on the registry.
    vgn_measurement::register_handlers(&mut reg, Arc::clone(&config));
    #[cfg(feature = "fitting")]
    vgn_fitting::register_handlers(&mut reg, Arc::clone(&config));

    LocalExecutor::from_cache_dir(reg, cache_dir).expect("Failed to create LocalExecutor")
}

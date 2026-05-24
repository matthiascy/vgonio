//! Shared test setup for integration tests in this crate.

use std::sync::OnceLock;

static INIT: OnceLock<()> = OnceLock::new();

/// Bootstrap the process-wide medium registry exactly once across all parallel
/// test threads. Subsequent calls are no-ops, so it's safe (and recommended)
/// for every `#[test]` to invoke this at its top.
pub fn init_test_registry() {
    INIT.get_or_init(|| {
        let _ = vgn_core::utils::medium::bootstrap(None, None);
    });
}

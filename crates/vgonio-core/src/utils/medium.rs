//! Medium of the surface.

mod error;
pub use error::MediumLoadError;

mod id;
pub use id::MediumId;

pub(crate) mod intern;

mod dto;

mod layer;
pub use layer::{merge_layers, Collision, MergePolicy, Provenance};

mod registry;
pub use registry::{MediumEntry, MediumRegistry};

use crate::error::VgonioError;
use std::str::FromStr;

use std::{path::Path, sync::OnceLock};

static REGISTRY: OnceLock<MediumRegistry> = OnceLock::new();

pub fn registry() -> Option<&'static MediumRegistry> { REGISTRY.get() }

/// Public bootstrap entry. Builds the registry from the embedded fallback plus
/// the (system, user) layer files in order, and installs it as the
/// process-wide registry. Calling twice returns `AlreadyInitialized`.
pub fn bootstrap(sys: Option<&Path>, user: Option<&Path>) -> Result<(), MediumLoadError> {
    let reg = MediumRegistry::build(sys, user)?;
    REGISTRY
        .set(reg)
        .map_err(|_| MediumLoadError::AlreadyInitialized)
}

#[cfg(test)]
mod bootstrap_tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize all tests in this module since they share the global OnceLock.
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn bootstrap_once_succeeds_and_registry_returns_some() {
        let _g = LOCK.lock().unwrap();
        // Pre-bootstrap: registry() may be None or Some depending on test order;
        // we just check that after bootstrap (or after some prior test has run),
        // it is Some.
        let _ = bootstrap(None, None);
        assert!(registry().is_some());
        assert_eq!(registry().unwrap().by_name("al").unwrap().id, MediumId::AL);
    }

    #[test]
    fn bootstrap_twice_returns_already_initialized() {
        let _g = LOCK.lock().unwrap();
        let _ = bootstrap(None, None);
        let err = bootstrap(None, None).unwrap_err();
        assert!(matches!(err, MediumLoadError::AlreadyInitialized));
    }
}

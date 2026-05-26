//! Medium identity and registry.
//!
//! # The shift
//!
//! `Medium` used to be a closed Rust enum (`Vacuum`, `Air`, `Al`, …). Adding a
//! new medium required editing source and recompiling the whole tree, which
//! blocked the IOR fetch tooling from accepting arbitrary
//! refractiveindex.info entries. This module is the data-driven replacement:
//!
//! - **[`MediumId`]** - a `Copy` newtype around a `&'static str` content name (`"vac"`, `"air"`,
//!   `"al"`, ...). Equality and hashing are content-based, so `MediumId::AIR` minted in one crate
//!   compares equal to the same name resolved through the registry in another. The static-string
//!   backing comes from a leak-pooled intern table (see [`intern`]) seeded at bootstrap and
//!   extended as registry entries load - there is no public `MediumId::new`, so a `MediumId` value
//!   is always either a shipped `const` or a successfully registered name.
//!
//! - **[`MediumRegistry`]** - the runtime database of media: canonical name, display name, aliases,
//!   optional IOR-data path. Built from layered sources (embedded fallback → system → user) via
//!   [`merge_layers`]; conflicts are resolved by [`MergePolicy`] and recorded as [`Provenance`].
//!
//! - **[`bootstrap`]** - process-wide install of a `MediumRegistry` behind a `OnceLock`. Subsequent
//!   calls return [`MediumLoadError::AlreadyInitialized`]. Pre-bootstrap, `MediumId::try_from_name`
//!   returns `None` rather than panicking, and `Display` falls back to the canonical short name.
//!
//! # Built-ins
//!
//! Seven constants on [`MediumId`] (`VACUUM`, `AIR`, `AL`, `CU`, `NI`, `PVC`,
//! `CR`) are guaranteed to exist on every binary that links this crate. The
//! IOR loader currently rejects baseline media without IOR data (see
//! `crates/vgonio-core/examples/add_ior.rs` for the shipping pattern).
//!
//! # Wire / file format compatibility
//!
//! `MediumId` serializes as its canonical short name string (intern-pool
//! addresses never cross the wire). The 3-byte medium field in BSDF file
//! headers is forward-compatible, which let the data-driven refactor land
//! without bumping the BSDF format past v0.1.0.

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

use std::{path::Path, sync::OnceLock};

static REGISTRY: OnceLock<MediumRegistry> = OnceLock::new();

/// Returns the process-wide medium registry installed by [`bootstrap`], or
/// `None` if [`bootstrap`] has not yet been called.
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

//! Compatibility shim: the measurement code moved to `vgn_measurement`
//! (DIST Task 2.5). Re-export so existing `crate::measure::*` paths keep
//! resolving while call sites are migrated to `vgn_measurement::*`.
//!
//! TODO(DIST 2.6+): retarget app call sites to `vgn_measurement::*` and delete
//! this shim.

pub use vgn_measurement::{bsdf, cache, io, mfd, orchestration, params, request};
pub use vgn_measurement::measurement::*;
pub use vgn_measurement::request::MeasureRequest;

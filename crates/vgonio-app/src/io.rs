//! Measurement file I/O.
//!
//! The VGMO codec, `legacy_medium` mapping, output-format dispatch
//! (`OutputOptions` / `OutputFileFormatOption`), and the `write_*` helpers
//! moved into `vgn_measurement::io`: `vgonio-app` is downstream
//! of `vgonio-measurement`, and the moved `Measurement` read/write paths call
//! the codec, so it cannot live here without a dependency cycle.
//!
//! This module is now a thin shim re-exporting `vgn_measurement::io::*` so the
//! app's existing `crate::io::*` references keep resolving. The
//! `write_measured_data_to_file` signature changed: it now takes
//! `&vgn_measurement::cache::ComputeCache` instead of the app `Cache` facade.

pub use vgn_measurement::io::*;

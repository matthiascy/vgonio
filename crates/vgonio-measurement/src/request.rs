//! Capability request types for the `measure` capability.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use vgn_core::io::{CompressionScheme, FileEncoding, OutputFormat};

/// Top-level capability request for `vgonio measure`. Mirrors the runtime
/// fields of `MeasureOptions` (app-side, clap-shaped) in a serde-friendly
/// form.
///
/// CLI-shell concerns (`nthreads`, `print_stats`) do not belong on the
/// request: rayon pool sizing is set up at the adapter before submission
/// and `print_stats` is presently unused.
///
/// The `From<&MeasureOptions>` conversion lives app-side in `cmd_measure`,
/// because it touches the clap-shaped `MeasureOptions` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureRequest {
    /// Measurement description files (or directories containing them).
    pub inputs: Vec<PathBuf>,
    /// Output directory. `None` defers to the configuration's default.
    pub output: Option<PathBuf>,
    /// Selects which file format(s) the writer emits.
    pub output_format: OutputFormat,
    /// Resolution for image-shaped outputs (EXR; the Vgbsdf disc grid).
    pub resolution: u32,
    /// On-disk encoding for the Vgmo container.
    pub encoding: FileEncoding,
    /// On-disk compression for the Vgmo container.
    pub compression: CompressionScheme,
}

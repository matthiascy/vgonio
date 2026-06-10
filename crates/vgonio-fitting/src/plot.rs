//! Plot-data hand-off for the fitting capability.
//!
//! The capability runs headless (it may be a remote worker) and so cannot
//! render matplotlib plots itself. When `--plot` is requested, the
//! orchestration collects the data the plots need into [`FitPlotData`] and
//! returns it in the job result payload; the CLI adapter (`cmd_fit`) renders
//! it client-side via the app's `pyplot` (pyo3 + matplotlib). This keeps the
//! capability crate free of any plotting / Python dependency.

use serde::{Deserialize, Serialize};

/// Plot data produced by one brute-force fit invocation (one measured BRDF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitPlotData {
    /// Decimal precision for plot annotations (the brute-force grid precision,
    /// `BrdfFitRequest::brute_precision`).
    pub n_digits: u32,
    /// One error-vs-roughness curve per fitted report (α-sorted ascending).
    pub error_vs_alpha: Vec<ErrorVsAlpha>,
    /// Per-wavelength best-α / error summary; `Some` only for isotropic,
    /// per-wavelength fits.
    pub per_wavelength: Option<PerWavelengthErr>,
}

/// One error-vs-roughness sweep over the brute-force grid, α-sorted ascending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorVsAlpha {
    /// Candidate roughness (α) values.
    pub alpha: Vec<f64>,
    /// Objective-function error at each α (same length as [`Self::alpha`]).
    pub error: Vec<f64>,
}

/// Best-fit roughness and error per wavelength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerWavelengthErr {
    /// Wavelengths, in nanometres.
    pub wavelengths: Vec<f32>,
    /// Best-fit α at each wavelength (same length as [`Self::wavelengths`]).
    pub alphas: Vec<f64>,
    /// Objective-function error at each wavelength.
    pub errors: Vec<f64>,
}

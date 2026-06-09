//! Fitting request types for vgonio.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use vgn_bxdf::{brdf::measured::MeasuredBrdfKind, distro::MicrofacetDistroKind, BrdfFamily};
use vgn_core::{
    error::VgonioError, units::Radians, utils::range::StepRangeIncl, BrdfLevel, ErrorMetric,
    Symmetry, Weighting,
};

/// Top-level capability request for `vgonio fit`.
///
/// The two arms correspond to the two top-level branches of the original
/// `fit` command: NDF fitting and BRDF fitting. They are deliberately
/// separate types — they share no fields and their pipelines diverge
/// immediately, so a single struct with a discriminator would be misleading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitRequest {
    /// Fit a microfacet distribution function directly to NDF data.
    Ndf(NdfFitRequest),
    /// Fit a microfacet (or other) model to measured BRDF data.
    Brdf(BrdfFitRequest),
}

/// NDF-fitting request. The NDF path always uses Nllsq with isotropic
/// symmetry and no weighting (matching the original orchestration); only the
/// inputs and the target distribution are user-configurable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NdfFitRequest {
    /// Measurement files to load and fit.
    pub inputs: Vec<PathBuf>,
    /// Microfacet distribution to fit to.
    pub distro: MicrofacetDistroKind,
}

/// BRDF-fitting request. Holds the runtime contract for the BRDF pipeline.
///
/// Several fields would have been redundant or CLI-shaped on `FitOptions`
/// and are restructured here:
///
/// - `[f64; 3]` roughness triplets become [`StepRangeIncl<f64>`] inside [`Roughness`] (no
///   double-parsing inside the fitter).
/// - Per-wavelength roughness files are read at the boundary; the request carries already-parsed
///   ranges, not paths.
/// - `kind`, `clausen`, `dense`, and `level` collapse into [`BrdfSource`] so that illegal
///   combinations (clausen with non-Vgonio kind, level set for non-Vgonio kind) are
///   unrepresentable.
/// - `output` is resolved at the boundary (the `"auto"` magic value is gone).
/// - `theta_limit` is stored as [`Radians`]; the CLI's degree input is converted once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrdfFitRequest {
    pub inputs: Vec<PathBuf>,
    pub source: BrdfSource,
    pub family: BrdfFamily,
    /// The microfacet distribution being fit. Required by the microfacet
    /// family; carried as `Option` because `FitOptions` allows other
    /// (currently unimplemented) families.
    pub distro: Option<MicrofacetDistroKind>,
    pub symmetry: Symmetry,
    pub roughness: Roughness,
    pub method: FittingMethod,
    /// Brute-force grid precision. Only consulted when [`method`] is
    /// [`FittingMethod::Brute`]; kept here because it pairs with the user's
    /// chosen precision rather than belonging to the method enum (which
    /// also has to be `clap::ValueEnum`).
    ///
    /// [`method`]: BrdfFitRequest::method
    pub brute_precision: u32,
    /// Error metric for the brute path. Even Nllsq consults this during its
    /// internal preliminary brute fit (isotropic case), so it lives at the
    /// top level rather than inside the brute variant.
    pub error_metric: Option<ErrorMetric>,
    pub weighting: Weighting,
    pub theta_limit: Option<Radians>,
    /// Iterate per wavelength. Independent of [`Roughness`] variant: a
    /// per-wavelength loop with a uniform range across wavelengths is a
    /// valid combination (and is what the original code did when
    /// `--per-wl` was set with `--ax/--ay`).
    pub per_wavelength: bool,
    pub plot: bool,
    #[cfg(feature = "cuda")]
    pub cuda: bool,
    /// Already-resolved output path (the `"auto"` magic from `FitOptions`
    /// has been expanded). `None` means "don't write".
    pub output: Option<PathBuf>,
}

/// Origin of the BRDF samples we are fitting to. Selects the loader and,
/// for Vgonio data, the bounce level and optional Clausen-pair resampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrdfSource {
    /// Vgonio-simulated BRDF. `level` selects a bounce subset; setting
    /// `clausen_resample` enables paired-input resampling against the
    /// matching Clausen-format reference, which was the only path that
    /// `--clausen` actually exercised in the original CLI.
    Vgonio {
        level: BrdfLevel,
        clausen_resample: Option<ClausenResample>,
    },
    Clausen,
    Merl,
    Rgl,
    Yan2018,
    Utia,
    /// Explicitly unknown — `run_brdf` reports an error.
    Unknown,
}

/// Tuning for Clausen-pair resampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClausenResample {
    /// Use 4× sample density during resampling.
    pub dense: bool,
}

/// Roughness range specification for the brute-force grid search. Nllsq
/// ignores this and computes its own range (`Default` is the natural
/// choice when method is Nllsq).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Roughness {
    /// No range supplied; the fitter falls back to its internal default
    /// (Nllsq path, or the proxy's "alpha = None" branch).
    Default,
    /// Single isotropic range. With `per_wavelength=true` the loop reuses
    /// this range for each wavelength.
    Iso(StepRangeIncl<f64>),
    /// Single anisotropic range. With `per_wavelength=true` the loop reuses
    /// this range for each wavelength.
    Aniso {
        ax: StepRangeIncl<f64>,
        ay: StepRangeIncl<f64>,
    },
    /// One anisotropic range per wavelength of the source data. Implies
    /// `per_wavelength=true`; the slice length must match the source's
    /// spectrum length (checked at the fit site, not at request build
    /// time, because the spectrum length is not known until the data is
    /// loaded).
    PerWavelengthAniso {
        ax: Box<[StepRangeIncl<f64>]>,
        ay: Box<[StepRangeIncl<f64>]>,
    },
}

impl BrdfFitRequest {
    #[inline]
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    #[inline]
    pub fn on_cpu(&self) -> bool { !self.on_gpu() }

    /// Checks structural invariants that must hold for any source of this
    /// request, not just the CLI path. Re-validating here is what protects
    /// deserialized envelopes (and tests constructing the struct directly)
    /// from silently dropping per-wavelength roughness in the non-per-wl
    /// branch of `brdf_fitting_brute_force`, or from triggering a panic
    /// downstream when `distro` is missing.
    pub fn validate(&self) -> Result<(), VgonioError> {
        if matches!(self.roughness, Roughness::PerWavelengthAniso { .. }) && !self.per_wavelength {
            return Err(VgonioError::new(
                "Roughness::PerWavelengthAniso requires per_wavelength=true.",
                None,
            ));
        }
        if self.source_invokes_fit() && self.distro.is_none() {
            return Err(VgonioError::new(
                "BRDF fitting requires `distro` to be set; brute/Nllsq paths assume a microfacet \
                 distribution and have no fallback.",
                None,
            ));
        }
        Ok(())
    }

    /// Returns whether `run_brdf` will actually invoke the fit pipeline for
    /// this source. `Utia` and `Unknown` short-circuit in `run_brdf` (no
    /// `load_and_fit` call), so their distro is never read; every other
    /// variant flows into `brdf_fitting_brute_force` / `brdf_fitting_nllsq`
    /// where `req.distro.expect(...)` would otherwise panic.
    fn source_invokes_fit(&self) -> bool {
        !matches!(self.source, BrdfSource::Utia | BrdfSource::Unknown)
    }

    /// The user-intended `--kind` value that this request was built from.
    /// Used for the CSV `kind` column so the Vgonio+Clausen resample path
    /// records "Vgonio" rather than the resampled brdf's "Clausen", matching
    /// the original (pre-request) orchestration.
    fn source_kind(&self) -> MeasuredBrdfKind {
        match self.source {
            BrdfSource::Vgonio { .. } => MeasuredBrdfKind::Vgonio,
            BrdfSource::Clausen => MeasuredBrdfKind::Clausen,
            BrdfSource::Merl => MeasuredBrdfKind::Merl,
            BrdfSource::Rgl => MeasuredBrdfKind::Rgl,
            BrdfSource::Yan2018 => MeasuredBrdfKind::Yan2018,
            BrdfSource::Utia => MeasuredBrdfKind::Utia,
            BrdfSource::Unknown => MeasuredBrdfKind::Unknown,
        }
    }
}

impl FitRequest {
    /// See [`BrdfFitRequest::validate`].
    pub fn validate(&self) -> Result<(), VgonioError> {
        match self {
            FitRequest::Ndf(_) => Ok(()),
            FitRequest::Brdf(req) => req.validate(),
        }
    }
}

impl Roughness {
    /// Checks the per-wavelength range count against a known spectrum length.
    /// Called at the fit site (after data load) because the spectrum length
    /// is not known at request build time. Returns Ok for non-per-wavelength
    /// variants -- they don't carry a per-wavelength count to validate.
    pub fn validate_against_spectrum_len(&self, n_wl: usize) -> Result<(), VgonioError> {
        if let Roughness::PerWavelengthAniso { ax, ay } = self {
            if ax.len() != n_wl || ay.len() != n_wl {
                return Err(VgonioError::new(
                    format!(
                        "Per-wavelength roughness range length ({}/{}) does not match source \
                         spectrum length ({}).",
                        ax.len(),
                        ay.len(),
                        n_wl
                    ),
                    None,
                ));
            }
        }
        Ok(())
    }
}

/// Brute-force vs nonlinear-least-squares. Kept as a bare enum (no data
/// variants) so the CLI can derive `clap::ValueEnum` on the same type —
/// `cmd_fit` reuses it for argument parsing. Brute-specific parameters
/// (precision, error metric, roughness range) live on [`BrdfFitRequest`]
/// alongside the method field.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FittingMethod {
    /// Brute force fitting method.
    Brute,
    /// Non-linear least squares fitting method.
    Nllsq,
}

impl TryFrom<&FitOptions> for FitRequest {
    type Error = VgonioError;

    fn try_from(opts: &FitOptions) -> Result<Self, Self::Error> {
        if opts.inputs.is_empty() {
            return Err(VgonioError::new(
                "No input files specified or some files do not exist.",
                None,
            ));
        }

        if opts.ndf {
            let distro = opts.distro.ok_or_else(|| {
                VgonioError::new("NDF fitting requires --distro to be set.", None)
            })?;
            return Ok(FitRequest::Ndf(NdfFitRequest {
                inputs: opts.inputs.clone(),
                distro,
            }));
        }

        if opts.clausen && opts.kind != MeasuredBrdfKind::Vgonio {
            return Err(VgonioError::new(
                "--clausen is only supported with --kind vgonio.",
                None,
            ));
        }

        let source = match opts.kind {
            MeasuredBrdfKind::Vgonio => BrdfSource::Vgonio {
                level: opts.level,
                clausen_resample: opts.clausen.then(|| ClausenResample { dense: opts.dense }),
            },
            MeasuredBrdfKind::Clausen => BrdfSource::Clausen,
            MeasuredBrdfKind::Merl => BrdfSource::Merl,
            MeasuredBrdfKind::Rgl => BrdfSource::Rgl,
            MeasuredBrdfKind::Yan2018 => BrdfSource::Yan2018,
            MeasuredBrdfKind::Utia => BrdfSource::Utia,
            MeasuredBrdfKind::Unknown => BrdfSource::Unknown,
        };

        let roughness = build_roughness(opts)?;
        let theta_limit = opts.theta_limit.map(Radians::from_degrees);
        let output = resolve_output(opts)?;

        Ok(FitRequest::Brdf(BrdfFitRequest {
            inputs: opts.inputs.clone(),
            source,
            family: opts.family,
            distro: opts.distro,
            symmetry: opts.symmetry,
            roughness,
            method: opts.method,
            brute_precision: opts.brute_precision,
            error_metric: opts.error_metric,
            weighting: opts.weighting,
            theta_limit,
            per_wavelength: opts.per_wavelength,
            plot: opts.plot,
            #[cfg(feature = "cuda")]
            cuda: opts.cuda,
            output,
        }))
    }
}

fn build_roughness(opts: &FitOptions) -> Result<Roughness, VgonioError> {
    let to_range = |[s, e, t]: [f64; 3]| StepRangeIncl::new(s, e, t);

    // Per-wavelength anisotropic file pair — clap guarantees both are set
    // together (`requires`) and conflict with the single ax/ay/a inputs.
    if let (Some(ax_path), Some(ay_path)) = (
        opts.per_wavelength_ax.as_ref(),
        opts.per_wavelength_ay.as_ref(),
    ) {
        let ax = read_per_wavelength_roughness_values_unbounded(ax_path).map_err(|e| {
            VgonioError::new(&format!("Failed to read --per-wl-ax file: {}", e), None)
        })?;
        let ay = read_per_wavelength_roughness_values_unbounded(ay_path).map_err(|e| {
            VgonioError::new(&format!("Failed to read --per-wl-ay file: {}", e), None)
        })?;
        if ax.len() != ay.len() {
            return Err(VgonioError::new(
                &format!(
                    "--per-wl-ax has {} entries but --per-wl-ay has {}.",
                    ax.len(),
                    ay.len()
                ),
                None,
            ));
        }
        return Ok(Roughness::PerWavelengthAniso { ax, ay });
    }

    // The TODO from the original code (--per-wl-a) is preserved: there is
    // no isotropic per-wavelength path yet. If a user supplies it, surface
    // a clear error rather than silently producing an empty Roughness.
    if opts.per_wavelength_a.is_some() {
        return Err(VgonioError::new(
            "--per-wl-a is not yet supported (per-wavelength isotropic ranges).",
            None,
        ));
    }

    if let Some(a) = opts.a {
        return Ok(Roughness::Iso(to_range(a)));
    }

    if let (Some(ax), Some(ay)) = (opts.ax, opts.ay) {
        return Ok(Roughness::Aniso {
            ax: to_range(ax),
            ay: to_range(ay),
        });
    }

    Ok(Roughness::Default)
}

fn resolve_output(opts: &FitOptions) -> Result<Option<PathBuf>, VgonioError> {
    let Some(raw) = opts.output.as_ref() else {
        return Ok(None);
    };
    if raw != "auto" {
        return Ok(Some(PathBuf::from(raw)));
    }
    // "auto" magic: build a filename from the first input + error metric +
    // distro + weighting. The original CLI would `unwrap()` and panic here;
    // we promote each missing input to a structured error so a user that
    // asked for output never silently gets none written.
    let first_input = opts.inputs.first().ok_or_else(|| {
        VgonioError::new(
            "--output auto requires at least one input file to derive a stem from.",
            None,
        )
    })?;
    let stem = first_input
        .file_stem()
        .ok_or_else(|| {
            VgonioError::new(
                format!(
                    "--output auto requires the first input ({}) to have a file stem.",
                    first_input.display()
                ),
                None,
            )
        })?
        .to_string_lossy()
        .into_owned();
    let err = opts.error_metric.ok_or_else(|| {
        VgonioError::new(
            "--output auto requires --err to be set so the filename can encode it.",
            None,
        )
    })?;
    let distro = opts.distro.ok_or_else(|| {
        VgonioError::new(
            "--output auto requires --distro to be set so the filename can encode it.",
            None,
        )
    })?;
    Ok(Some(PathBuf::from(format!(
        "{}_{}_{:?}_{:?}.csv",
        stem, err, distro, opts.weighting
    ))))
}

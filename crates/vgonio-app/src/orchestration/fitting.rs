//! Fitting orchestration. CLI shape (clap) stays in
//! [`crate::app::cli::cmd_fit`]; Phase 2 moves this module into a capability
//! crate.
//!
//! The capability contract is [`FitRequest`]. It is built from `FitOptions`
//! via `TryFrom` at the CLI boundary (path resolution, per-wavelength file
//! reads, mode flattening), and is the only thing the runtime sees. CLI
//! artifacts (the `"auto"` output magic, raw `[f64; 3]` triplets, the
//! `ndf`/`clausen` flags, the per-wavelength file paths) are absent from the
//! request by design: This module will be moved into `vgonio-fitting`
//! where those concepts do not exist.
//!
//! # CLI-print inventory
//!
//! Every `cli_*` call here is ORCHESTRATION -- there is no top-level ADAPTER
//! banner to move (unlike [`crate::orchestration::measure`]; `fit` never printed an
//! `Indent::ROOT` "Executing 'vgonio fit'…" line). When Capability Separation
//! moves this module into a capability crate, `vgn_core::cli` is unreachable;
//! Contracts + local executor replaces each call with a `ProgressEvent`.
//! Mapping (see inline `[0.5]`):
//!
//! | Site | Context | -> Phase 1 |
//! |---|---|---|
//! | "Fitting (…) to model: …" | `measured_brdf_fitting` | `ProgressEvent::Step` |
//! | "Fitting with brute force method…" | `brdf_fitting_brute_force` | `ProgressEvent::Step` |
//! | "Pre-allocating GPU memory…" | cuda path | `ProgressEvent::Step` |
//! | "Fitting for wavelength: …" | per-λ brute loop | `ProgressEvent::Step` |
//! | "Took: …" | timing | `ProgressEvent::Note` |
//! | "λ = …:" | per-λ report header | `ProgressEvent::Step` |
//! | "Fitting to distribution @…" | NDF branch in `run` | `ProgressEvent::Step` |
//! | "Fitting to model …@…" | BRDF branch in `run` | `ProgressEvent::Step` |
//! | "Fitting simulated data to Clausen's data." | clausen path | `ProgressEvent::Step` |
//! | "Unknown measured BRDF kind…" | error arm | `ProgressEvent::Error` |
//!
//! Do NOT migrate to `ProgressEvent` in Phase 0 (`vgonio-job-api` does not
//! exist yet). This is the Phase 1 migration checklist.

use crate::{measure::bsdf::BsdfMeasurement, pyplot::plot_err, FitOptions};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use vgn_core::{
    cli::{self, cli_error, cli_note, cli_step, format_duration, Indent},
    config::Config,
    error::VgonioError,
    optics::IorReg,
    units::{Nanometres, Radians, Rads},
    utils::range::StepRangeIncl,
    BrdfLevel, ErrorMetric, Symmetry, Weighting,
};

use crate::{
    app::cache::{Cache, UiCache},
    fitting::{MfdFittingData, MicrofacetDistributionFittingProblem},
    measure::mfd::MeasuredNdfData,
    pyplot::plot_per_wavelength_err,
};
use vgn_bxdf::{
    brdf::{
        measured::{
            merl::MerlBrdf, rgl::RglBrdf, yan::Yan18Brdf, ClausenBrdf, MeasuredBrdfKind, VgonioBrdf,
        },
        AnalyticalBrdf,
    },
    distro::MicrofacetDistroKind,
    fitting::{proxy::BrdfProxy, FittingProblem, FittingReport, Roughness as BxdfRoughness},
    AnyMeasured, AnyMeasuredBrdf, BrdfFamily,
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
#[derive(
    clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
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

pub fn run(req: FitRequest, config: Arc<Config>) -> Result<(), VgonioError> {
    // Re-validate structural invariants here: the request may have arrived
    // via JSON/envelope rather than `TryFrom<&FitOptions>`, and validating
    // at the run boundary is what makes the contract enforceable end-to-end.
    req.validate()?;
    match req {
        FitRequest::Ndf(req) => run_ndf(req, &config),
        FitRequest::Brdf(req) => run_brdf(req, &config),
    }
}

fn run_ndf(req: NdfFitRequest, config: &Config) -> Result<(), VgonioError> {
    let cache = Cache::new(config.cache_dir());
    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    cli_step!(Indent::SECTION, "Fitting to distribution @{:?}", req.distro);
    cache.write(|cache| {
        cache.load_ior_database(&config);
        for input in req.inputs.iter() {
            let handle = cache
                .load_micro_surface_measurement(&config, input)
                .unwrap();
            let measurement = cache.get_measurement(handle).unwrap();
            let ndf = measurement
                .measured
                .downcast_ref::<MeasuredNdfData>()
                .unwrap();
            let problem = MicrofacetDistributionFittingProblem::new(
                MfdFittingData::Ndf(ndf),
                req.distro,
                1.0,
            );
            let report = problem.nllsq_fit(
                req.distro,
                Symmetry::Isotropic,
                Weighting::None,
                StepRangeIncl::new(0.0001, 1.0, 0.001),
                None,
                None,
            );
            report.print_fitting_report(0, 4);
        }
    });
    Ok(())
}

fn run_brdf(req: BrdfFitRequest, config: &Config) -> Result<(), VgonioError> {
    let cache = Cache::new(config.cache_dir());
    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    // Guarded on `source_invokes_fit()` because `validate()` deliberately
    // allows `Utia` / `Unknown` sources to omit `distro` (they short-circuit
    // in the match below). Unwrapping unconditionally would panic on
    // deserialized envelopes for those no-op sources.
    if req.source_invokes_fit() {
        cli_step!(
            Indent::SECTION,
            "Fitting to model {:?}@{:?}",
            req.family,
            req.distro
                .expect("distro: invariant guaranteed by BrdfFitRequest::validate()")
        );
    }
    cache.write(|cache| {
        cache.load_ior_database(&config);
        match &req.source {
            BrdfSource::Vgonio {
                level,
                clausen_resample: Some(resample),
            } => fit_vgonio_clausen_pairs(&req, *level, resample.clone(), cache, &config),
            BrdfSource::Vgonio {
                level: _,
                clausen_resample: None,
            } => {
                // Non-Clausen Vgonio data is stored as a single-level
                // `VgonioBrdf`; the `level` selector only matters for the
                // multi-level `BsdfMeasurement` consumed by the Clausen
                // resample path below.
                load_and_fit::<VgonioBrdf>(&req, cache, &config)
            },
            BrdfSource::Clausen => load_and_fit::<ClausenBrdf>(&req, cache, &config),
            BrdfSource::Merl => load_and_fit::<MerlBrdf>(&req, cache, &config),
            BrdfSource::Rgl => load_and_fit::<RglBrdf>(&req, cache, &config),
            BrdfSource::Yan2018 => load_and_fit::<Yan18Brdf>(&req, cache, &config),
            BrdfSource::Utia => Ok(()),
            BrdfSource::Unknown => {
                // [0.5] orchestration → Phase 1 ProgressEvent::Error
                cli_error!(
                    Indent::SECTION,
                    "Unknown measured BRDF kind specified, cannot fit!"
                );
                Ok(())
            },
        }
    })
}

fn load_and_fit<F: AnyMeasured + AnyMeasuredBrdf + 'static>(
    req: &BrdfFitRequest,
    cache: &mut UiCache,
    config: &Config,
) -> Result<(), VgonioError> {
    for input in &req.inputs {
        let measurement = cache.load_micro_surface_measurement(config, input).unwrap();
        if let Some(brdf) = cache
            .get_measurement(measurement)
            .unwrap()
            .measured
            .downcast_ref::<F>()
        {
            #[cfg(debug_assertions)]
            log::debug!("BRDF incident medium {:?}", brdf.incident_medium());
            measured_brdf_fitting(req, brdf, &cache.iors)?;
        }
    }
    Ok(())
}

fn fit_vgonio_clausen_pairs(
    req: &BrdfFitRequest,
    level: BrdfLevel,
    resample: ClausenResample,
    cache: &mut UiCache,
    config: &Config,
) -> Result<(), VgonioError> {
    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    cli_step!(Indent::SECTION, "Fitting simulated data to Clausen's data.");
    if req.inputs.len() % 2 != 0 {
        return Err(VgonioError::new(
            "The input files should be in pairs of measured data and corresponding Clausen's data.",
            None,
        ));
    }
    for pair in req.inputs.chunks(2) {
        log::debug!("inputs: {:?}, {:?}", pair[0], pair[1]);
        let brdf = {
            let handles = pair
                .iter()
                .map(|p| cache.load_micro_surface_measurement(config, p).unwrap())
                .collect::<Vec<_>>();
            let loaded = handles
                .iter()
                .map(|h| &cache.get_measurement(*h).unwrap().measured)
                .collect::<Vec<_>>();
            if loaded.iter().all(|m| {
                let brdf = m.as_any_brdf(BrdfLevel::L0).unwrap();
                brdf.kind() == MeasuredBrdfKind::Clausen
            }) || loaded
                .iter()
                .all(|m| m.as_any_brdf(BrdfLevel::L0).is_none())
            {
                return Err(VgonioError::new(
                    "The input files should be in pairs of measured data and corresponding \
                     Clausen's data.",
                    None,
                ));
            }
            let simulated_brdf_index = if loaded[0].as_any_brdf(BrdfLevel::L0).unwrap().kind()
                == MeasuredBrdfKind::Clausen
            {
                1
            } else {
                0
            };
            let clausen_brdf_index = simulated_brdf_index ^ 1;
            let simulated_brdf = loaded[simulated_brdf_index]
                .downcast_ref::<BsdfMeasurement>()
                .unwrap();
            let clausen_brdf = loaded[clausen_brdf_index]
                .downcast_ref::<ClausenBrdf>()
                .unwrap();
            log::debug!("Resampling the measured data, dense: {}", resample.dense);
            simulated_brdf.resample(&clausen_brdf.params, level, resample.dense, Rads::ZERO)
        };
        log::debug!("BRDF extraction done, starting fitting.");
        measured_brdf_fitting(req, &brdf, &cache.iors)?;
    }
    Ok(())
}

fn measured_brdf_fitting<F: AnyMeasuredBrdf>(
    req: &BrdfFitRequest,
    brdf: &F,
    iors: &IorReg,
) -> Result<(), VgonioError> {
    let limit = req.theta_limit.unwrap_or(Radians::HALF_PI);
    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    cli_step!(
        Indent::SUBSECTION,
        "Fitting ({:?}) to model: {:?}, distro: {:?}, symmetry: {}, method: {:?}, error metric: \
         {}, weighting: {:?}, θ < {}",
        brdf.kind(),
        req.family,
        req.distro
            .expect("distro: invariant guaranteed by BrdfFitRequest::validate()"),
        req.symmetry,
        req.method,
        if req.method == FittingMethod::Brute {
            req.error_metric.unwrap_or(ErrorMetric::Mse)
        } else {
            ErrorMetric::Nllsq
        },
        req.weighting,
        limit.prettified()
    );

    let mut out = req.output.as_ref().map(|path| {
        let exists = path.exists();
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open(path)
                .expect("Failed to open the output file."),
        );
        if !exists {
            writer
                .write(b"surface,kind,weighting,distro,wavelength,alphax,alphay,error,mse\n")
                .unwrap();
        }
        writer
    });

    match req.method {
        FittingMethod::Brute => brdf_fitting_brute_force(brdf, req, iors, out.as_mut()),
        FittingMethod::Nllsq => {
            brdf_fitting_nllsq(brdf, req, iors, out.as_mut());
            Ok(())
        },
    }
}

// TODO: error handling
/// Read the roughness values from a file.
///
/// The file should contain a list of triplets, each triplet is a range of
/// roughness values for a wavelength. The number of triplets should be equal
/// to `n_wl`. The values are separated by `:` and each triplet is separated
/// by a newline.
///
/// The bounded form is kept as a length-checking helper exercised by the
/// unit tests; the runtime path uses
/// [`read_per_wavelength_roughness_values_unbounded`] (the spectrum length
/// is not known at request-build time) and validates length at the fit
/// site.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn read_per_wavelength_roughness_values(
    path: &Path,
    n_wl: usize,
) -> Result<Box<[StepRangeIncl<f64>]>, &'static str> {
    let values = read_per_wavelength_roughness_values_unbounded(path)?;
    if values.len() != n_wl {
        return Err("The number of triplets should be equal to the number of wavelengths.");
    }
    Ok(values)
}

/// Read all roughness triplets from a file without checking the count.
/// Used at the CLI boundary (in `TryFrom<&FitOptions>`) where the source
/// spectrum length is not yet known; the consumer checks the length.
fn read_per_wavelength_roughness_values_unbounded(
    path: &Path,
) -> Result<Box<[StepRangeIncl<f64>]>, &'static str> {
    let file = File::open(path).map_err(|_| "Failed to open the roughness values file.")?;
    let reader = std::io::BufReader::new(file);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|_| "Failed to read the roughness values file.")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let [start, stop, step] = parse_roughness_values(line)?;
        values.push(StepRangeIncl::new(start, stop, step));
    }
    Ok(values.into_boxed_slice())
}

// TODO: add intermediate fitting results output
fn brdf_fitting_brute_force<F: AnyMeasuredBrdf>(
    brdf: &F,
    req: &BrdfFitRequest,
    iors: &IorReg,
    writer: Option<&mut BufWriter<File>>,
) -> Result<(), VgonioError> {
    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    // TODO [cli-report worthy]: brute force searches a roughness grid
    // (START:END:STEP, possibly thousands of points) after this single banner
    // with no further feedback until the report. Phase 1 should emit a
    // ProgressEvent with percent/ETA across the grid (and per-wavelength when
    // `req.per_wavelength`), not just a start line.
    cli_step!(
        Indent::DETAIL,
        "Fitting with brute force method... {} {}",
        if req.per_wavelength {
            "per wavelength"
        } else {
            ""
        },
        if req.on_cpu() { "on CPU" } else { "on GPU" }
    );
    let start = std::time::Instant::now();
    log::debug!(
        "BRDF proxy created, starting fitting. Number of wavelengths: {}, {:?}",
        brdf.spectrum().len(),
        brdf.spectrum()
    );
    let full_proxy = brdf.proxy(iors);
    let wavelengths = brdf.spectrum();

    let reports = if req.per_wavelength {
        // GPU optimization: Pre-create all wavelength proxies to enable batch
        // GPU memory allocation. This reduces redundant GPU memory transfers
        // by allowing the GPU layer to batch allocate and transfer all
        // wavelength data in one operation instead of per-wavelength transfers.
        #[cfg(feature = "cuda")]
        let wavelength_proxies: Option<Vec<_>> = if req.cuda {
            // [0.5] orchestration → Phase 1 ProgressEvent::Step
            cli_step!(
                Indent::DETAIL,
                "Pre-allocating GPU memory for {} wavelengths...",
                wavelengths.len()
            );
            Some(
                wavelengths
                    .iter()
                    .enumerate()
                    .map(|(i, _)| full_proxy.per_wavelength(i))
                    .collect(),
            )
        } else {
            None
        };

        // Validate per-wavelength range length once, up front. Bubbling this
        // as a structured error matters at the request boundary: a malformed
        // JSON envelope must surface as a `VgonioError`, not a panic that
        // crashes the handler thread.
        req.roughness
            .validate_against_spectrum_len(wavelengths.len())?;

        let mut reports = Box::new_uninit_slice(wavelengths.len());
        for (i, w) in wavelengths.iter().enumerate() {
            let (alpha, ax_str, ay_str) = match &req.roughness {
                Roughness::PerWavelengthAniso { ax, ay } => {
                    let ax_r = ax[i];
                    let ay_r = ay[i];
                    (
                        Some(BxdfRoughness::Anisotropic { ax: ax_r, ay: ay_r }),
                        format_range(ax_r),
                        format_range(ay_r),
                    )
                },
                Roughness::Aniso { ax, ay } => (
                    Some(BxdfRoughness::Anisotropic { ax: *ax, ay: *ay }),
                    format_range(*ax),
                    format_range(*ay),
                ),
                // Iso / Default in a per-wavelength loop matches the original
                // CLI's behavior: it silently fell through to a fit with no
                // anisotropic range (alpha=None).
                Roughness::Iso(_) | Roughness::Default => {
                    (None, "none".to_string(), "none".to_string())
                },
            };

            // [0.5] orchestration → Phase 1 ProgressEvent::Step (per wavelength)
            cli_step!(
                Indent::DETAIL,
                "Fitting for wavelength: {:?}, in range ax: {}, ay: {}",
                w,
                ax_str,
                ay_str
            );

            // Call fitting with pre-allocated proxy (GPU) or create on-demand (CPU)
            #[cfg(feature = "cuda")]
            let report = if let Some(prealloc) = wavelength_proxies.as_ref() {
                brdf_fitting_brute_force_inner(&prealloc[i], req, 0, Some(*w), alpha)
            } else {
                let proxy = full_proxy.per_wavelength(i);
                brdf_fitting_brute_force_inner(&proxy, req, 0, Some(*w), alpha)
            };

            #[cfg(not(feature = "cuda"))]
            let report = {
                let proxy = full_proxy.per_wavelength(i);
                brdf_fitting_brute_force_inner(&proxy, req, 0, Some(*w), alpha)
            };

            reports[i].write((Some(*w), report));
        }
        unsafe { reports.assume_init() }
    } else {
        let alpha = match &req.roughness {
            Roughness::Iso(a) => Some(BxdfRoughness::Isotropic { a: *a }),
            Roughness::Aniso { ax, ay } => Some(BxdfRoughness::Anisotropic { ax: *ax, ay: *ay }),
            Roughness::PerWavelengthAniso { .. } | Roughness::Default => None,
        };
        Box::new([(
            None,
            brdf_fitting_brute_force_inner(&full_proxy, req, 4, None, alpha),
        )])
    };
    let end = std::time::Instant::now();
    // [0.5] orchestration → Phase 1 ProgressEvent::Note
    cli_note!(Indent::SUBSECTION, "Took: {}", format_duration(end - start));

    let surface = req
        .inputs
        .first()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let kind = req.source_kind();
    write_fitting_reports(
        writer,
        surface,
        kind,
        req.weighting,
        req.distro
            .expect("distro: invariant guaranteed by BrdfFitRequest::validate()"),
        &reports,
    );

    if req.plot {
        // Per wavelength fitting
        if req.per_wavelength && req.symmetry.is_isotropic() {
            let wls = brdf
                .spectrum()
                .iter()
                .map(|w| w.as_f32())
                .collect::<Vec<_>>();
            let (alphas, errors): (Vec<_>, Vec<_>) = reports
                .iter()
                .map(|(_, report)| {
                    let best = report.best_model().unwrap();
                    let best_report = report.best_model_report().unwrap();
                    (best.params()[0], best_report.1.objective_fn)
                })
                .unzip();
            plot_per_wavelength_err(
                &wls,
                alphas.as_slice(),
                errors.as_slice(),
                req.brute_precision,
            )
        }

        for (_, report) in reports.iter() {
            let mut alpha_error_pairs = report
                .reports
                .iter()
                .map(|(m, r)| (m.params()[0], r.objective_fn))
                .collect::<Vec<_>>();
            alpha_error_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (alpha, error): (Vec<_>, Vec<_>) = alpha_error_pairs.into_iter().unzip();

            plot_err(error.as_slice(), alpha.as_slice(), req.brute_precision)
                .expect("Failed to plot the error.");
        }
    }

    fn brdf_fitting_brute_force_inner(
        proxy: &BrdfProxy,
        req: &BrdfFitRequest,
        n: usize,
        w: Option<Nanometres>,
        alpha: Option<BxdfRoughness>,
    ) -> FittingReport<Box<dyn AnalyticalBrdf<[f64; 2]>>> {
        let report = proxy.brute_fit(
            req.distro
                .expect("distro: invariant guaranteed by BrdfFitRequest::validate()"),
            req.symmetry,
            req.error_metric.unwrap_or(ErrorMetric::Mse),
            req.weighting,
            req.theta_limit,
            req.theta_limit,
            req.brute_precision,
            #[cfg(feature = "cuda")]
            req.cuda,
            alpha,
        );
        if let Some(w) = w {
            cli::step_inline(6, format_args!("λ = {:?}:", w));
        }
        report.print_fitting_report(n, 6);
        report
    }

    Ok(())
}

fn format_range(r: StepRangeIncl<f64>) -> String {
    format!("{}:{}:{}", r.start, r.stop, r.step_size)
}

fn brdf_fitting_nllsq<F: AnyMeasuredBrdf>(
    brdf: &F,
    req: &BrdfFitRequest,
    iors: &IorReg,
    writer: Option<&mut BufWriter<File>>,
) {
    let full_proxy = brdf.proxy(iors);
    let reports = if req.per_wavelength {
        let mut reports = Box::new_uninit_slice(brdf.spectrum().len());
        let wavelengths = brdf.spectrum();
        for (i, w) in wavelengths.iter().enumerate() {
            let proxy = full_proxy.per_wavelength(i);
            reports[i].write((Some(*w), brdf_fitting_nllsq_inner(&proxy, req, 0, Some(*w))));
        }
        unsafe { reports.assume_init() }
    } else {
        Box::new([(None, brdf_fitting_nllsq_inner(&full_proxy, req, 4, None))])
    };

    let surface = req
        .inputs
        .first()
        .and_then(|p| p.file_stem())
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let kind = req.source_kind();
    write_fitting_reports(
        writer,
        surface,
        kind,
        req.weighting,
        req.distro
            .expect("distro: invariant guaranteed by BrdfFitRequest::validate()"),
        &reports,
    );

    fn brdf_fitting_nllsq_inner(
        proxy: &BrdfProxy,
        req: &BrdfFitRequest,
        n: usize,
        w: Option<Nanometres>,
    ) -> FittingReport<Box<dyn AnalyticalBrdf<[f64; 2]>>> {
        let alpha = match req.symmetry {
            Symmetry::Isotropic => {
                let report = proxy.brute_fit(
                    req.distro
                        .expect("distro: invariant guaranteed by BrdfFitRequest::validate()"),
                    req.symmetry,
                    req.error_metric.unwrap_or(ErrorMetric::Mse),
                    req.weighting,
                    req.theta_limit,
                    req.theta_limit,
                    2,
                    #[cfg(feature = "cuda")]
                    false,
                    None,
                );
                let mid = report.best_model().unwrap().params()[0];
                StepRangeIncl::new(mid - 0.01, mid + 0.1, 0.001)
            },
            Symmetry::Anisotropic => StepRangeIncl::new(0.0, 1.0, 0.02),
        };
        let report = proxy.nllsq_fit(
            req.distro
                .expect("distro: invariant guaranteed by BrdfFitRequest::validate()"),
            req.symmetry,
            req.weighting,
            alpha,
            req.theta_limit,
            req.theta_limit,
        );
        if let Some(w) = w {
            // [0.5] orchestration → Phase 1 ProgressEvent::Step (per-λ report header)
            cli_step!(Indent::DETAIL, "λ = {:?}:", w);
        }
        report.print_fitting_report(n, 6);
        report
    }
}

/// Write a fitting report into a file.
///
/// The output file is a CSV file with the following fields:
///
/// - [0] surface
/// - [1] kind
/// - [2] weighting
/// - [3] distro
/// - [4] wavelength
/// - [5] alphax
/// - [6] alphay
/// - [7] error
/// - [8] mse
fn write_fitting_reports(
    writer: Option<&mut BufWriter<File>>,
    surface: &str,
    kind: MeasuredBrdfKind,
    weighting: Weighting,
    distro: MicrofacetDistroKind,
    reports: &[(
        Option<Nanometres>,
        FittingReport<Box<dyn AnalyticalBrdf<[f64; 2]>>>,
    )],
) {
    if let Some(writer) = writer {
        for (wavelength, report) in reports.iter() {
            let w = match wavelength {
                None => String::from("none"),
                Some(lambda) => lambda.to_string(),
            };

            writer
                .write(
                    format!(
                        "{},{:?},{:?},{:?},{},{},{},{},{}\n",
                        surface,
                        kind,
                        weighting,
                        distro,
                        w,
                        report.best_model().unwrap().params()[0],
                        report.best_model().unwrap().params()[1],
                        report.best_model_report().unwrap().1.objective_fn,
                        report.best_model_report().unwrap().1.mse()
                    )
                    .as_bytes(),
                )
                .unwrap();
        }
    }
}

/// Parse the roughness values from a string formatted as `START:END:STEP`.
pub(crate) fn parse_roughness_values(arg: &str) -> Result<[f64; 3], &'static str> {
    let values = arg
        .split(':')
        .map(|arg| {
            arg.parse::<f64>()
                .map_err(|_| "roughness value is not a valid floating point number")
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "Must provide 3 values separated by ':'")?;
    let [start, stop, step] = values;
    if !(start.is_finite() && stop.is_finite() && step.is_finite()) {
        return Err("roughness values must be finite");
    }
    if step <= 0.0 {
        return Err("roughness step must be greater than 0");
    }
    if start > stop {
        return Err("roughness range start must be <= end");
    }
    Ok([start, stop, step])
}

#[cfg(test)]
mod tests {
    use super::{
        build_roughness, parse_roughness_values, read_per_wavelength_roughness_values,
        read_per_wavelength_roughness_values_unbounded, resolve_output, BrdfSource, FitRequest,
        FittingMethod, Roughness,
    };
    use crate::FitOptions;
    use proptest::prelude::*;
    use std::{io::Write, path::PathBuf};
    use vgn_bxdf::{brdf::measured::MeasuredBrdfKind, distro::MicrofacetDistroKind, BrdfFamily};
    use vgn_core::{utils::range::StepRangeIncl, BrdfLevel, ErrorMetric, Symmetry, Weighting};

    /// Builds a `FitOptions` populated with the defaults clap would produce for
    /// a minimal valid `vgonio fit` invocation. Individual tests mutate only
    /// the fields they care about, so changes in unrelated defaults stay local
    /// to this helper.
    fn default_fit_options() -> FitOptions {
        FitOptions {
            inputs: vec![PathBuf::from("input.vgmo")],
            kind: MeasuredBrdfKind::Vgonio,
            ax: None,
            ay: None,
            per_wavelength_ax: None,
            per_wavelength_ay: None,
            a: None,
            per_wavelength_a: None,
            clausen: false,
            dense: false,
            output: None,
            family: BrdfFamily::Microfacet,
            symmetry: Symmetry::Isotropic,
            distro: Some(MicrofacetDistroKind::TrowbridgeReitz),
            level: BrdfLevel::L0,
            theta_limit: None,
            method: FittingMethod::Brute,
            brute_precision: 6,
            error_metric: Some(ErrorMetric::Mse),
            weighting: Weighting::None,
            per_wavelength: false,
            plot: false,
            #[cfg(feature = "cuda")]
            cuda: false,
            ndf: false,
        }
    }

    fn write_triplets(triplets: &[[f64; 3]]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for [s, e, t] in triplets {
            writeln!(file, "{}:{}:{}", s, e, t).unwrap();
        }
        file.flush().unwrap();
        file
    }

    // ------------------------------------------------------------------
    // parse_roughness_values
    // ------------------------------------------------------------------

    #[test]
    fn parse_roughness_values_accepts_valid_triplet() {
        let parsed = parse_roughness_values("0.1:0.5:0.01").unwrap();
        assert_eq!(parsed, [0.1, 0.5, 0.01]);
    }

    #[test]
    fn parse_roughness_values_rejects_non_numeric_values() {
        let err = parse_roughness_values("a:b:c").unwrap_err();
        assert_eq!(err, "roughness value is not a valid floating point number");
    }

    #[test]
    fn parse_roughness_values_rejects_non_positive_step() {
        let err = parse_roughness_values("0.1:0.5:0.0").unwrap_err();
        assert_eq!(err, "roughness step must be greater than 0");
    }

    #[test]
    fn parse_roughness_values_rejects_negative_step() {
        let err = parse_roughness_values("0.1:0.5:-0.01").unwrap_err();
        assert_eq!(err, "roughness step must be greater than 0");
    }

    #[test]
    fn parse_roughness_values_rejects_reversed_range() {
        let err = parse_roughness_values("0.5:0.1:0.01").unwrap_err();
        assert_eq!(err, "roughness range start must be <= end");
    }

    #[test]
    fn parse_roughness_values_accepts_degenerate_zero_width_range() {
        // start == stop is the boundary of the start <= stop check; covered
        // here explicitly because the brute-force grid degenerates to a single
        // sample, which is still a legitimate request.
        let parsed = parse_roughness_values("0.25:0.25:0.01").unwrap();
        assert_eq!(parsed, [0.25, 0.25, 0.01]);
    }

    #[test]
    fn parse_roughness_values_rejects_wrong_arity() {
        assert!(parse_roughness_values("0.1:0.5").is_err());
        assert!(parse_roughness_values("0.1:0.2:0.3:0.4").is_err());
        assert!(parse_roughness_values("").is_err());
    }

    #[test]
    fn parse_roughness_values_rejects_non_finite() {
        assert_eq!(
            parse_roughness_values("nan:0.5:0.01").unwrap_err(),
            "roughness values must be finite"
        );
        assert_eq!(
            parse_roughness_values("0.1:inf:0.01").unwrap_err(),
            "roughness values must be finite"
        );
        assert_eq!(
            parse_roughness_values("0.1:0.5:inf").unwrap_err(),
            "roughness values must be finite"
        );
    }

    proptest! {
        #[test]
        fn proptest_parse_roughness_values_round_trip(
            start in -1.0e6f64..1.0e6f64,
            extra in 0.0f64..1.0e3f64,
            step in 1.0e-6f64..1.0e3f64,
        ) {
            let stop = start + extra;
            let s = format!("{}:{}:{}", start, stop, step);
            let parsed = parse_roughness_values(&s).unwrap();
            // Use bitwise equality via to_bits because the formatter is
            // round-trip-stable for f64 in Rust.
            prop_assert_eq!(parsed[0].to_bits(), start.to_bits());
            prop_assert_eq!(parsed[1].to_bits(), stop.to_bits());
            prop_assert_eq!(parsed[2].to_bits(), step.to_bits());
        }

        #[test]
        fn proptest_parse_roughness_values_rejects_non_positive_step(
            start in -1.0e3f64..1.0e3f64,
            extra in 0.0f64..1.0e3f64,
            step in -1.0e3f64..=0.0f64,
        ) {
            let stop = start + extra;
            let s = format!("{}:{}:{}", start, stop, step);
            prop_assert_eq!(
                parse_roughness_values(&s).unwrap_err(),
                "roughness step must be greater than 0"
            );
        }

        #[test]
        fn proptest_parse_roughness_values_rejects_reversed_range(
            stop in -1.0e3f64..1.0e3f64,
            extra in 1.0e-9f64..1.0e3f64,
            step in 1.0e-6f64..1.0e3f64,
        ) {
            let start = stop + extra;
            let s = format!("{}:{}:{}", start, stop, step);
            prop_assert_eq!(
                parse_roughness_values(&s).unwrap_err(),
                "roughness range start must be <= end"
            );
        }
    }

    // ------------------------------------------------------------------
    // read_per_wavelength_roughness_values{,_unbounded}
    // ------------------------------------------------------------------

    #[test]
    fn read_per_wavelength_roughness_values_parses_valid_file() {
        let file = write_triplets(&[[0.1, 0.2, 0.01], [0.2, 0.3, 0.01], [0.3, 0.4, 0.01]]);
        let values = read_per_wavelength_roughness_values(file.path(), 3).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0].start, 0.1);
        assert_eq!(values[2].stop, 0.4);
    }

    #[test]
    fn read_per_wavelength_roughness_values_rejects_length_mismatch() {
        let file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let err = read_per_wavelength_roughness_values(file.path(), 2).unwrap_err();
        assert_eq!(
            err,
            "The number of triplets should be equal to the number of wavelengths."
        );
    }

    #[test]
    fn read_per_wavelength_roughness_values_unbounded_skips_blank_lines() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "0.1:0.2:0.01").unwrap();
        writeln!(file).unwrap();
        writeln!(file, "   ").unwrap();
        writeln!(file, "0.3:0.4:0.01").unwrap();
        file.flush().unwrap();

        let values = read_per_wavelength_roughness_values_unbounded(file.path()).unwrap();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn read_per_wavelength_roughness_values_unbounded_surfaces_parse_errors() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "0.1:0.2:0.01").unwrap();
        writeln!(file, "this is not a triplet").unwrap();
        file.flush().unwrap();

        let err = read_per_wavelength_roughness_values_unbounded(file.path()).unwrap_err();
        // Whichever validation triggered first, it should propagate as a
        // `&'static str`, not panic.
        assert!(!err.is_empty());
    }

    #[test]
    fn read_per_wavelength_roughness_values_unbounded_missing_file_errors() {
        let path = PathBuf::from("/this/path/does/not/exist_42.txt");
        let err = read_per_wavelength_roughness_values_unbounded(&path).unwrap_err();
        assert_eq!(err, "Failed to open the roughness values file.");
    }

    proptest! {
        #[test]
        fn proptest_per_wavelength_roughness_round_trip(
            triplets in proptest::collection::vec(
                (-1.0e3f64..1.0e3f64, 0.0f64..1.0e3f64, 1.0e-6f64..1.0e3f64),
                1..16,
            ),
        ) {
            let triplets: Vec<[f64; 3]> = triplets
                .into_iter()
                .map(|(s, extra, step)| [s, s + extra, step])
                .collect();
            let file = write_triplets(&triplets);
            let values = read_per_wavelength_roughness_values(file.path(), triplets.len()).unwrap();
            prop_assert_eq!(values.len(), triplets.len());
            for (got, want) in values.iter().zip(triplets.iter()) {
                prop_assert_eq!(got.start.to_bits(), want[0].to_bits());
                prop_assert_eq!(got.stop.to_bits(), want[1].to_bits());
                prop_assert_eq!(got.step_size.to_bits(), want[2].to_bits());
            }
        }
    }

    // ------------------------------------------------------------------
    // build_roughness
    // ------------------------------------------------------------------

    #[test]
    fn build_roughness_defaults_to_default_variant_when_unspecified() {
        let opts = default_fit_options();
        match build_roughness(&opts).unwrap() {
            Roughness::Default => {},
            other => panic!("expected Roughness::Default, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_isotropic_from_a() {
        let mut opts = default_fit_options();
        opts.a = Some([0.1, 0.5, 0.01]);
        match build_roughness(&opts).unwrap() {
            Roughness::Iso(r) => assert_eq!(r, StepRangeIncl::new(0.1, 0.5, 0.01)),
            other => panic!("expected Roughness::Iso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_anisotropic_from_ax_ay() {
        let mut opts = default_fit_options();
        opts.ax = Some([0.0, 1.0, 0.05]);
        opts.ay = Some([0.1, 0.9, 0.02]);
        match build_roughness(&opts).unwrap() {
            Roughness::Aniso { ax, ay } => {
                assert_eq!(ax, StepRangeIncl::new(0.0, 1.0, 0.05));
                assert_eq!(ay, StepRangeIncl::new(0.1, 0.9, 0.02));
            },
            other => panic!("expected Roughness::Aniso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_prefers_a_over_ax_ay_when_both_set() {
        // clap's `conflicts_with` ordinarily prevents this combination, but the
        // builder must still pick a deterministic precedence (isotropic wins)
        // if it ever sees both — codifies the early `if let Some(a)` branch.
        let mut opts = default_fit_options();
        opts.a = Some([0.2, 0.3, 0.01]);
        opts.ax = Some([0.0, 1.0, 0.05]);
        opts.ay = Some([0.1, 0.9, 0.02]);
        match build_roughness(&opts).unwrap() {
            Roughness::Iso(r) => assert_eq!(r, StepRangeIncl::new(0.2, 0.3, 0.01)),
            other => panic!("expected Roughness::Iso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_per_wavelength_aniso_reads_paired_files() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01], [0.7, 0.8, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());

        match build_roughness(&opts).unwrap() {
            Roughness::PerWavelengthAniso { ax, ay } => {
                assert_eq!(ax.len(), 2);
                assert_eq!(ay.len(), 2);
                assert_eq!(ax[0], StepRangeIncl::new(0.1, 0.2, 0.01));
                assert_eq!(ay[1], StepRangeIncl::new(0.7, 0.8, 0.01));
            },
            other => panic!("expected Roughness::PerWavelengthAniso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_per_wavelength_aniso_rejects_length_mismatch() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());

        let err = build_roughness(&opts).unwrap_err();
        assert!(err.message().contains("--per-wl-ax has 2 entries"));
    }

    #[test]
    fn build_roughness_per_wavelength_a_unsupported() {
        let a_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength_a = Some(a_file.path().to_path_buf());

        let err = build_roughness(&opts).unwrap_err();
        assert!(err.message().contains("--per-wl-a is not yet supported"));
    }

    #[test]
    fn build_roughness_surfaces_io_error_for_missing_per_wavelength_file() {
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(PathBuf::from("/no/such/file_ax.txt"));
        opts.per_wavelength_ay = Some(PathBuf::from("/no/such/file_ay.txt"));

        let err = build_roughness(&opts).unwrap_err();
        assert!(err.message().contains("Failed to read --per-wl-ax file"));
    }

    // ------------------------------------------------------------------
    // resolve_output
    // ------------------------------------------------------------------

    #[test]
    fn resolve_output_none_when_unset() {
        let opts = default_fit_options();
        assert_eq!(resolve_output(&opts).unwrap(), None);
    }

    #[test]
    fn resolve_output_passes_explicit_path_through() {
        let mut opts = default_fit_options();
        opts.output = Some("fits/out.csv".to_string());
        assert_eq!(
            resolve_output(&opts).unwrap(),
            Some(PathBuf::from("fits/out.csv"))
        );
    }

    #[test]
    fn resolve_output_auto_builds_filename_from_inputs_and_metadata() {
        let mut opts = default_fit_options();
        opts.inputs = vec![PathBuf::from("/data/measure_xyz.vgmo")];
        opts.output = Some("auto".to_string());
        opts.error_metric = Some(ErrorMetric::Rmse);
        opts.distro = Some(MicrofacetDistroKind::Beckmann);
        opts.weighting = Weighting::LnCos;
        let out = resolve_output(&opts).unwrap().unwrap();
        // Error metric is rendered via `Display` (lowercase "rmse") to match
        // the pre-request-split filename format that downstream scripts read.
        assert_eq!(out, PathBuf::from("measure_xyz_rmse_Beckmann_LnCos.csv"));
    }

    #[test]
    fn resolve_output_auto_errors_when_inputs_empty() {
        let mut opts = default_fit_options();
        opts.inputs.clear();
        opts.output = Some("auto".to_string());
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("at least one input file"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn resolve_output_auto_errors_when_error_metric_missing() {
        let mut opts = default_fit_options();
        opts.output = Some("auto".to_string());
        opts.error_metric = None;
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("--err"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn resolve_output_auto_errors_when_distro_missing() {
        let mut opts = default_fit_options();
        opts.output = Some("auto".to_string());
        opts.distro = None;
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("--distro"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn resolve_output_auto_errors_when_input_has_no_file_stem() {
        let mut opts = default_fit_options();
        // Bare "/" has no file_stem; auto-derivation must fail loudly.
        opts.inputs = vec![PathBuf::from("/")];
        opts.output = Some("auto".to_string());
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("file stem"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn try_from_propagates_resolve_output_error() {
        // "auto" with no distro must surface as a TryFrom error, not be
        // silently swallowed into output=None.
        let mut opts = default_fit_options();
        opts.output = Some("auto".to_string());
        opts.distro = None;
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("--distro"));
    }

    // ------------------------------------------------------------------
    // TryFrom<&FitOptions> for FitRequest
    // ------------------------------------------------------------------

    #[test]
    fn try_from_rejects_empty_inputs() {
        let mut opts = default_fit_options();
        opts.inputs.clear();
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("No input files"));
    }

    #[test]
    fn try_from_ndf_branch_requires_distro() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        opts.distro = None;
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("NDF fitting requires --distro"));
    }

    #[test]
    fn try_from_ndf_branch_produces_ndf_request() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        opts.distro = Some(MicrofacetDistroKind::TrowbridgeReitz);
        opts.inputs = vec![PathBuf::from("a.vgmo"), PathBuf::from("b.vgmo")];
        match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Ndf(req) => {
                assert_eq!(req.distro, MicrofacetDistroKind::TrowbridgeReitz);
                assert_eq!(req.inputs.len(), 2);
            },
            FitRequest::Brdf(_) => panic!("expected NDF branch"),
        }
    }

    #[test]
    fn try_from_clausen_flag_requires_vgonio_kind() {
        let mut opts = default_fit_options();
        opts.clausen = true;
        opts.kind = MeasuredBrdfKind::Merl;
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("--clausen is only supported"));
    }

    #[test]
    fn try_from_vgonio_kind_without_clausen_produces_plain_vgonio_source() {
        let mut opts = default_fit_options();
        opts.kind = MeasuredBrdfKind::Vgonio;
        opts.level = BrdfLevel::L2;
        opts.clausen = false;
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        match req.source {
            BrdfSource::Vgonio {
                level,
                clausen_resample: None,
            } => assert_eq!(level, BrdfLevel::L2),
            other => panic!("expected Vgonio source, got {:?}", other),
        }
    }

    #[test]
    fn try_from_vgonio_kind_with_clausen_carries_dense_flag() {
        let mut opts = default_fit_options();
        opts.kind = MeasuredBrdfKind::Vgonio;
        opts.clausen = true;
        opts.dense = true;
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        match req.source {
            BrdfSource::Vgonio {
                clausen_resample: Some(resample),
                ..
            } => assert!(resample.dense),
            other => panic!("expected Vgonio with clausen_resample, got {:?}", other),
        }
    }

    #[test]
    fn try_from_maps_each_brdf_kind_to_matching_source() {
        let cases = [
            (MeasuredBrdfKind::Clausen, "Clausen"),
            (MeasuredBrdfKind::Merl, "Merl"),
            (MeasuredBrdfKind::Rgl, "Rgl"),
            (MeasuredBrdfKind::Yan2018, "Yan2018"),
            (MeasuredBrdfKind::Utia, "Utia"),
            (MeasuredBrdfKind::Unknown, "Unknown"),
        ];
        for (kind, name) in cases {
            let mut opts = default_fit_options();
            opts.kind = kind;
            let req = match FitRequest::try_from(&opts).unwrap() {
                FitRequest::Brdf(req) => req,
                FitRequest::Ndf(_) => panic!("expected BRDF branch for {name}"),
            };
            let matched = matches!(
                (&req.source, kind),
                (BrdfSource::Clausen, MeasuredBrdfKind::Clausen)
                    | (BrdfSource::Merl, MeasuredBrdfKind::Merl)
                    | (BrdfSource::Rgl, MeasuredBrdfKind::Rgl)
                    | (BrdfSource::Yan2018, MeasuredBrdfKind::Yan2018)
                    | (BrdfSource::Utia, MeasuredBrdfKind::Utia)
                    | (BrdfSource::Unknown, MeasuredBrdfKind::Unknown)
            );
            assert!(matched, "kind {name} did not map to a matching BrdfSource");
        }
    }

    #[test]
    fn try_from_converts_theta_limit_degrees_to_radians() {
        let mut opts = default_fit_options();
        opts.theta_limit = Some(45.0);
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        let limit = req.theta_limit.expect("theta_limit should be Some");
        let expected = std::f32::consts::FRAC_PI_4;
        assert!(
            (limit.as_f32() - expected).abs() < 1.0e-6,
            "expected ~PI/4, got {}",
            limit.as_f32()
        );
    }

    #[test]
    fn try_from_propagates_brute_precision_and_metric() {
        let mut opts = default_fit_options();
        opts.brute_precision = 9;
        opts.error_metric = Some(ErrorMetric::L2);
        opts.weighting = Weighting::LnCos;
        opts.method = FittingMethod::Nllsq;
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        assert_eq!(req.brute_precision, 9);
        assert_eq!(req.error_metric, Some(ErrorMetric::L2));
        assert_eq!(req.weighting, Weighting::LnCos);
        assert_eq!(req.method, FittingMethod::Nllsq);
    }

    // ------------------------------------------------------------------
    // serde round-trip (envelopes carry FitRequest as JSON)
    // ------------------------------------------------------------------

    fn assert_json_round_trip(req: &FitRequest) {
        let encoded = serde_json::to_string(req).expect("FitRequest should serialize");
        let decoded: FitRequest =
            serde_json::from_str(&encoded).expect("FitRequest should deserialize");
        // Re-encode the decoded value; structural equality is asserted via the
        // re-encoded payload because FitRequest does not derive PartialEq.
        let re_encoded =
            serde_json::to_string(&decoded).expect("decoded FitRequest should serialize");
        assert_eq!(encoded, re_encoded);
    }

    #[test]
    fn serde_round_trip_ndf_request() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    #[test]
    fn serde_round_trip_brdf_request_default_roughness() {
        let opts = default_fit_options();
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    #[test]
    fn serde_round_trip_brdf_request_anisotropic_roughness() {
        let mut opts = default_fit_options();
        opts.ax = Some([0.0, 1.0, 0.05]);
        opts.ay = Some([0.1, 0.9, 0.02]);
        opts.symmetry = Symmetry::Anisotropic;
        opts.theta_limit = Some(60.0);
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    #[test]
    fn serde_round_trip_brdf_request_per_wavelength_aniso() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01], [0.7, 0.8, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    // ------------------------------------------------------------------
    // validate(): structural invariants enforced at the run boundary so that
    // requests arriving via JSON envelopes can't bypass what TryFrom enforces.
    // ------------------------------------------------------------------

    fn brdf_request_from(opts: &FitOptions) -> super::BrdfFitRequest {
        match FitRequest::try_from(opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        }
    }

    #[test]
    fn validate_accepts_request_from_try_from() {
        // Every TryFrom-produced request must satisfy the invariant; this
        // guards against a future TryFrom change silently constructing
        // PerWavelengthAniso with per_wavelength=false.
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01], [0.7, 0.8, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let req = FitRequest::try_from(&opts).unwrap();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn validate_rejects_per_wavelength_aniso_without_per_wavelength_flag() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let mut req = brdf_request_from(&opts);
        // Simulate a malformed deserialized request (or a hand-built one).
        req.per_wavelength = false;
        let err = req.validate().unwrap_err();
        assert!(err.message().contains("requires per_wavelength=true"));
    }

    #[test]
    fn validate_via_top_level_dispatches_to_brdf() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let mut req = brdf_request_from(&opts);
        req.per_wavelength = false;
        let req = FitRequest::Brdf(req);
        assert!(req.validate().is_err());
    }

    #[test]
    fn validate_ndf_is_unconditionally_ok() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        let req = FitRequest::try_from(&opts).unwrap();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn validate_passes_for_default_roughness_with_per_wavelength_false() {
        // Only PerWavelengthAniso is gated; other variants combine freely
        // with either per_wavelength setting.
        let opts = default_fit_options();
        let req = brdf_request_from(&opts);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn validate_rejects_brdf_request_with_fit_invoking_source_and_no_distro() {
        // Every fit-invoking source (anything except Utia/Unknown) must error
        // with the same diagnostic so an operator gets a uniform signal at
        // the request boundary instead of a per-source surprise downstream.
        let fit_sources = [
            BrdfSource::Vgonio {
                level: BrdfLevel::L0,
                clausen_resample: None,
            },
            BrdfSource::Clausen,
            BrdfSource::Merl,
            BrdfSource::Rgl,
            BrdfSource::Yan2018,
        ];
        for source in fit_sources {
            let mut opts = default_fit_options();
            opts.kind = MeasuredBrdfKind::Vgonio;
            let mut req = brdf_request_from(&opts);
            req.source = source;
            req.distro = None;
            let err = req.validate().unwrap_err();
            assert!(
                err.message().contains("`distro` to be set"),
                "expected distro-required error, got: {}",
                err.message()
            );
        }
    }

    #[test]
    fn validate_accepts_no_op_sources_without_distro() {
        // Utia and Unknown short-circuit in run_brdf without ever calling the
        // fit pipeline, so they don't need distro -- requiring it here would
        // over-constrain harmless no-op envelopes.
        for source in [BrdfSource::Utia, BrdfSource::Unknown] {
            let mut opts = default_fit_options();
            let mut req = brdf_request_from(&opts);
            req.source = source;
            req.distro = None;
            assert!(
                req.validate().is_ok(),
                "expected Ok for no-op source, got: {:?}",
                req.validate().unwrap_err().message()
            );
            // suppress unused
            let _ = &mut opts;
        }
    }

    #[test]
    fn validate_rejects_deserialized_brdf_request_missing_distro() {
        // The schema lets `distro: null` round-trip cleanly; validate() is
        // what catches it before fit downstream panics on the expect().
        let mut opts = default_fit_options();
        opts.kind = MeasuredBrdfKind::Merl;
        let req = FitRequest::try_from(&opts).unwrap();
        let encoded = serde_json::to_string(&req).unwrap();
        let mut decoded: FitRequest = serde_json::from_str(&encoded).unwrap();
        if let FitRequest::Brdf(brdf) = &mut decoded {
            brdf.distro = None;
        } else {
            panic!("expected BRDF branch");
        }
        let err = decoded.validate().unwrap_err();
        assert!(err.message().contains("`distro` to be set"));
    }

    #[test]
    fn validate_recovers_invariant_from_round_trip_then_mutation() {
        // Round-trip a valid request, then corrupt the deserialized copy in
        // the same way a hand-crafted JSON envelope might. The corruption
        // round-trips cleanly through JSON (the schema doesn't forbid it),
        // so validate() is the only line of defense.
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let req = FitRequest::try_from(&opts).unwrap();
        let encoded = serde_json::to_string(&req).unwrap();
        let mut decoded: FitRequest = serde_json::from_str(&encoded).unwrap();
        if let FitRequest::Brdf(brdf) = &mut decoded {
            brdf.per_wavelength = false;
        } else {
            panic!("expected BRDF branch");
        }
        assert!(decoded.validate().is_err());
    }

    // ------------------------------------------------------------------
    // Roughness::validate_against_spectrum_len: the only spot the fit path
    // still needs a dynamic check (spectrum length is unknown until data is
    // loaded). The brute-force loop calls this and propagates the error
    // instead of panicking, so we test it directly here.
    // ------------------------------------------------------------------

    #[test]
    fn roughness_length_check_passes_for_non_per_wavelength_variants() {
        assert!(Roughness::Default.validate_against_spectrum_len(0).is_ok());
        assert!(Roughness::Default.validate_against_spectrum_len(5).is_ok());
        assert!(Roughness::Iso(StepRangeIncl::new(0.1, 0.5, 0.01))
            .validate_against_spectrum_len(3)
            .is_ok());
        assert!(Roughness::Aniso {
            ax: StepRangeIncl::new(0.0, 1.0, 0.05),
            ay: StepRangeIncl::new(0.1, 0.9, 0.02),
        }
        .validate_against_spectrum_len(3)
        .is_ok());
    }

    #[test]
    fn roughness_length_check_accepts_matching_per_wavelength() {
        let r = Roughness::PerWavelengthAniso {
            ax: vec![
                StepRangeIncl::new(0.1, 0.2, 0.01),
                StepRangeIncl::new(0.3, 0.4, 0.01),
            ]
            .into_boxed_slice(),
            ay: vec![
                StepRangeIncl::new(0.5, 0.6, 0.01),
                StepRangeIncl::new(0.7, 0.8, 0.01),
            ]
            .into_boxed_slice(),
        };
        assert!(r.validate_against_spectrum_len(2).is_ok());
    }

    #[test]
    fn roughness_length_check_rejects_mismatched_per_wavelength() {
        let r = Roughness::PerWavelengthAniso {
            ax: vec![StepRangeIncl::new(0.1, 0.2, 0.01)].into_boxed_slice(),
            ay: vec![StepRangeIncl::new(0.5, 0.6, 0.01)].into_boxed_slice(),
        };
        let err = r.validate_against_spectrum_len(3).unwrap_err();
        assert!(err.message().contains("does not match"));
        // Both ax and ay lengths plus the spectrum length should appear in
        // the message so the operator can correlate envelope vs data.
        assert!(err.message().contains("1/1"));
        assert!(err.message().contains("(3)"));
    }

    // ------------------------------------------------------------------
    // source_kind(): the CSV `kind` column reflects the user-passed
    // `--kind`, not the runtime brdf type. The Vgonio+Clausen resample
    // path is the regression-pinning row: the resampled brdf is a
    // `ClausenBrdf`, but the CSV must still record "Vgonio".
    // ------------------------------------------------------------------

    #[test]
    fn source_kind_reflects_user_kind_for_each_brdf_source() {
        let cases: &[(MeasuredBrdfKind, bool, MeasuredBrdfKind)] = &[
            // (--kind, --clausen, expected source_kind)
            (MeasuredBrdfKind::Vgonio, false, MeasuredBrdfKind::Vgonio),
            // Resampling against Clausen reference data does NOT change the
            // reported kind: the user asked to fit Vgonio simulation output.
            (MeasuredBrdfKind::Vgonio, true, MeasuredBrdfKind::Vgonio),
            (MeasuredBrdfKind::Clausen, false, MeasuredBrdfKind::Clausen),
            (MeasuredBrdfKind::Merl, false, MeasuredBrdfKind::Merl),
            (MeasuredBrdfKind::Rgl, false, MeasuredBrdfKind::Rgl),
            (MeasuredBrdfKind::Yan2018, false, MeasuredBrdfKind::Yan2018),
            (MeasuredBrdfKind::Utia, false, MeasuredBrdfKind::Utia),
            (MeasuredBrdfKind::Unknown, false, MeasuredBrdfKind::Unknown),
        ];
        for &(kind, clausen, expected) in cases {
            // Paired inputs needed for the clausen-resample TryFrom path:
            // the boundary doesn't actually open them, but the request
            // build itself doesn't enforce the parity check (the chunk-2
            // assertion runs at `run` time). One input is enough here.
            let mut opts = default_fit_options();
            opts.kind = kind;
            opts.clausen = clausen;
            let req = brdf_request_from(&opts);
            assert_eq!(
                req.source_kind(),
                expected,
                "source_kind mismatch for kind={:?} clausen={}",
                kind,
                clausen
            );
        }
    }
}

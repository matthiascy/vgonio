//! Fitting orchestration.

use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufWriter},
    path::Path,
    sync::Arc,
};

use std::io::Write;
use vgn_bxdf::{
    brdf::{
        measured::{rgl::RglBrdf, ClausenBrdf, MeasuredBrdfKind, MerlBrdf, VgonioBrdf, Yan18Brdf},
        AnalyticalBrdf,
    },
    distro::MicrofacetDistroKind,
    fitting::{proxy::BrdfProxy, FittingProblem, FittingReport, Roughness as BxdfRoughness},
    AnyMeasured, AnyMeasuredBrdf,
};
use vgn_core::{
    config::Config,
    error::VgonioError,
    optics::IorReg,
    units::{Nanometres, Radians, Rads},
    utils::range::StepRangeIncl,
    BrdfLevel, ErrorMetric, Symmetry, Weighting,
};
use vgn_job_api::{
    context::JobContext,
    progress::{Activity, PhaseInstanceId, PhaseKind, PhaseOutcome},
};
use vgn_measurement::{bsdf::BsdfMeasurement, cache::ComputeCache, mfd::MeasuredNdfData};

use crate::{
    mfd::{MfdFittingData, MicrofacetDistributionFittingProblem},
    plot::{ErrorVsAlpha, FitPlotData, PerWavelengthErr},
    request::{
        BrdfFitRequest, BrdfSource, ClausenResample, FitRequest, FittingMethod, NdfFitRequest,
        Roughness,
    },
};
pub fn run(
    req: FitRequest,
    config: Arc<Config>,
    ctx: JobContext,
) -> Result<Vec<FitPlotData>, VgonioError> {
    // Re-validate structural invariants here: the request may have arrived
    // via JSON/envelope rather than `TryFrom<&FitOptions>`, and validating
    // at the run boundary is what makes the contract enforceable end-to-end.
    req.validate()?;
    match req {
        FitRequest::Ndf(req) => run_ndf(req, &config, ctx),
        FitRequest::Brdf(req) => run_brdf(req, &config, ctx),
    }
}

fn run_ndf(
    req: NdfFitRequest,
    config: &Config,
    ctx: JobContext,
) -> Result<Vec<FitPlotData>, VgonioError> {
    let mut cache = ComputeCache::new(config.cache_dir());
    let phase = ctx.next_phase_id();
    let started = std::time::Instant::now();
    ctx.progress.emit_activity(Activity::PhaseBegin {
        instance: phase,
        parent: None,
        kind: PhaseKind::fit_ndf(),
        label: format!("Fitting to distribution @{:?}", req.distro),
        at: chrono::Utc::now(),
    });
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
    ctx.progress.emit_activity(Activity::PhaseEnd {
        instance: phase,
        outcome: PhaseOutcome::Ok,
        duration_micros: Some(started.elapsed().as_micros() as u64),
        summary: None,
        at: chrono::Utc::now(),
    });
    // NDF fitting never produces brute-force plot data.
    Ok(vec![])
}

fn run_brdf(
    req: BrdfFitRequest,
    config: &Config,
    ctx: JobContext,
) -> Result<Vec<FitPlotData>, VgonioError> {
    let mut cache = ComputeCache::new(config.cache_dir());
    // The `fit.brdf` phase is opened only for sources that actually fit:
    // `validate()` deliberately allows `Utia` / `Unknown` to omit `distro`
    // (they short-circuit in the match below), so unwrapping it for the
    // banner label would panic on deserialized envelopes for those no-op
    // sources.
    let brdf_phase = ctx.next_phase_id();
    let started = std::time::Instant::now();
    if req.source_invokes_fit() {
        ctx.progress.emit_activity(Activity::PhaseBegin {
            instance: brdf_phase,
            parent: None,
            kind: PhaseKind::fit_brdf(),
            label: format!(
                "Fitting to model {:?}@{:?}",
                req.family,
                req.distro
                    .expect("distro: invariant guaranteed by BrdfFitRequest::validate()")
            ),
            at: chrono::Utc::now(),
        });
    }
    let result = cache.write(|cache| {
        cache.load_ior_database(&config);
        match &req.source {
            BrdfSource::Vgonio {
                level,
                clausen_resample: Some(resample),
            } => fit_vgonio_clausen_pairs(
                &req,
                *level,
                resample.clone(),
                cache,
                &config,
                &ctx,
                brdf_phase,
            ),
            BrdfSource::Vgonio {
                level: _,
                clausen_resample: None,
            } => {
                // Non-Clausen Vgonio data is stored as a single-level
                // `VgonioBrdf`; the `level` selector only matters for the
                // multi-level `BsdfMeasurement` consumed by the Clausen
                // resample path below.
                load_and_fit::<VgonioBrdf>(&req, cache, &config, &ctx, brdf_phase)
            },
            BrdfSource::Clausen => {
                load_and_fit::<ClausenBrdf>(&req, cache, &config, &ctx, brdf_phase)
            },
            BrdfSource::Merl => load_and_fit::<MerlBrdf>(&req, cache, &config, &ctx, brdf_phase),
            BrdfSource::Rgl => load_and_fit::<RglBrdf>(&req, cache, &config, &ctx, brdf_phase),
            BrdfSource::Yan2018 => {
                load_and_fit::<Yan18Brdf>(&req, cache, &config, &ctx, brdf_phase)
            },
            BrdfSource::Utia => Ok(vec![]),
            BrdfSource::Unknown => {
                // `source_invokes_fit()` is false for `Unknown`, so no
                // `fit.brdf` phase was opened above. Open one here so the
                // warning attaches to a real phase, then close it `Skipped`
                // (this arm is a non-fatal no-op, not an error).
                let phase = ctx.next_phase_id();
                ctx.progress.emit_activity(Activity::PhaseBegin {
                    instance: phase,
                    parent: None,
                    kind: PhaseKind::fit_brdf(),
                    label: "Fitting BRDF".into(),
                    at: chrono::Utc::now(),
                });
                ctx.progress.emit_activity(Activity::Warning {
                    phase,
                    text: "Unknown measured BRDF kind specified, cannot fit!".into(),
                });
                ctx.progress.emit_activity(Activity::PhaseEnd {
                    instance: phase,
                    outcome: PhaseOutcome::Skipped {
                        reason: "unknown measured BRDF kind".into(),
                    },
                    duration_micros: None,
                    summary: None,
                    at: chrono::Utc::now(),
                });
                Ok(vec![])
            },
        }
    });
    if req.source_invokes_fit() && result.is_ok() {
        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: brdf_phase,
            outcome: PhaseOutcome::Ok,
            duration_micros: Some(started.elapsed().as_micros() as u64),
            summary: None,
            at: chrono::Utc::now(),
        });
    }
    result
}

fn load_and_fit<F: AnyMeasured + AnyMeasuredBrdf + 'static>(
    req: &BrdfFitRequest,
    cache: &mut ComputeCache,
    config: &Config,
    ctx: &JobContext,
    parent: PhaseInstanceId,
) -> Result<Vec<FitPlotData>, VgonioError> {
    let mut plots = Vec::new();
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
            if let Some(plot) = measured_brdf_fitting(req, brdf, &cache.iors, ctx, parent)? {
                plots.push(plot);
            }
        }
    }
    Ok(plots)
}

fn fit_vgonio_clausen_pairs(
    req: &BrdfFitRequest,
    level: BrdfLevel,
    resample: ClausenResample,
    cache: &mut ComputeCache,
    config: &Config,
    ctx: &JobContext,
    parent: PhaseInstanceId,
) -> Result<Vec<FitPlotData>, VgonioError> {
    let mut plots = Vec::new();
    let phase = ctx.next_phase_id();
    let started = std::time::Instant::now();
    ctx.progress.emit_activity(Activity::PhaseBegin {
        instance: phase,
        parent: Some(parent),
        kind: PhaseKind::fit_brdf_clausen_resample(),
        label: "Fitting simulated data to Clausen's data.".into(),
        at: chrono::Utc::now(),
    });
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
        if let Some(plot) = measured_brdf_fitting(req, &brdf, &cache.iors, ctx, phase)? {
            plots.push(plot);
        }
    }
    ctx.progress.emit_activity(Activity::PhaseEnd {
        instance: phase,
        outcome: PhaseOutcome::Ok,
        duration_micros: Some(started.elapsed().as_micros() as u64),
        summary: None,
        at: chrono::Utc::now(),
    });
    Ok(plots)
}

fn measured_brdf_fitting<F: AnyMeasuredBrdf>(
    req: &BrdfFitRequest,
    brdf: &F,
    iors: &IorReg,
    ctx: &JobContext,
    parent: PhaseInstanceId,
) -> Result<Option<FitPlotData>, VgonioError> {
    let limit = req.theta_limit.unwrap_or(Radians::HALF_PI);
    let phase = ctx.next_phase_id();
    let started = std::time::Instant::now();
    ctx.progress.emit_activity(Activity::PhaseBegin {
        instance: phase,
        parent: Some(parent),
        kind: PhaseKind::fit_brdf_measured(),
        label: format!(
            "Fitting ({:?}) to model: {:?}, distro: {:?}, symmetry: {}, method: {:?}, error \
             metric: {}, weighting: {:?}, θ < {}",
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
        ),
        at: chrono::Utc::now(),
    });

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

    let result = match req.method {
        FittingMethod::Brute => brdf_fitting_brute_force(brdf, req, iors, out.as_mut(), ctx, phase),
        FittingMethod::Nllsq => {
            // Nllsq has no brute-force error-landscape to plot.
            brdf_fitting_nllsq(brdf, req, iors, out.as_mut(), ctx, phase);
            Ok(None)
        },
    };
    if result.is_ok() {
        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: phase,
            outcome: PhaseOutcome::Ok,
            duration_micros: Some(started.elapsed().as_micros() as u64),
            summary: None,
            at: chrono::Utc::now(),
        });
    }
    result
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
pub fn read_per_wavelength_roughness_values(
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
pub fn read_per_wavelength_roughness_values_unbounded(
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
    ctx: &JobContext,
    parent: PhaseInstanceId,
) -> Result<Option<FitPlotData>, VgonioError> {
    // TODO [cli-report worthy]: brute force searches a roughness grid
    // (START:END:STEP, possibly thousands of points) after this single banner
    // with no further feedback until the report. The fitter should emit
    // `Activity::Progress` with percent/ETA across the grid (and per-wavelength
    // when `req.per_wavelength`), not just a start line.
    let phase = ctx.next_phase_id();
    ctx.progress.emit_activity(Activity::PhaseBegin {
        instance: phase,
        parent: Some(parent),
        kind: PhaseKind::fit_brdf_brute_force(),
        label: format!(
            "Fitting with brute force method... {} {}",
            if req.per_wavelength {
                "per wavelength"
            } else {
                ""
            },
            if req.on_cpu() { "on CPU" } else { "on GPU" }
        ),
        at: chrono::Utc::now(),
    });
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
            let prealloc_phase = ctx.next_phase_id();
            let prealloc_started = std::time::Instant::now();
            ctx.progress.emit_activity(Activity::PhaseBegin {
                instance: prealloc_phase,
                parent: Some(phase),
                kind: PhaseKind::fit_brdf_brute_force_gpu_prealloc(),
                label: format!(
                    "Pre-allocating GPU memory for {} wavelengths...",
                    wavelengths.len()
                ),
                at: chrono::Utc::now(),
            });
            let proxies = wavelengths
                .iter()
                .enumerate()
                .map(|(i, _)| full_proxy.per_wavelength(i))
                .collect();
            ctx.progress.emit_activity(Activity::PhaseEnd {
                instance: prealloc_phase,
                outcome: PhaseOutcome::Ok,
                duration_micros: Some(prealloc_started.elapsed().as_micros() as u64),
                summary: None,
                at: chrono::Utc::now(),
            });
            Some(proxies)
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

            ctx.progress.emit_activity(Activity::Message {
                phase,
                level: 0,
                text: format!(
                    "Fitting for wavelength: {:?}, in range ax: {}, ay: {}",
                    w, ax_str, ay_str
                ),
            });

            // Call fitting with pre-allocated proxy (GPU) or create on-demand (CPU)
            #[cfg(feature = "cuda")]
            let report = if let Some(prealloc) = wavelength_proxies.as_ref() {
                brdf_fitting_brute_force_inner(&prealloc[i], req, 0, Some(*w), alpha, ctx, phase)
            } else {
                let proxy = full_proxy.per_wavelength(i);
                brdf_fitting_brute_force_inner(&proxy, req, 0, Some(*w), alpha, ctx, phase)
            };

            #[cfg(not(feature = "cuda"))]
            let report = {
                let proxy = full_proxy.per_wavelength(i);
                brdf_fitting_brute_force_inner(&proxy, req, 0, Some(*w), alpha, ctx, phase)
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
            brdf_fitting_brute_force_inner(&full_proxy, req, 4, None, alpha, ctx, phase),
        )])
    };
    let end = std::time::Instant::now();
    ctx.progress.emit_activity(Activity::PhaseEnd {
        instance: phase,
        outcome: PhaseOutcome::Ok,
        duration_micros: Some((end - start).as_micros() as u64),
        summary: None,
        at: chrono::Utc::now(),
    });

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

    // Collect the data the CLI needs to render plots client-side (the
    // capability is headless and cannot drive matplotlib itself). Returned in
    // the job payload; `cmd_fit` renders it via the app's `pyplot`.
    let plot_data = if req.plot {
        // Per-wavelength best-α / error summary (isotropic per-wavelength only).
        let per_wavelength = if req.per_wavelength && req.symmetry.is_isotropic() {
            let wavelengths = brdf
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
            Some(PerWavelengthErr {
                wavelengths,
                alphas,
                errors,
            })
        } else {
            None
        };

        // One error-vs-α sweep per report, α-sorted ascending.
        let error_vs_alpha = reports
            .iter()
            .map(|(_, report)| {
                let mut pairs = report
                    .reports
                    .iter()
                    .map(|(m, r)| (m.params()[0], r.objective_fn))
                    .collect::<Vec<_>>();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let (alpha, error): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                ErrorVsAlpha { alpha, error }
            })
            .collect::<Vec<_>>();

        Some(FitPlotData {
            n_digits: req.brute_precision,
            error_vs_alpha,
            per_wavelength,
        })
    } else {
        None
    };

    fn brdf_fitting_brute_force_inner(
        proxy: &BrdfProxy,
        req: &BrdfFitRequest,
        n: usize,
        w: Option<Nanometres>,
        alpha: Option<BxdfRoughness>,
        ctx: &JobContext,
        phase: PhaseInstanceId,
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
            ctx.progress.emit_activity(Activity::Message {
                phase,
                level: 0,
                text: format!("λ = {:?}:", w),
            });
        }
        report.print_fitting_report(n, 6);
        report
    }

    Ok(plot_data)
}

fn format_range(r: StepRangeIncl<f64>) -> String {
    format!("{}:{}:{}", r.start, r.stop, r.step_size)
}

fn brdf_fitting_nllsq<F: AnyMeasuredBrdf>(
    brdf: &F,
    req: &BrdfFitRequest,
    iors: &IorReg,
    writer: Option<&mut BufWriter<File>>,
    ctx: &JobContext,
    parent: PhaseInstanceId,
) {
    let phase = ctx.next_phase_id();
    let started = std::time::Instant::now();
    ctx.progress.emit_activity(Activity::PhaseBegin {
        instance: phase,
        parent: Some(parent),
        kind: PhaseKind::fit_brdf_nllsq(),
        label: "Fitting with nonlinear least squares method...".into(),
        at: chrono::Utc::now(),
    });
    let full_proxy = brdf.proxy(iors);
    let reports = if req.per_wavelength {
        let mut reports = Box::new_uninit_slice(brdf.spectrum().len());
        let wavelengths = brdf.spectrum();
        for (i, w) in wavelengths.iter().enumerate() {
            let proxy = full_proxy.per_wavelength(i);
            reports[i].write((
                Some(*w),
                brdf_fitting_nllsq_inner(&proxy, req, 0, Some(*w), ctx, phase),
            ));
        }
        unsafe { reports.assume_init() }
    } else {
        Box::new([(
            None,
            brdf_fitting_nllsq_inner(&full_proxy, req, 4, None, ctx, phase),
        )])
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

    ctx.progress.emit_activity(Activity::PhaseEnd {
        instance: phase,
        outcome: PhaseOutcome::Ok,
        duration_micros: Some(started.elapsed().as_micros() as u64),
        summary: None,
        at: chrono::Utc::now(),
    });

    fn brdf_fitting_nllsq_inner(
        proxy: &BrdfProxy,
        req: &BrdfFitRequest,
        n: usize,
        w: Option<Nanometres>,
        ctx: &JobContext,
        phase: PhaseInstanceId,
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
            ctx.progress.emit_activity(Activity::Message {
                phase,
                level: 0,
                text: format!("λ = {:?}:", w),
            });
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
pub fn parse_roughness_values(arg: &str) -> Result<[f64; 3], &'static str> {
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

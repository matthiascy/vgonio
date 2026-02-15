use crate::{measure::bsdf::BsdfMeasurement, pyplot::plot_err};
use dirs::cache_dir;
use std::{
    fmt::{format, Debug},
    fs::{File, OpenOptions},
    io::{BufRead, BufWriter, Write},
    path::{Path, PathBuf},
};
use vgn_core::{
    cli::{self, cli_error, cli_note, cli_step, format_duration, Indent},
    config::Config,
    error::VgonioError,
    optics::IorReg,
    res::DataStore,
    units::{Radians, Rads},
    utils::range::StepRangeIncl,
    BrdfLevel, ErrorMetric, Symmetry, Weighting,
};

use crate::{
    app::cache::Cache,
    fitting::{MfdFittingData, MicrofacetDistributionFittingProblem},
    measure::mfd::MeasuredNdfData,
    pyplot::plot_per_wavelength_err,
};
use clap::builder::ValueParser;
use vgn_bxdf::{
    brdf::{
        measured::{merl::MerlBrdf, rgl::RglBrdf, yan::Yan18Brdf, ClausenBrdf, MeasuredBrdfKind},
        AnalyticalBrdf,
    },
    distro::MicrofacetDistroKind,
    fitting::{proxy::BrdfProxy, FittingProblem, FittingReport, Roughness},
    AnyMeasuredBrdf, BrdfFamily,
};
use vgn_core::units::Nanometres;

macro_rules! load_and_fit {
    ($brdf:ty, $opts:expr, $cache:expr, $config:expr, $inputs:expr, $theta_limit:expr) => {
        for input in $inputs {
            let measurement = $cache
                .load_micro_surface_measurement(&$config, &input)
                .unwrap();
            if let Some(brdf) = $cache
                .get_measurement(measurement)
                .unwrap()
                .measured
                .downcast_ref::<$brdf>()
            {
                #[cfg(debug_assertions)]
                log::debug!("BRDF incident medium {:?}", brdf.incident_medium());
                measured_brdf_fitting(&$opts, brdf, &$cache.iors, $theta_limit);
            }
        }
    };
    (vgonio $opts:expr, $cache:expr, $config:expr, $inputs:expr, $theta_limit:expr) => {
        for input in $inputs {
            let measurement = $cache
                .load_micro_surface_measurement(&$config, &input)
                .unwrap();
            if let Some(measured) = $cache
                .get_measurement(measurement)
                .unwrap()
                .measured
                .downcast_ref::<BsdfMeasurement>()
            {
                let brdf = measured.brdf_at(BrdfLevel::from($opts.level)).unwrap();
                measured_brdf_fitting(&$opts, brdf, &$cache.iors, $theta_limit);
            }
        }
    };
}

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    if opts.inputs.is_empty() {
        return Err(VgonioError::new(
            "No input files specified or some files do not exist.",
            None,
        ));
    }

    // Load the data from the cache if the fitting is BxDF
    let cache = Cache::new(config.cache_dir());

    // Temporary fix for adding the NDF fitting
    if opts.ndf {
        cli_step!(
            Indent::SECTION,
            "Fitting to distribution @{:?}",
            opts.distro.unwrap()
        );
        // Load the data from the cache if the fitting is NDF
        cache.write(|cache| {
            cache.load_ior_database(&config);
            for input in opts.inputs.iter() {
                let handle = cache
                    .load_micro_surface_measurement(&config, input)
                    .unwrap();
                let measurement = cache.get_measurement(handle).unwrap();
                let ndf = measurement
                    .measured
                    .downcast_ref::<MeasuredNdfData>()
                    .unwrap();
                let model = opts.distro.unwrap();
                let problem =
                    MicrofacetDistributionFittingProblem::new(MfdFittingData::Ndf(ndf), model, 1.0);
                let report = problem.nllsq_fit(
                    model,
                    Symmetry::Isotropic,
                    Weighting::None,
                    StepRangeIncl::new(0.0001, 1.0, 0.001),
                    None,
                    None,
                );
                report.print_fitting_report(0, 4);
            }
        });

        return Ok(());
    }

    cli_step!(
        Indent::SECTION,
        "Fitting to model {:?}@{:?}",
        opts.family,
        opts.distro.unwrap()
    );
    let theta_limit = opts
        .theta_limit
        .and_then(|t| Some(Radians::from_degrees(t)));
    cache.write(|cache| {
        cache.load_ior_database(&config);
        if opts.kind == MeasuredBrdfKind::Vgonio && opts.clausen {
            cli_step!(Indent::SECTION, "Fitting simulated data to Clausen's data.");
            if opts.inputs.len() % 2 != 0 {
                return Err(VgonioError::new(
                    "The input files should be in pairs of measured data and corresponding \
                     Clausen's data.",
                    None,
                ));
            }
            for pair in opts.inputs.chunks(2) {
                log::debug!("inputs: {:?}, {:?}", pair[0], pair[1]);
                let brdf = {
                    let handles = pair
                        .iter()
                        .map(|p| cache.load_micro_surface_measurement(&config, p).unwrap())
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
                            "The input files should be in pairs of measured data and \
                             corresponding Clausen's data.",
                            None,
                        ));
                    }
                    let simulated_brdf_index =
                        if loaded[0].as_any_brdf(BrdfLevel::L0).unwrap().kind()
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
                    log::debug!("Resampling the measured data, dense: {}", opts.dense);
                    simulated_brdf.resample(
                        &clausen_brdf.params,
                        opts.level,
                        opts.dense,
                        Rads::ZERO,
                    )
                };
                log::debug!("BRDF extraction done, starting fitting.");
                measured_brdf_fitting(&opts, &brdf, &cache.iors, theta_limit);
            }
        } else {
            match opts.kind {
                MeasuredBrdfKind::Clausen => {
                    load_and_fit!(ClausenBrdf, opts, cache, config, &opts.inputs, theta_limit);
                },
                MeasuredBrdfKind::Merl => {
                    load_and_fit!(MerlBrdf, opts, cache, config, &opts.inputs, theta_limit);
                },
                MeasuredBrdfKind::Utia => {},
                MeasuredBrdfKind::Vgonio => {
                    load_and_fit!(vgonio opts, cache, config, &opts.inputs, theta_limit);
                },
                MeasuredBrdfKind::Yan2018 => {
                    load_and_fit!(Yan18Brdf, opts, cache, config, &opts.inputs, theta_limit);
                },
                MeasuredBrdfKind::Rgl => {
                    load_and_fit!(RglBrdf, opts, cache, config, &opts.inputs, theta_limit);
                },
                MeasuredBrdfKind::Unknown => {
                    cli_error!(
                        Indent::SECTION,
                        "Unknown measured BRDF kind specified, cannot fit!"
                    );
                },
            }
        }
        Ok(())
    })
}

// TODO: error handling
/// Read the roughness values from a file.
/// The file should contain a list of triplets, each triplet is a range of
/// roughness values for a wavelength. The number of triplets should be equal to
/// the number of wavelengths. The values are separated by `:` and each triplet
/// is separated by a newline.
fn read_per_wavelength_roughness_values(
    path: &Path,
    n_wl: usize,
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
    if values.len() != n_wl {
        return Err("The number of triplets should be equal to the number of wavelengths.");
    }
    Ok(values.into_boxed_slice())
}

// TODO: add intermediate fitting results output, and add error handling
fn brdf_fitting_brute_force<F: AnyMeasuredBrdf>(
    brdf: &F,
    opts: &FitOptions,
    iors: &IorReg,
    writer: Option<&mut BufWriter<File>>,
) {
    cli_step!(
        Indent::DETAIL,
        "Fitting with brute force method... {} {}",
        if opts.per_wavelength {
            "per wavelength"
        } else {
            ""
        },
        if opts.on_cpu() { "on CPU" } else { "on GPU" }
    );
    let start = std::time::Instant::now();
    log::debug!(
        "BRDF proxy created, starting fitting. Number of wavelengths: {}, {:?}",
        brdf.spectrum().len(),
        brdf.spectrum()
    );
    let full_proxy = brdf.proxy(iors);

    let reports = if opts.per_wavelength {
        // Use Box<[MaybeUninit<_>]> to store the reports for each wavelength to avoid
        // unnecessary initialization, since the number of wavelengths can be large
        // and the fitting process can be time-consuming.
        let mut reports = Box::new_uninit_slice(brdf.spectrum().len());
        let wavelengths = brdf.spectrum();
        // Read the roughness values from the file provided by
        // --per-wl-ax and --per-wl-ay
        let per_wl_ax = opts.per_wavelength_ax.as_ref().map(|path| {
            read_per_wavelength_roughness_values(path, wavelengths.len())
                .expect("Failed to read the roughness values for the anisotropic roughness in x.")
        });
        let per_wl_ay = opts.per_wavelength_ay.as_ref().map(|path| {
            read_per_wavelength_roughness_values(path, wavelengths.len())
                .expect("Failed to read the roughness values for the anisotropic roughness in y.")
        });
        let ax = opts
            .ax
            .as_ref()
            .map(|[start, end, step]| StepRangeIncl::new(*start, *end, *step));
        let ay = opts
            .ay
            .as_ref()
            .map(|[start, end, step]| StepRangeIncl::new(*start, *end, *step));

        // GPU optimization: Pre-create all wavelength proxies to enable batch GPU memory allocation
        // This reduces redundant GPU memory transfers by allowing the GPU layer to batch allocate
        // and transfer all wavelength data in one operation instead of per-wavelength transfers.
        #[cfg(feature = "cuda")]
        let wavelength_proxies: Option<Vec<_>> = if opts.cuda {
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

        for (i, w) in wavelengths.iter().enumerate() {
            // Use pre-allocated proxy on GPU to avoid redundant memory transfers,
            // or create on-demand for CPU processing
            #[cfg(feature = "cuda")]
            let use_prealloc = wavelength_proxies.is_some();

            let ax = if per_wl_ax.is_none() {
                ax
            } else {
                per_wl_ax.as_ref().map(|ax| ax[i])
            };
            let ay = if per_wl_ay.is_none() {
                ay
            } else {
                per_wl_ay.as_ref().map(|ay| ay[i])
            };
            let ax_str = if ax.is_none() {
                "none"
            } else {
                &format!(
                    "{}:{}:{}",
                    ax.unwrap().start,
                    ax.unwrap().stop,
                    ax.unwrap().step_size
                )
            };
            let ay_str = if ay.is_none() {
                "none"
            } else {
                &format!(
                    "{}:{}:{}",
                    ay.unwrap().start,
                    ay.unwrap().stop,
                    ay.unwrap().step_size
                )
            };
            cli_step!(
                Indent::DETAIL,
                "Fitting for wavelength: {:?}, in range ax: {}, ay: {}",
                w,
                ax_str,
                ay_str
            );
            let a = ax.zip(ay).map(|(ax, ay)| Roughness::Anisotropic { ax, ay });

            // Call fitting with pre-allocated proxy (GPU) or create on-demand (CPU)
            #[cfg(feature = "cuda")]
            let report = if use_prealloc {
                brdf_fitting_brute_force_inner(
                    &wavelength_proxies.as_ref().unwrap()[i],
                    opts,
                    0,
                    Some(*w),
                    a,
                )
            } else {
                let proxy = full_proxy.per_wavelength(i);
                brdf_fitting_brute_force_inner(&proxy, opts, 0, Some(*w), a)
            };

            #[cfg(not(feature = "cuda"))]
            let report = {
                let proxy = full_proxy.per_wavelength(i);
                brdf_fitting_brute_force_inner(&proxy, opts, 0, Some(*w), a)
            };

            reports[i].write((Some(*w), report));
        }
        unsafe { reports.assume_init() }
    } else {
        // For the non-per-wavelength fitting, we only need to fit once, so we can
        // directly compute the roughness range without reading from a file.
        let alpha = match opts.symmetry {
            Symmetry::Isotropic => opts.a.map(|a| Roughness::Isotropic {
                a: StepRangeIncl::new(a[0], a[1], a[2]),
            }),
            Symmetry::Anisotropic => opts.ax.zip(opts.ay).map(|(ax, ay)| Roughness::Anisotropic {
                ax: StepRangeIncl::new(ax[0], ax[1], ax[2]),
                ay: StepRangeIncl::new(ay[0], ay[1], ay[2]),
            }),
        };
        Box::new([(
            None,
            brdf_fitting_brute_force_inner(&full_proxy, opts, 4, None, alpha),
        )])
    };
    let end = std::time::Instant::now();
    cli_note!(Indent::SUBSECTION, "Took: {}", format_duration(end - start));

    write_fitting_reports(
        writer,
        opts.inputs[0].file_stem().unwrap().to_str().unwrap(),
        opts.kind,
        opts.weighting,
        opts.distro.unwrap(),
        &reports,
    );

    if opts.plot {
        // Per wavelength fitting
        if opts.per_wavelength && opts.symmetry.is_isotropic() {
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
            plot_per_wavelength_err(
                &wavelengths,
                alphas.as_slice(),
                errors.as_slice(),
                opts.brute_precision,
            )
        }

        for (_, report) in reports.iter() {
            let mut alpha_error_pairs = report
                .reports
                .iter()
                .map(|(m, r)| (m.params()[0], r.objective_fn))
                .collect::<Vec<_>>();
            // Sort the error by alpha value for later plotting
            alpha_error_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (alpha, error): (Vec<_>, Vec<_>) = alpha_error_pairs.into_iter().unzip();

            plot_err(error.as_slice(), alpha.as_slice(), opts.brute_precision)
                .expect("Failed to plot the error.");
        }
    }

    fn brdf_fitting_brute_force_inner(
        proxy: &BrdfProxy,
        opts: &FitOptions,
        n: usize,
        w: Option<Nanometres>,
        alpha: Option<Roughness>,
    ) -> FittingReport<Box<dyn AnalyticalBrdf<[f64; 2]>>> {
        let report = proxy.brute_fit(
            opts.distro.unwrap(),
            opts.symmetry,
            opts.error_metric.unwrap_or(ErrorMetric::Mse),
            opts.weighting,
            opts.theta_limit.map(|t| Radians::from_degrees(t)),
            opts.theta_limit.map(|t| Radians::from_degrees(t)),
            opts.brute_precision,
            #[cfg(feature = "cuda")]
            opts.cuda,
            alpha,
        );
        // Print the fitting report
        if let Some(w) = w {
            cli::step_inline(6, format_args!("λ = {:?}:", w));
        }
        report.print_fitting_report(n, 6);
        report
    }
}

fn brdf_fitting_nllsq<F: AnyMeasuredBrdf>(
    brdf: &F,
    opts: &FitOptions,
    iors: &IorReg,
    writer: Option<&mut BufWriter<File>>,
) {
    let full_proxy = brdf.proxy(iors);
    let reports = if opts.per_wavelength {
        let mut reports = Box::new_uninit_slice(brdf.spectrum().len());
        let wavelengths = brdf.spectrum();
        for (i, w) in wavelengths.iter().enumerate() {
            let proxy = full_proxy.per_wavelength(i);
            reports[i].write((
                Some(*w),
                brdf_fitting_nllsq_inner(&proxy, opts, 0, Some(*w)),
            ));
        }
        unsafe { reports.assume_init() }
    } else {
        Box::new([(None, brdf_fitting_nllsq_inner(&full_proxy, opts, 4, None))])
    };

    write_fitting_reports(
        writer,
        opts.inputs[0].file_stem().unwrap().to_str().unwrap(),
        opts.kind,
        opts.weighting,
        opts.distro.unwrap(),
        &reports,
    );

    fn brdf_fitting_nllsq_inner(
        proxy: &BrdfProxy,
        opts: &FitOptions,
        n: usize,
        w: Option<Nanometres>,
    ) -> FittingReport<Box<dyn AnalyticalBrdf<[f64; 2]>>> {
        // Adjust the alpha range only if the model is isotropic
        let alpha = match opts.symmetry {
            Symmetry::Isotropic => {
                let report = proxy.brute_fit(
                    opts.distro.unwrap(),
                    opts.symmetry,
                    opts.error_metric.unwrap_or(ErrorMetric::Mse),
                    opts.weighting,
                    opts.theta_limit.map(|t| Radians::from_degrees(t)),
                    opts.theta_limit.map(|t| Radians::from_degrees(t)),
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
            opts.distro.unwrap(),
            opts.symmetry,
            opts.weighting,
            alpha,
            opts.theta_limit.map(|t| Radians::from_degrees(t)),
            opts.theta_limit.map(|t| Radians::from_degrees(t)),
        );
        if let Some(w) = w {
            cli_step!(Indent::DETAIL, "λ = {:?}:", w);
        }
        report.print_fitting_report(n, 6);
        report
    }
}

fn measured_brdf_fitting<F: AnyMeasuredBrdf>(
    opts: &FitOptions,
    brdf: &F,
    iors: &IorReg,
    theta_limit: Option<Radians>,
) {
    let limit = theta_limit.unwrap_or(Radians::HALF_PI);
    cli_step!(
        Indent::SUBSECTION,
        "Fitting ({:?}) to model: {:?}, distro: {:?}, symmetry: {}, method: {:?}, error metric: \
         {}, weighting: {:?}, θ < {}",
        brdf.kind(),
        opts.family,
        opts.distro,
        opts.symmetry,
        opts.method,
        if opts.method == FittingMethod::Brute {
            opts.error_metric.unwrap_or(ErrorMetric::Mse)
        } else {
            ErrorMetric::Nllsq
        },
        opts.weighting,
        limit.prettified()
    );

    let mut out = opts.output.as_ref().and_then(|output| {
        let filepath = if output == "auto" {
            format!(
                "{}_{}_{:?}_{:?}.csv",
                opts.inputs[0].file_stem().unwrap().display(),
                opts.error_metric.unwrap(),
                opts.distro.unwrap(),
                opts.weighting,
            )
        } else {
            output.clone()
        };

        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open(&filepath)
                .expect("Failed to open the output file."),
        );

        if !Path::new(&filepath).exists() {
            writer
                .write(b"surface,kind,weighting,distro,wavelength,alphax,alphay,error,mse\n")
                .unwrap();
        }

        Some(writer)
    });

    match opts.method {
        FittingMethod::Brute => brdf_fitting_brute_force(brdf, opts, iors, out.as_mut()),
        FittingMethod::Nllsq => brdf_fitting_nllsq(brdf, opts, iors, out.as_mut()),
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
fn parse_roughness_values(arg: &str) -> Result<[f64; 3], &'static str> {
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

// TODO: separate the NDF fitting & the BRDF fitting

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(num_args = 1.., value_delimiter = ' ', help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

    #[clap(long, short, help = "Kind of the measured BRDF data.")]
    pub kind: MeasuredBrdfKind,

    #[clap(
        long = "ax",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(parse_roughness_values),
        requires("ay"),
        help = "Anisotropic roughness in x direction (inclusive)."
    )]
    pub ax: Option<[f64; 3]>,

    #[clap(
        long = "ay",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(parse_roughness_values),
        requires("ax"),
        help = "Anisotropic roughness in y direction (inclusive)."
    )]
    pub ay: Option<[f64; 3]>,

    #[clap(
        long = "per-wl-ax",
        requires("per_wavelength_ay"),
        requires("per_wavelength"),
        conflicts_with_all(&["ax", "a", "per_wavelength_a"]),
        help = "Anisotropic roughness in y direction for each wavelength. The values are \
                specified as a list of triplets, each triplet is a range of roughness values for \
                a wavelength. The number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_ax: Option<PathBuf>,

    #[clap(
        long = "per-wl-ay",
        requires("per_wavelength_ax"),
        requires("per_wavelength"),
        conflicts_with_all(&["ay", "a", "per_wavelength_a"]),
        help = "Anisotropic roughness in y direction for each wavelength. The values are \
                specified as a list of triplets, each triplet is a range of roughness values for \
                a wavelength. The number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_ay: Option<PathBuf>,

    #[clap(
        long = "a",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(parse_roughness_values),
        conflicts_with_all(&["ax", "ay", "per_wavelength_ax", "per_wavelength_ay", "per_wavelength_a"]),
        help = "Isotropic roughness (inclusive)."
    )]
    pub a: Option<[f64; 3]>,

    // TODO: Add support for per-wavelength isotropic roughness
    #[clap(
        long = "per-wl-a",
        conflicts_with_all(&["per_wavelength_ax", "per_wavelength_ay", "ax", "ay", "a"]),
        help = "Isotropic roughness for each wavelength. The values are specified as a list of \
                triplets, each triplet is a range of roughness values for a wavelength. The \
                number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_a: Option<PathBuf>,

    #[clap(
        long,
        help = "Whether to match the measured BRDF data to physically measured in-plane BRDF data \
                by O. Clausen. If true, the inputs should be in pairs of measured data and O. \
                Clausen's data.",
        default_value = "false"
    )]
    pub clausen: bool,

    #[clap(
        long,
        help = "Whether to use 4 times mores samples while resampling the measured data to match \
                the resolution of the Clausen's data.",
        default_value = "false"
    )]
    pub dense: bool,

    #[clap(
        long,
        short,
        help = "Specifies the output file for saving the fitted data. If omitted, the data is \
                written to standard output. Use \"auto\" to automatically determine the output \
                filename."
    )]
    pub output: Option<String>,

    #[clap(
        long,
        short,
        help = "Model to fit the measurement to. If not specified, the default model will be used."
    )]
    pub family: BrdfFamily,

    #[clap(long, help = "Symmetry of the microfacet model.")]
    pub symmetry: Symmetry,

    #[clap(
        long,
        short,
        help = "Distribution to use for the microfacet model. If not specified, the default \
                distribution will be used.",
        required_if_eq("family", "microfacet")
    )]
    pub distro: Option<MicrofacetDistroKind>,

    #[clap(
        long,
        short,
        help = "Level of the measured BRDF data to fit. Only used for the Vgonio kind.",
        required_if_eq("kind", "vgonio"),
        default_value = "l0"
    )]
    pub level: BrdfLevel,

    #[clap(long, help = "Theta limit for the fitting in degrees. Default to 90°.")]
    pub theta_limit: Option<f32>,

    #[clap(long, short, help = "Method to use for the fitting.")]
    pub method: FittingMethod,

    #[clap(
        long = "bprec",
        help = "Precision of the brute force fitting: number of digits after the decimal point.",
        default_value = "6"
    )]
    pub brute_precision: u32,

    #[clap(
        long = "err",
        help = "Error metric used during calculation of objective function (residuals).",
        default_value = "mse",
        required_if_eq("method", "brute")
    )]
    pub error_metric: Option<ErrorMetric>,

    #[clap(
        short,
        long,
        help = "The weighting to use to weight the measured data.",
        default_value = "none"
    )]
    pub weighting: Weighting,

    #[clap(
        long = "per-wl",
        help = "Whether to fit the measured data per wavelength.",
        default_value = "false"
    )]
    pub per_wavelength: bool,

    #[clap(
        long,
        help = "Whether to plot the fitted data.",
        default_value = "false"
    )]
    pub plot: bool,

    #[cfg(feature = "cuda")]
    #[clap(long, help = "Enable CUDA acceleration for the fitting.")]
    pub cuda: bool,

    // Temporary fix for adding the NDF fitting
    #[clap(long, help = "NDF fitting.")]
    pub ndf: bool,
}

impl FitOptions {
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
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FittingMethod {
    /// Brute force fitting method.
    Brute,
    /// Non-linear least squares fitting method.
    Nllsq,
}

#[cfg(test)]
mod tests {
    use super::{parse_roughness_values, read_per_wavelength_roughness_values};
    use std::io::Write;

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
    fn parse_roughness_values_rejects_reversed_range() {
        let err = parse_roughness_values("0.5:0.1:0.01").unwrap_err();
        assert_eq!(err, "roughness range start must be <= end");
    }

    #[test]
    fn read_per_wavelength_roughness_values_parses_valid_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "0.1:0.2:0.01").unwrap();
        writeln!(file, "0.2:0.3:0.01").unwrap();
        writeln!(file, "0.3:0.4:0.01").unwrap();
        file.flush().unwrap();

        let values = read_per_wavelength_roughness_values(file.path(), 3).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0].start, 0.1);
        assert_eq!(values[2].stop, 0.4);
    }

    #[test]
    fn read_per_wavelength_roughness_values_rejects_length_mismatch() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "0.1:0.2:0.01").unwrap();
        file.flush().unwrap();

        let err = read_per_wavelength_roughness_values(file.path(), 2).unwrap_err();
        assert_eq!(
            err,
            "The number of triplets should be equal to the number of wavelengths."
        );
    }
}

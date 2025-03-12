use crate::{measure::bsdf::BsdfMeasurement, pyplot::plot_err};
use dirs::cache_dir;
use std::{
    fmt::{format, Debug},
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use vgonio_core::{
    cli,
    config::Config,
    error::VgonioError,
    optics::IorReg,
    res::DataStore,
    units::{Radians, Rads},
    utils::range::StepRangeIncl,
    AnyMeasuredBrdf, BrdfLevel, ErrorMetric, Symmetry, Weighting,
};

use crate::{app::cache::Cache, pyplot::plot_per_wavelength_err};
use vgonio_bxdf::{
    brdf::measured::{merl::MerlBrdf, rgl::RglBrdf, yan::Yan18Brdf, ClausenBrdf},
    fitting::{FittingProblem, FittingReport},
};
use vgonio_core::{
    bxdf::{AnalyticalBrdf, BrdfFamily, BrdfProxy, MeasuredBrdfKind, MicrofacetDistroKind},
    cli::ansi,
    units::Nanometres,
};

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
    cli::println(
        '>',
        2,
        format_args!(
            "Fitting to model {:?}@{:?}",
            opts.family,
            opts.distro.unwrap()
        ),
        ansi::Color::BrightYellow,
    );
    let theta_limit = opts
        .theta_limit
        .and_then(|t| Some(Radians::from_degrees(t)));
    // Load the data from the cache if the fitting is BxDF
    let cache = Cache::new(config.cache_dir());
    cache.write(|cache| {
        cache.load_ior_database(&config);
        if opts.kind == MeasuredBrdfKind::Vgonio && opts.clausen {
            println!("Fitting simulated data to Clausen's data.");
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
                    println!("Unknown measured BRDF kind specified, cannot fit!");
                },
            }
        }
        Ok(())
    })
}

fn brdf_fitting_brute_force<F: AnyMeasuredBrdf>(
    brdf: &F,
    opts: &FitOptions,
    iors: &IorReg,
    writer: Option<&mut BufWriter<File>>,
) {
    println!(
        "      {} Fitting using brute force method... {}",
        ansi::YELLOW_GT,
        if opts.per_wavelength {
            "per wavelength"
        } else {
            ""
        }
    );
    let start = std::time::Instant::now();
    let full_proxy = brdf.proxy(iors);
    let reports = if opts.per_wavelength {
        let mut reports = Box::new_uninit_slice(brdf.spectrum().len());
        let wavelengths = brdf.spectrum();
        for (i, w) in wavelengths.iter().enumerate() {
            let proxy = full_proxy.per_wavelength(i);
            let report = brdf_fitting_brute_force_inner(proxy, opts, 0, Some(*w));
            reports[i].write((Some(*w), report));
        }
        unsafe { reports.assume_init() }
    } else {
        Box::new([(
            None,
            brdf_fitting_brute_force_inner(full_proxy, opts, 4, None),
        )])
    };
    let end = std::time::Instant::now();
    println!("    {} Took: {:?}", ansi::YELLOW_GT, end - start);

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
        proxy: BrdfProxy,
        opts: &FitOptions,
        n: usize,
        w: Option<Nanometres>,
    ) -> FittingReport<Box<dyn AnalyticalBrdf<Params = [f64; 2]>>> {
        let report = proxy.brute_fit(
            opts.distro.unwrap(),
            opts.symmetry,
            opts.error_metric.unwrap_or(ErrorMetric::Mse),
            opts.weighting,
            opts.theta_limit.map(|t| Radians::from_degrees(t)),
            opts.theta_limit.map(|t| Radians::from_degrees(t)),
            opts.brute_precision,
        );
        // Print the fitting report
        if let Some(w) = w {
            print!("      {} λ = {:?}: ", ansi::YELLOW_GT, w);
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
    ) -> FittingReport<Box<dyn AnalyticalBrdf<Params = [f64; 2]>>> {
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
            println!("      {} λ = {:?}: ", ansi::YELLOW_GT, w);
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
    cli::println(
        '>',
        4,
        format_args!(
            "Fitting ({:?}) to model: {:?}, distro: {:?}, symmetry: {}, method: {:?}, error \
             metric: {}, weighting: {:?}, θ < {}",
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
        ),
        ansi::Color::BrightGreen,
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
            std::fs::OpenOptions::new()
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
        FittingReport<Box<dyn AnalyticalBrdf<Params = [f64; 2]>>>,
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

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(num_args = 1.., value_delimiter = ' ', help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

    #[clap(long, short, help = "Kind of the measured BRDF data.")]
    pub kind: MeasuredBrdfKind,

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
        help = "Error metric to use ONLY for the brute force fitting.",
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
        long,
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
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FittingMethod {
    /// Brute force fitting method.
    Brute,
    /// Non-linear least squares fitting method.
    Nllsq,
}

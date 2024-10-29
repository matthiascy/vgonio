use crate::{
    app::{cache::Cache, cli::ansi, Config},
    fitting::{err, FittingProblem, MicrofacetBrdfFittingProblem},
    measure::bsdf::{receiver::ReceiverParams, MeasuredBrdfLevel, MeasuredBsdfData},
    pyplot::plot_err,
};
use base::{
    error::VgonioError,
    math::Sph2,
    medium::Medium,
    optics::ior::RefractiveIndexRegistry,
    partition::{PartitionScheme, SphericalDomain},
    range::RangeByStepSizeInclusive,
    units::{nm, Radians, Rads},
    ErrorMetric, Isotropy, ResidualErrorMetric,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::{
            AnalyticalFit, ClausenBrdf, MeasuredBrdfKind, VgonioBrdf, VgonioBrdfParameterisation,
        },
        Bxdf, BxdfFamily,
    },
    distro::MicrofacetDistroKind,
};
use core::slice::SlicePattern;
use jabr::array::DyArr;
use std::path::PathBuf;

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    println!(
        "  {}>{} Fitting to model: {:?}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        opts.family,
    );
    let theta_limit = opts
        .theta_limit
        .and_then(|t| Some(Radians::from_degrees(t)));
    // Load the data from the cache if the fitting is BxDF
    let cache = Cache::new(config.cache_dir());
    let alpha = RangeByStepSizeInclusive::new(
        opts.alpha_start.unwrap(),
        opts.alpha_stop.unwrap(),
        opts.alpha_step.unwrap(),
    );
    cache.write(|cache| {
        cache.load_ior_database(&config);
        if opts.generate {
            let roughness = RangeByStepSizeInclusive::new(
                opts.alpha_start.unwrap(),
                opts.alpha_stop.unwrap(),
                opts.alpha_step.unwrap(),
            );
            let alphas = roughness.values();
            let models = match opts.family {
                BxdfFamily::Microfacet => match opts.distro {
                    MicrofacetDistroKind::Beckmann => {
                        let mut models = vec![];
                        for alpha in alphas {
                            models.push(Box::new(BeckmannBrdf::new(alpha, alpha))
                                as Box<dyn Bxdf<Params = [f64; 2]>>);
                        }
                        models.into_boxed_slice()
                    },
                    MicrofacetDistroKind::TrowbridgeReitz => {
                        let mut models = vec![];
                        for alpha in alphas {
                            models.push(Box::new(TrowbridgeReitzBrdf::new(alpha, alpha))
                                as Box<dyn Bxdf<Params = [f64; 2]>>);
                        }
                        models.into_boxed_slice()
                    },
                },
                _ => unimplemented!("Only microfacet-based BxDFs are supported."),
            };
            let spectrum = RangeByStepSizeInclusive {
                start: nm!(400.0),
                stop: nm!(700.0),
                step_size: nm!(300.0),
            }
            .values()
            .collect::<Vec<_>>();
            let zenith = RangeByStepSizeInclusive {
                start: Rads::ZERO,
                stop: Rads::HALF_PI,
                step_size: Rads::from_degrees(5.0),
            };
            let azimuth = RangeByStepSizeInclusive {
                start: Rads::ZERO,
                stop: Rads::TWO_PI,
                step_size: Rads::from_degrees(60.0),
            };
            let mut incoming =
                DyArr::<Sph2>::zeros([zenith.step_count_wrapped() * azimuth.step_count_wrapped()]);
            for (i, theta) in zenith.values_wrapped().enumerate() {
                for (j, phi) in azimuth.values_wrapped().enumerate() {
                    incoming[i * azimuth.step_count_wrapped() + j] = Sph2::new(theta, phi);
                }
            }
            let params = VgonioBrdfParameterisation {
                n_zenith_i: zenith.step_count_wrapped(),
                incoming,
                outgoing: ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision: Sph2::new(Rads::from_degrees(2.0), Rads::from_degrees(2.0)),
                    scheme: PartitionScheme::Beckers,
                }
                .partitioning(),
            };
            for model in models.iter() {
                let brdf = VgonioBrdf::new_analytical(
                    Medium::Air,
                    Medium::Aluminium,
                    &spectrum,
                    &params,
                    &**model,
                    &cache.iors,
                );
                let output = config.output_dir().join(format!(
                    "{:?}_{}.exr",
                    model.family(),
                    model.params()[0],
                ));
                brdf.write_as_exr(&output, &chrono::Local::now(), 512)
                    .unwrap();
            }
        } else {
            if opts.clausen {
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
                        if loaded.iter().all(|m| m.is_clausen())
                            || loaded.iter().all(|m| !m.is_clausen())
                        {
                            return Err(VgonioError::new(
                                "The input files should be in pairs of measured data and \
                                 corresponding Clausen's data.",
                                None,
                            ));
                        }
                        let simulated_brdf_index = if loaded[0].is_clausen() { 1 } else { 0 };
                        let clausen_brdf_index = simulated_brdf_index ^ 1;
                        let simulated_brdf = loaded[simulated_brdf_index]
                            .downcast_ref::<MeasuredBsdfData>()
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
                    measured_brdf_fitting(&opts, &pair[0], &brdf, alpha, &cache.iors, theta_limit);
                }
            } else {
                for input in &opts.inputs {
                    let measurement = cache
                        .load_micro_surface_measurement(&config, &input)
                        .unwrap();
                    let measured = &cache.get_measurement(measurement).unwrap().measured;
                    match measured.downcast_ref::<MeasuredBsdfData>() {
                        None => {
                            let brdf = measured.downcast_ref::<ClausenBrdf>().unwrap();
                            measured_brdf_fitting(
                                &opts,
                                &input,
                                brdf,
                                alpha,
                                &cache.iors,
                                theta_limit,
                            );
                        },
                        Some(brdfs) => {
                            let brdf = brdfs.brdf_at(MeasuredBrdfLevel::from(opts.level)).unwrap();
                            measured_brdf_fitting(
                                &opts,
                                &input,
                                brdf,
                                alpha,
                                &cache.iors,
                                theta_limit,
                            )
                        },
                    }
                }
            }
        }
        Ok(())
    })
}

fn brdf_fitting_brute_force<F: AnalyticalFit + Sync>(
    brdf: &F,
    filepath: &PathBuf,
    opts: &FitOptions,
    alpha: RangeByStepSizeInclusive<f64>,
    iors: &RefractiveIndexRegistry,
) {
    let errs = err::compute_microfacet_brdf_err(
        brdf,
        opts.distro,
        alpha,
        iors,
        opts.theta_limit
            .and_then(|t| Some(Radians::from_degrees(t)))
            .unwrap_or(Radians::HALF_PI),
        opts.error_metric.unwrap_or(ErrorMetric::Mse),
        opts.residual_error_metric,
    );
    let min_err = errs.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    let min_idx = errs.iter().position(|&x| x == min_err).unwrap();
    println!(
        "    {}>{} MSEs ({}) {:?}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        filepath.file_name().unwrap().display(),
        errs.as_slice()
    );
    println!(
        "    {}>{} Minimum error: {} at alpha = {}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        min_err,
        alpha.values().nth(min_idx).unwrap()
    );
    if opts.plot {
        plot_err(
            errs.as_slice(),
            opts.alpha_start.unwrap_or(0.01),
            opts.alpha_stop.unwrap_or(1.0),
            opts.alpha_step.unwrap_or(0.01),
        )
        .expect("Failed to plot the error.");
    }
}

fn measured_brdf_fitting<F: AnalyticalFit + Sync>(
    opts: &FitOptions,
    filepath: &PathBuf,
    brdf: &F,
    alpha: RangeByStepSizeInclusive<f64>,
    iors: &RefractiveIndexRegistry,
    theta_limit: Option<Radians>,
) {
    let limit = theta_limit.unwrap_or(Radians::HALF_PI);
    println!(
        "    {}>{} Fitting ({:?}) to model: {:?} [{:?}], distro: {:?}, isotropy: {}, method: \
         {:?}, theta limit: {}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        brdf.kind(),
        opts.family,
        opts.residual_error_metric,
        opts.distro,
        opts.isotropy,
        opts.method,
        limit.prettified()
    );
    if brdf.kind() == MeasuredBrdfKind::Clausen {
        let brdf = brdf.as_any().downcast_ref::<ClausenBrdf>().unwrap();
        println!(
            "    {}>{} Clausen's data, data point positions:",
            ansi::BRIGHT_YELLOW,
            ansi::RESET,
        );
        for (_, (wi, wos)) in brdf.params.wi_wos_iter() {
            print!("        wi: {} - wo:", wi);
            for wo in wos.iter() {
                print!(" {}", wo);
            }
            println!();
        }
    }
    match opts.method {
        FittingMethod::Brute => brdf_fitting_brute_force(brdf, filepath, opts, alpha, iors),
        FittingMethod::Nllsq => {
            let problem = MicrofacetBrdfFittingProblem::new(
                brdf,
                opts.distro,
                alpha,
                opts.level,
                iors,
                limit,
            );
            problem
                .lsq_lm_fit(opts.isotropy, opts.residual_error_metric)
                .print_fitting_report();
        },
    }
}

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(long, short, help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

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
        help = "Output file to save the fitted data. If not specified, the fitted data will be \
                written to the standard output."
    )]
    pub output: Option<String>,

    #[clap(
        long,
        short,
        help = "Model to fit the measurement to. If not specified, the default model will be used."
    )]
    pub family: BxdfFamily,

    #[clap(
        long,
        help = "Isotropy of the microfacet model. If not specified, the default isotropy will be \
                used.",
        default_value = "isotropic"
    )]
    pub isotropy: Isotropy,

    #[clap(
        long,
        help = "Distribution to use for the microfacet model. If not specified, the default \
                distribution will be used.",
        default_value = "beckmann",
        required_if_eq_all([("family", "microfacet"), ("generate", "false")])
    )]
    pub distro: MicrofacetDistroKind,

    #[clap(
        short,
        long,
        help = "Level of the measured BRDF data to fit.",
        default_value = "l0"
    )]
    pub level: MeasuredBrdfLevel,

    #[clap(
        long,
        help = "Generate the analytical model for the given family and distribution.",
        default_value = "false"
    )]
    pub generate: bool,

    #[clap(
        short = 's',
        long = "alpha_start",
        help = "Start roughness.",
        default_value = "0.01"
    )]
    pub alpha_start: Option<f64>,

    #[clap(
        short = 'e',
        long = "alpha_stop",
        help = "End roughness.",
        default_value = "1.0"
    )]
    pub alpha_stop: Option<f64>,

    #[clap(
        short = 'p',
        long = "alpha_step",
        help = "Roughness step size.",
        default_value = "0.01"
    )]
    pub alpha_step: Option<f64>,

    #[clap(long, help = "Theta limit for the fitting in degrees. Default to 90°.")]
    pub theta_limit: Option<f32>,

    #[clap(
        short,
        long,
        help = "Method to use for the fitting.",
        default_value = "brute"
    )]
    pub method: FittingMethod,

    #[clap(
        long = "err",
        help = "Error metric to use ONLY for the brute force fitting.",
        default_value = "mse",
        required_if_eq_all([("method", "bruteforce"), ("generate", "false")])
    )]
    pub error_metric: Option<ErrorMetric>,

    #[clap(
        long,
        help = "The error metric to use to weight the measured data.",
        default_value = "identity"
    )]
    pub residual_error_metric: ResidualErrorMetric,

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

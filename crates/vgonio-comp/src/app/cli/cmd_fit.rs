use crate::{
    app::{cache::Cache, cli::ansi, Config},
    measure::bsdf::{MeasuredBrdfLevel, MeasuredBsdfData},
    pyplot::plot_err,
};
use base::{
    error::VgonioError,
    optics::ior::IorRegistry,
    range::StepRangeIncl,
    units::{Radians, Rads},
    ErrorMetric, Isotropy, MeasuredBrdfKind, Weighting,
};
use bxdf::{
    brdf::{
        measured::{merl::MerlBrdf, rgl::RglBrdf, yan::Yan2018Brdf, ClausenBrdf},
        BxdfFamily,
    },
    distro::MicrofacetDistroKind,
    fitting::{brdf::AnalyticalFit2, FittingProblem},
};
use std::path::PathBuf;

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    println!(
        "  {}>{} Fitting to model: {:?}@{:?}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        opts.family,
        opts.distro
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
                    if loaded
                        .iter()
                        .all(|m| m.brdf_kind() == Some(MeasuredBrdfKind::Clausen))
                        || loaded
                            .iter()
                            .all(|m| m.brdf_kind() != Some(MeasuredBrdfKind::Clausen))
                    {
                        return Err(VgonioError::new(
                            "The input files should be in pairs of measured data and \
                             corresponding Clausen's data.",
                            None,
                        ));
                    }
                    let simulated_brdf_index =
                        if loaded[0].brdf_kind() == Some(MeasuredBrdfKind::Clausen) {
                            1
                        } else {
                            0
                        };
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
                measured_brdf_fitting(&opts, &brdf, &cache.iors, theta_limit);
            }
        } else {
            match opts.kind {
                MeasuredBrdfKind::Clausen => {
                    for input in &opts.inputs {
                        let measurement = cache
                            .load_micro_surface_measurement(&config, &input)
                            .unwrap();
                        if let Some(brdf) = cache
                            .get_measurement(measurement)
                            .unwrap()
                            .measured
                            .downcast_ref::<ClausenBrdf>()
                        {
                            measured_brdf_fitting(&opts, brdf, &cache.iors, theta_limit);
                        }
                    }
                },
                MeasuredBrdfKind::Merl => {
                    for input in &opts.inputs {
                        let measurement = cache
                            .load_micro_surface_measurement(&config, &input)
                            .unwrap();
                        println!("Get merl measurement");
                        if let Some(brdf) = cache
                            .get_measurement(measurement)
                            .unwrap()
                            .measured
                            .downcast_ref::<MerlBrdf>()
                        {
                            measured_brdf_fitting(&opts, brdf, &cache.iors, theta_limit);
                        }
                    }
                },
                MeasuredBrdfKind::Utia => {},
                MeasuredBrdfKind::Vgonio => {
                    for input in &opts.inputs {
                        let measurement = cache
                            .load_micro_surface_measurement(&config, &input)
                            .unwrap();
                        if let Some(measured) = cache
                            .get_measurement(measurement)
                            .unwrap()
                            .measured
                            .downcast_ref::<MeasuredBsdfData>()
                        {
                            let brdf = measured
                                .brdf_at(MeasuredBrdfLevel::from(opts.level))
                                .unwrap();
                            measured_brdf_fitting(&opts, brdf, &cache.iors, theta_limit);
                        }
                    }
                },
                MeasuredBrdfKind::Yan2018 => {
                    for input in &opts.inputs {
                        let measurement = cache
                            .load_micro_surface_measurement(&config, &input)
                            .unwrap();
                        if let Some(brdf) = cache
                            .get_measurement(measurement)
                            .unwrap()
                            .measured
                            .downcast_ref::<Yan2018Brdf>()
                        {
                            measured_brdf_fitting(&opts, brdf, &cache.iors, theta_limit);
                        }
                    }
                },
                MeasuredBrdfKind::Rgl => {
                    for input in &opts.inputs {
                        let measurement = cache
                            .load_micro_surface_measurement(&config, &input)
                            .unwrap();
                        if let Some(brdf) = cache
                            .get_measurement(measurement)
                            .unwrap()
                            .measured
                            .downcast_ref::<RglBrdf>()
                        {
                            log::debug!("RGL BRDF fitting");
                            measured_brdf_fitting(&opts, brdf, &cache.iors, theta_limit);
                        }
                    }
                },
                MeasuredBrdfKind::Unknown => {
                    println!("Unknown measured BRDF kind specified, cannot fit!");
                },
            }
        }
        Ok(())
    })
}

fn brdf_fitting_brute_force<F: AnalyticalFit2>(brdf: &F, opts: &FitOptions, iors: &IorRegistry) {
    let start = std::time::Instant::now();
    let report = brdf.proxy(iors).brute_fit(
        opts.distro,
        opts.error_metric.unwrap_or(ErrorMetric::Mse),
        opts.weighting,
        opts.theta_limit.map(|t| Radians::from_degrees(t)),
        opts.theta_limit.map(|t| Radians::from_degrees(t)),
        opts.brute_precision,
    );
    let end = std::time::Instant::now();
    println!("    {} Took: {:?}", ansi::YELLOW_GT, end - start);

    report.print_fitting_report(4);

    if opts.plot {
        let mut alpha_error_pairs = report
            .reports
            .iter()
            .map(|(m, r)| (m.params()[0], r.objective_function))
            .collect::<Vec<_>>();
        // Sort the error by alpha value for later plotting
        alpha_error_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (alpha, error): (Vec<_>, Vec<_>) = alpha_error_pairs.into_iter().unzip();

        plot_err(error.as_slice(), alpha.as_slice(), opts.brute_precision)
            .expect("Failed to plot the error.");
    }
}

fn measured_brdf_fitting<F: AnalyticalFit2>(
    opts: &FitOptions,
    brdf: &F,
    iors: &IorRegistry,
    theta_limit: Option<Radians>,
) {
    let limit = theta_limit.unwrap_or(Radians::HALF_PI);
    println!(
        "    {} Fitting ({:?}) to model: {:?}, distro: {:?}, isotropy: {}, method: {:?}, error \
         metric: {:?}, weighting: {:?}, θ < {}",
        ansi::YELLOW_GT,
        brdf.kind(),
        opts.family,
        opts.distro,
        opts.isotropy,
        opts.method,
        opts.error_metric.unwrap_or(ErrorMetric::Mse),
        opts.weighting,
        limit.prettified()
    );

    match opts.method {
        FittingMethod::Brute => brdf_fitting_brute_force(brdf, opts, iors),
        FittingMethod::Nllsq => {
            let proxy = brdf.proxy(iors);

            // Adjust the alpha range only if the model is isotropic
            let alpha = match opts.isotropy {
                Isotropy::Isotropic => {
                    let report = brdf.proxy(iors).brute_fit(
                        opts.distro,
                        opts.error_metric.unwrap_or(ErrorMetric::Mse),
                        opts.weighting,
                        opts.theta_limit.map(|t| Radians::from_degrees(t)),
                        opts.theta_limit.map(|t| Radians::from_degrees(t)),
                        2,
                    );
                    let mid = report.best_model().unwrap().params()[0];
                    StepRangeIncl::new(mid - 0.01, mid + 0.1, 0.001)
                },
                Isotropy::Anisotropic => StepRangeIncl::new(0.0, 1.0, 0.01),
            };

            let report = proxy.nllsq_fit(
                opts.distro,
                opts.isotropy,
                opts.weighting,
                alpha,
                opts.theta_limit.map(|t| Radians::from_degrees(t)),
                opts.theta_limit.map(|t| Radians::from_degrees(t)),
            );
            report.print_fitting_report(8);
        },
    }
}

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(long, short, help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

    #[clap(long, help = "Kind of the measured BRDF data.")]
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
        required_if_eq("family", "microfacet")
    )]
    pub distro: MicrofacetDistroKind,

    #[clap(
        short,
        long,
        help = "Level of the measured BRDF data to fit.",
        default_value = "l0"
    )]
    pub level: MeasuredBrdfLevel,

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
        long = "bprecision",
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

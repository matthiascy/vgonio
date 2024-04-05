#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::RtcMethod::Embree;
use crate::{
    app::{cache::Cache, cli::ansi, Config},
    fitting::{
        err::{compute_iso_microfacet_brdf_err, generate_analytical_brdf, ErrorMetric},
        FittingProblem, MicrofacetBrdfFittingProblem, SampledBrdfFittingProblem,
    },
    measure::{
        bsdf::{
            emitter::EmitterParams,
            receiver::{DataRetrieval, ReceiverParams},
            BsdfKind,
        },
        params::{BsdfMeasurementParams, SimulationKind},
    },
    partition::PartitionScheme,
    SphericalDomain,
};
use base::{
    error::VgonioError,
    math::Sph2,
    medium::Medium,
    range::RangeByStepSizeInclusive,
    units::{deg, nm, Rads},
    Isotropy,
};
use bxdf::{
    brdf::{
        microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf, TrowbridgeReitzBrdfModel},
        Bxdf, BxdfFamily,
    },
    distro::MicrofacetDistroKind,
};
use core::slice::SlicePattern;
use rayon::iter::ParallelIterator;
use std::path::PathBuf;

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    println!(
        "  {}>{} Fitting to model: {:?}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        opts.family,
    );
    // Load the data from the cache if the fitting is BxDF
    let cache = Cache::new(config.cache_dir());
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
                BxdfFamily::Microfacet => {
                    match opts
                        .distro
                        .expect("Distribution must be specified for microfacet family")
                    {
                        MicrofacetDistroKind::Beckmann => {
                            let mut models = vec![];
                            for alpha in alphas {
                                models.push(Box::new(BeckmannBrdf::new(alpha, alpha))
                                    as Box<dyn Bxdf<Params = [f64; 2]>>);
                            }
                            models.into_boxed_slice()
                        }
                        MicrofacetDistroKind::TrowbridgeReitz => {
                            let mut models = vec![];
                            for alpha in alphas {
                                models.push(Box::new(TrowbridgeReitzBrdf::new(alpha, alpha))
                                    as Box<dyn Bxdf<Params = [f64; 2]>>);
                            }
                            models.into_boxed_slice()
                        }
                    }
                }
                _ => unimplemented!("Only microfacet-based BxDFs are supported."),
            };
            let params = BsdfMeasurementParams {
                kind: BsdfKind::Brdf,
                sim_kind: SimulationKind::GeomOptics(Embree),
                incident_medium: Medium::Air,
                transmitted_medium: Medium::Aluminium,
                emitter: EmitterParams {
                    num_rays: 0,
                    max_bounces: 0,
                    zenith: RangeByStepSizeInclusive {
                        start: Rads::ZERO,
                        stop: Rads::HALF_PI,
                        step_size: Rads::from_degrees(5.0),
                    },
                    azimuth: RangeByStepSizeInclusive {
                        start: Rads::ZERO,
                        stop: Rads::TWO_PI,
                        step_size: Rads::from_degrees(60.0),
                    },
                    spectrum: RangeByStepSizeInclusive {
                        start: nm!(400.0),
                        stop: nm!(700.0),
                        step_size: nm!(300.0),
                    },
                },
                receiver: ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision: Sph2::new(Rads::from_degrees(2.0), Rads::from_degrees(2.0)),
                    scheme: PartitionScheme::Beckers,
                    retrieval: DataRetrieval::BsdfOnly,
                },
                fresnel: true,
            };
            for model in models.iter() {
                let (mut brdf, max_values) =
                    generate_analytical_brdf(&params, &**model, &cache.iors, opts.normalize);
                let output = config.output_dir().join(format!(
                    "{:?}_{}.exr",
                    model.family(),
                    model.params()[0],
                ));
                brdf.write_as_exr(&output, &chrono::Local::now(), 512, opts.normalize)
                    .unwrap();
            }
        } else {
            for input in opts.inputs {
                let measurement = cache
                    .load_micro_surface_measurement(&config, &input)
                    .unwrap();
                let measured = &cache.get_measurement_data(measurement).unwrap().measured;
                match measured.as_bsdf() {
                    None => {
                        let measured_brdf = measured.as_sampled_brdf().unwrap();
                        match opts.method {
                            FittingMethod::Bruteforce => {
                                // TODO: unify SampledBrdfModel and MeasuredBrdfModel
                                todo!("Bruteforce fitting for sampled BRDFs")
                            }
                            FittingMethod::Nlls => {
                                let problem = SampledBrdfFittingProblem::new(
                                    measured_brdf,
                                    opts.distro.expect(
                                        "Distribution must be specified for microfacet family",
                                    ),
                                    RangeByStepSizeInclusive::new(
                                        opts.alpha_start.unwrap(),
                                        opts.alpha_stop.unwrap(),
                                        opts.alpha_step.unwrap(),
                                    ),
                                    cache,
                                );
                                let report = problem.lsq_lm_fit(opts.isotropy);
                                report.print_fitting_report();
                            }
                        }
                    }
                    Some(measured_brdf) => {
                        match opts.method {
                            FittingMethod::Bruteforce => {
                                let mses = compute_iso_microfacet_brdf_err(
                                    &measured_brdf,
                                    opts.max_theta_o.map(|t| deg!(t as f32)),
                                    opts.distro.expect(
                                        "Distribution must be specified for microfacet family",
                                    ),
                                    RangeByStepSizeInclusive::new(
                                        opts.alpha_start.unwrap(),
                                        opts.alpha_stop.unwrap(),
                                        opts.alpha_step.unwrap(),
                                    ),
                                    &cache,
                                    opts.normalize,
                                    opts.error_metric.unwrap_or(ErrorMetric::Mse),
                                );
                                println!(
                                    "    {}>{} MSE ({}) {:?}",
                                    ansi::BRIGHT_YELLOW,
                                    ansi::RESET,
                                    input.file_name().unwrap().display(),
                                    mses.as_slice()
                                );
                            }
                            FittingMethod::Nlls => {
                                // TODO: unify BrdfModel and MicrofacetBasedBrdfModel
                                let problem = MicrofacetBrdfFittingProblem::new(
                                    measured_brdf,
                                    opts.distro.expect(
                                        "Distribution must be specified for microfacet family",
                                    ),
                                    RangeByStepSizeInclusive::new(
                                        opts.alpha_start.unwrap(),
                                        opts.alpha_stop.unwrap(),
                                        opts.alpha_step.unwrap(),
                                    ),
                                    cache,
                                );
                                let report = problem.lsq_lm_fit(opts.isotropy);
                                report.print_fitting_report();
                            }
                        }
                    }
                }
            }
        }
    });

    Ok(())
}

// TODO: complete fit kind

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    pub inputs: Vec<PathBuf>,
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
                used."
    )]
    pub isotropy: Isotropy,
    #[clap(
        long,
        help = "Distribution to use for the microfacet model. If not specified, the default \
                distribution will be used.",
        required_if_eq_all([("family", "microfacet"), ("generate", "false")])
    )]
    pub distro: Option<MicrofacetDistroKind>,
    #[clap(
        long,
        help = "Maximum colatitude angle in degrees for outgoing directions."
    )]
    pub max_theta_o: Option<f64>,
    #[clap(
        long,
        help = "Generate the analytical model for the given model. If not specified, the \
                analytical model will not be generated.",
        default_value = "false"
    )]
    pub generate: bool,
    #[clap(
        long,
        help = "Whether to normalize the generated analytical model. If not specified, the \
                generated analytical model will not be normalized.",
        default_value = "false"
    )]
    pub normalize: bool,
    #[clap(long = "astart", help = "Start roughness.", default_value = "0.01")]
    pub alpha_start: Option<f64>,
    #[clap(long = "astop", help = "End roughness.", default_value = "1.0")]
    pub alpha_stop: Option<f64>,
    #[clap(long = "astep", help = "Roughness step size.", default_value = "0.01")]
    pub alpha_step: Option<f64>,
    #[clap(
        long,
        help = "Method to use for the fitting.",
        default_value = "bruteforce"
    )]
    pub method: FittingMethod,
    #[clap(
        short,
        help = "Error metric to use for the fitting.",
        default_value = "mse"
    )]
    pub error_metric: Option<ErrorMetric>,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FittingMethod {
    Bruteforce,
    Nlls,
}

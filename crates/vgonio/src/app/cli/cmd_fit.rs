use crate::{
    app::{cache::Cache, cli::ansi, Config},
    fitting::{
        err::compute_microfacet_brdf_err, FittingProblem, FittingReport,
        MicrofacetBrdfFittingProblem,
    },
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
    units::{nm, Rads},
    ErrorMetric, Isotropy,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::{AnalyticalFit, ClausenBrdf, VgonioBrdf, VgonioBrdfParameterisation},
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
                    }
                    MicrofacetDistroKind::TrowbridgeReitz => {
                        let mut models = vec![];
                        for alpha in alphas {
                            models.push(Box::new(TrowbridgeReitzBrdf::new(alpha, alpha))
                                as Box<dyn Bxdf<Params = [f64; 2]>>);
                        }
                        models.into_boxed_slice()
                    }
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
            if opts.olaf {
                println!("Fitting to Olaf data.");
                if opts.inputs.len() % 2 != 0 {
                    return Err(VgonioError::new(
                        "The input files should be in pairs of measured data and corresponding \
                         Olaf data.",
                        None,
                    ));
                }
                for input in opts.inputs.chunks(2) {
                    log::debug!("inputs: {:?}, {:?}", input[0], input[1]);
                    let brdf = {
                        let measured = cache
                            .load_micro_surface_measurement(&config, &input[0])
                            .unwrap();
                        let olaf = cache
                            .load_micro_surface_measurement(&config, &input[1])
                            .unwrap();
                        let simulated_brdf = cache
                            .get_measurement(measured)
                            .unwrap()
                            .measured
                            .downcast_ref::<MeasuredBsdfData>()
                            .unwrap();
                        let level = MeasuredBrdfLevel::from(opts.level);
                        #[cfg(debug_assertions)]
                        {
                            let brdf = simulated_brdf.brdf_at(level).unwrap();
                            let n_spectrum = simulated_brdf.n_spectrum();
                            let partition = &simulated_brdf.raw.outgoing;
                            for snapshot in brdf.snapshots() {
                                log::debug!("  = snapshot | wi: {}", snapshot.wi);
                                for (i, ring) in partition.rings.iter().enumerate() {
                                    log::debug!(
                                        "    - ring[{}], center: (min {}, max {})",
                                        i,
                                        ring.theta_min.to_degrees(),
                                        ring.theta_max.to_degrees()
                                    );
                                    for pid in 0..ring.patch_count {
                                        let patch_idx = pid + ring.base_index;
                                        log::debug!(
                                            "      - patch[{}] (min: {}, max: {}, center: {}): \
                                             {:?}",
                                            patch_idx,
                                            partition.patches[patch_idx].min,
                                            partition.patches[patch_idx].max,
                                            partition.patches[patch_idx].center(),
                                            &snapshot.samples.as_slice()[patch_idx * n_spectrum
                                                ..(patch_idx + 1) * n_spectrum]
                                        );
                                    }
                                }
                            }
                        }
                        let olaf_data = cache
                            .get_measurement(olaf)
                            .unwrap()
                            .measured
                            .downcast_ref::<ClausenBrdf>()
                            .unwrap();
                        let dense = if std::env::var("DENSE")
                            .ok()
                            .map(|s| s == "1")
                            .unwrap_or(false)
                        {
                            true
                        } else {
                            false
                        };
                        log::debug!("Resampling the measured data, dense: {}", dense);
                        simulated_brdf.resample(&olaf_data.params, level, dense, Rads::ZERO)
                    };
                    log::debug!("BRDF extraction done, starting fitting.");
                    measured_brdf_fitting(opts.method, &input[0], &brdf, &opts, alpha, &cache.iors);
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
                                opts.method,
                                &input,
                                brdf,
                                &opts,
                                alpha,
                                &cache.iors,
                            );
                        }
                        Some(brdfs) => {
                            let brdf = brdfs.brdf_at(MeasuredBrdfLevel::from(opts.level)).unwrap();
                            measured_brdf_fitting(
                                opts.method,
                                &input,
                                brdf,
                                &opts,
                                alpha,
                                &cache.iors,
                            )
                        }
                    }
                }
            }
        }
        Ok(())
    })
}

fn brdf_fitting_brute_force<F: AnalyticalFit>(
    brdf: &F,
    filepath: &PathBuf,
    opts: &FitOptions,
    alpha: RangeByStepSizeInclusive<f64>,
    iors: &RefractiveIndexRegistry,
) {
    let errs = compute_microfacet_brdf_err(
        brdf,
        opts.distro,
        alpha,
        iors,
        opts.error_metric.unwrap_or(ErrorMetric::Mse),
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
    method: FittingMethod,
    filepath: &PathBuf,
    brdf: &F,
    opts: &FitOptions,
    alpha: RangeByStepSizeInclusive<f64>,
    iors: &RefractiveIndexRegistry,
) {
    println!(
        "    {}>{} Fitting to model: {:?} , distro: {:?}, isotropy: {}, method: {:?}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        opts.family,
        opts.distro,
        opts.isotropy,
        opts.method,
    );
    match method {
        FittingMethod::Bruteforce => brdf_fitting_brute_force(brdf, filepath, opts, alpha, iors),
        FittingMethod::Nlls => {
            brdf_fitting_nonlin_lsq(brdf, opts, alpha, iors).print_fitting_report()
        }
    }
}

fn brdf_fitting_nonlin_lsq<F: AnalyticalFit + Sync>(
    brdf: &F,
    opts: &FitOptions,
    alpha: RangeByStepSizeInclusive<f64>,
    iors: &RefractiveIndexRegistry,
) -> FittingReport<Box<dyn Bxdf<Params = [f64; 2]>>> {
    let problem = MicrofacetBrdfFittingProblem::new(
        brdf,
        opts.distro,
        alpha,
        MeasuredBrdfLevel::from(opts.level),
        iors,
    );
    problem.lsq_lm_fit(opts.isotropy)
}

// TODO: complete fit kind

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(long, short, help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

    #[clap(
        long,
        help = "Whether to match the measured BRDF data to physically measured in-plane \
                BRDF\ndata by Olaf. If true, the inputs should be in pairs of measured data and \
                Olaf data.",
        default_value = "false"
    )]
    pub olaf: bool,

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
        default_value = "0"
    )]
    pub level: u32,

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

    #[clap(
        short,
        long,
        help = "Method to use for the fitting.",
        default_value = "bruteforce"
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
        help = "Whether to plot the fitted data.",
        default_value = "false"
    )]
    pub plot: bool,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FittingMethod {
    /// Brute force fitting method.
    Bruteforce,
    /// Non-linear least squares fitting method.
    Nlls,
}

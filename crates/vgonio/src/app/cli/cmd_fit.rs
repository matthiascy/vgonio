#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::RtcMethod::Embree;
use crate::{
    app::{
        cache::{Cache, RefractiveIndexRegistry},
        cli::ansi,
        Config,
    },
    fitting::mse::{compute_iso_brdf_mse, generate_analytical_brdf},
    measure::{
        bsdf::{
            emitter::EmitterParams,
            receiver::{DataRetrieval, ReceiverParams},
            BsdfKind, BsdfSnapshot, MeasuredBsdfData, SpectralSamples,
        },
        params::{BsdfMeasurementParams, SimulationKind},
    },
    partition::PartitionScheme,
    RangeByStepSizeInclusive, SphericalDomain,
};
use base::{
    error::VgonioError,
    math::{sph_to_cart, Sph2, Vec3},
    medium::Medium,
    units::{nm, Rads},
};
use bxdf::{
    brdf::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel},
    MicrofacetBasedBrdfModel,
};
use core::slice::SlicePattern;
use rayon::iter::ParallelIterator;
use std::path::PathBuf;

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    println!(
        "  {}>{} Fitting to model: {:?}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        opts.model,
    );
    // Load the data from the cache if the fitting is BxDF
    let cache = Cache::new(config.cache_dir());
    cache.write(|cache| {
        cache.load_ior_database(&config);
        if opts.generate {
            let roughness = RangeByStepSizeInclusive::<f64>::new(0.1, 1.0, 0.1);
            let count = roughness.step_count();
            let models = match opts.model {
                BrdfModel::Beckmann => {
                    let mut models = Vec::with_capacity(count);
                    for i in 0..count {
                        let alpha = i as f64 / count as f64 * roughness.span() + roughness.start;
                        models.push(Box::new(BeckmannBrdfModel::new(alpha, alpha))
                            as Box<dyn MicrofacetBasedBrdfModel>);
                    }
                    models.into_boxed_slice()
                }
                BrdfModel::TrowbridgeReitz => {
                    let mut models = Vec::with_capacity(count);
                    for i in 0..count {
                        let alpha = i as f64 / count as f64 * roughness.span() + roughness.start;
                        models.push(Box::new(TrowbridgeReitzBrdfModel::new(alpha, alpha))
                            as Box<dyn MicrofacetBasedBrdfModel>);
                    }
                    models.into_boxed_slice()
                }
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
                let (mut brdf, max_values) = generate_analytical_brdf(&params, model, &cache.iors);
                if opts.normalize {
                    for (snapshot, max_vals) in brdf.snapshots.iter_mut().zip(max_values.iter()) {
                        for spectral_samples in snapshot.samples.iter_mut() {
                            for (sample, max) in spectral_samples.iter_mut().zip(max_vals.iter()) {
                                *sample /= *max as f32;
                            }
                        }
                    }
                }
                let output =
                    config
                        .output_dir()
                        .join(format!("{:?}_{}.exr", model.kind(), model.alpha_x()));
                brdf.write_as_exr(&output, &chrono::Local::now(), 512)
                    .unwrap();
            }
        } else {
            for input in opts.inputs {
                let measurement = cache
                    .load_micro_surface_measurement(&config, &input)
                    .unwrap();
                let measured_brdf = cache
                    .get_measurement_data(measurement)
                    .unwrap()
                    .measured
                    .as_bsdf()
                    .unwrap();
                let mses = compute_iso_brdf_mse(
                    &measured_brdf,
                    opts.max_theta_o,
                    opts.model,
                    ALPHA_RANGE,
                    &cache,
                );
                println!(
                    "    {}>{} MSE ({}) {:?}",
                    ansi::BRIGHT_YELLOW,
                    ansi::RESET,
                    input.file_name().unwrap().display(),
                    mses.as_slice()
                );
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
    pub model: BrdfModel,
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
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrdfModel {
    Beckmann,
    TrowbridgeReitz,
}

pub const ALPHA_RANGE: RangeByStepSizeInclusive<f64> =
    RangeByStepSizeInclusive::new(0.01, 1.0, 0.01);

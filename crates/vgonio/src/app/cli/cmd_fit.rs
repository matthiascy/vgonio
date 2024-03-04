#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::RtcMethod::Embree;
use crate::{
    app::{cache::Cache, cli::ansi, Config},
    fitting::mse::compute_iso_brdf_mse,
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
            let iors_i = cache
                .iors
                .ior_of_spectrum(Medium::Air, &[nm!(400.0), nm!(700.0)])
                .unwrap();
            let iors_o = cache
                .iors
                .ior_of_spectrum(Medium::Aluminium, &[nm!(400.0), nm!(700.0)])
                .unwrap();
            for model in models.iter() {
                let mut brdf = MeasuredBsdfData {
                    params: BsdfMeasurementParams {
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
                    },
                    snapshots: Box::new([]),
                    raw_snapshots: None,
                };
                let partition = brdf.params.receiver.partitioning();
                let mut snapshots = vec![];
                for theta in (0..=90).step_by(5) {
                    for phi in (0..=360).step_by(60) {
                        let wi_sph = Sph2::new(
                            Rads::from_degrees(theta as f32),
                            Rads::from_degrees(phi as f32),
                        );
                        let wi = sph_to_cart(
                            Rads::from_degrees(theta as f32),
                            Rads::from_degrees(phi as f32),
                        );
                        let mut samples = vec![];
                        for patch in partition.patches.iter() {
                            let wo_sph = patch.center();
                            let wo = sph_to_cart(wo_sph.theta, wo_sph.phi);
                            samples.push(SpectralSamples::from_vec(vec![
                                model.eval(wi, wo, &iors_i[0], &iors_o[0]) as f32,
                                model.eval(wi, wo, &iors_i[1], &iors_o[1]) as f32,
                            ]));
                        }
                        snapshots.push(BsdfSnapshot {
                            wi: wi_sph,
                            samples: samples.into_boxed_slice(),
                            #[cfg(any(feature = "visu-dbg", debug_assertions))]
                            trajectories: vec![],
                            #[cfg(any(feature = "visu-dbg", debug_assertions))]
                            hit_points: vec![],
                        });
                    }
                }
                brdf.snapshots = snapshots.into_boxed_slice();
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
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrdfModel {
    Beckmann,
    TrowbridgeReitz,
}

pub const ALPHA_RANGE: RangeByStepSizeInclusive<f64> =
    RangeByStepSizeInclusive::new(0.01, 1.0, 0.01);

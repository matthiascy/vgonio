use crate::app::{cache::Cache, cli::ansi, Config};
use base::{error::VgonioError, math::spherical_to_cartesian};
use bxdf::{
    brdf::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel},
    MicrofacetBasedBrdfModel,
};
use rayon::{
    iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator},
    slice::ParallelSlice,
};
use std::{path::PathBuf, process::abort};

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
            let partition = measured_brdf.params.receiver.partitioning();
            let wavelengths = measured_brdf
                .params
                .emitter
                .spectrum
                .values()
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let iors_i = cache
                .iors
                .ior_of_spectrum(measured_brdf.params.incident_medium, &wavelengths)
                .unwrap_or_else(|| {
                    panic!(
                        "missing refractive indices for {:?}",
                        measured_brdf.params.incident_medium
                    )
                });
            let iors_t = cache
                .iors
                .ior_of_spectrum(measured_brdf.params.transmitted_medium, &wavelengths)
                .unwrap_or_else(|| {
                    panic!(
                        "missing refractive indices for {:?}",
                        measured_brdf.params.transmitted_medium
                    )
                });
            let models: Box<[Box<dyn MicrofacetBasedBrdfModel>]> = {
                let mut models = Box::new_uninit_slice(100);
                match opts.model {
                    BrdfModel::Beckmann => {
                        for (i, model) in models.iter_mut().enumerate() {
                            model.write(Box::new(BeckmannBrdfModel::new(
                                (i + 1) as f64 / 100.0,
                                (i + 1) as f64 / 100.0,
                            ))
                                as Box<dyn MicrofacetBasedBrdfModel>);
                        }
                    }
                    BrdfModel::TrowbridgeReitz => {
                        for (i, model) in models.iter_mut().enumerate() {
                            model.write(Box::new(TrowbridgeReitzBrdfModel::new(
                                (i + 1) as f64 / 100.0,
                                (i + 1) as f64 / 100.0,
                            ))
                                as Box<dyn MicrofacetBasedBrdfModel>);
                        }
                    }
                }
                unsafe { models.assume_init() }
            };

            let mut difference = [0.0; 100];
            // Search the best fit for alpha in brute force way between 0 and 1
            measured_brdf.snapshots.iter().for_each(|snapshot| {
                let wi = {
                    let sph = snapshot.w_i;
                    spherical_to_cartesian(1.0, sph.theta, sph.phi)
                };
                let diff = snapshot
                    .samples
                    .iter()
                    .zip(partition.patches.iter())
                    .par_bridge()
                    .map(|(sample, patch)| {
                        let wo = {
                            let sph = patch.center();
                            spherical_to_cartesian(1.0, sph.theta, sph.phi)
                        };
                        let brdf = sample[0] as f64;
                        let mut difference = [0.0; 100];
                        for (i, model) in models.iter().enumerate() {
                            let model_brdf = model.eval(wi, wo, &iors_i[0], &iors_t[0]);
                            difference[i] = (brdf - model_brdf).abs();
                        }
                        difference
                    })
                    .reduce(
                        || [0.0; 100],
                        |mut a, b| {
                            for (a, b) in a.iter_mut().zip(b.iter()) {
                                *a += *b;
                            }
                            a
                        },
                    );
                for (a, b) in difference.iter_mut().zip(diff.iter()) {
                    *a += *b;
                }
            });
            println!(
                "    {}>{} Differences: {:?}",
                ansi::BRIGHT_YELLOW,
                ansi::RESET,
                difference,
            );
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
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrdfModel {
    Beckmann,
    TrowbridgeReitz,
}

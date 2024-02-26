use crate::{
    app::cache::Cache,
    measure::{bsdf::MeasuredBsdfData, data::MeasuredData},
};
use base::math::{spherical_to_cartesian, sqr};
use bxdf::MicrofacetBasedBrdfModel;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use std::sync::atomic::AtomicU64;

/// Compute the mean squared error between the measured data and the model.
///
/// # Arguments
///
/// * `measured` - The measured data.
/// * `model` - The model to compare with the measured data.
/// * `max_theta_o` - The maximum colatitude angle for the outgoing directions.
///   If set, the samples with the outgoing directions with the colatitude angle
///   greater than `max_theta_o` will be ignored.
/// * `cache` - The cache to use for loading the IOR database.
pub fn compute_brdf_mse(
    measured: &MeasuredBsdfData,
    model: &[Box<dyn MicrofacetBasedBrdfModel>],
    max_theta_o: Option<f64>,
    cache: &Cache,
) -> Box<[f64]> {
    let partition = measured.params.receiver.partitioning();
    let wavelengths = measured
        .params
        .emitter
        .spectrum
        .values()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let (iors_i, iors_o) = cache.read(|cache| {
        (
            cache
                .iors
                .ior_of_spectrum(measured.params.incident_medium, &wavelengths)
                .unwrap_or_else(|| {
                    panic!(
                        "missing refractive indices for {:?}",
                        measured.params.incident_medium
                    )
                }),
            cache
                .iors
                .ior_of_spectrum(measured.params.transmitted_medium, &wavelengths)
                .unwrap_or_else(|| {
                    panic!(
                        "missing refractive indices for {:?}",
                        measured.params.transmitted_medium
                    )
                }),
        )
    });

    let max_theta_o = max_theta_o.unwrap_or(95.0f64.to_radians());
    let num_samples = AtomicU64::new(0);
    let mut mses = vec![0.0; model.len()].into_boxed_slice();
    // Iterate over the samples and compute the mean squared error.
    measured.snapshots.iter().for_each(|snapshot| {
        let sqr_err = snapshot
            .samples
            .iter()
            .zip(partition.patches.iter())
            .par_bridge()
            .map(|(sample, patch)| {
                let wo_sph = patch.center();
                if wo_sph.theta.as_f64() > max_theta_o {
                    return vec![0.0; model.len()].into_boxed_slice();
                }
                let wo = spherical_to_cartesian(1.0, wo_sph.theta, wo_sph.phi);
                let brdf = sample[0] as f64;
                let mut sqr_err = vec![0.0; model.len()].into_boxed_slice();
                model
                    .iter()
                    .zip(sqr_err.iter_mut())
                    .for_each(|(model, sqr_err)| {
                        *sqr_err = sqr(model.eval(wo, wo, &iors_i[0], &iors_o[0]) - brdf);
                    });
                num_samples.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                sqr_err
            })
            .reduce(
                || vec![0.0; model.len()].into_boxed_slice(),
                |mut acc, sqr_err| {
                    for (acc, sqr_err) in acc.iter_mut().zip(sqr_err.iter()) {
                        *acc += *sqr_err;
                    }
                    acc
                },
            );
        mses.iter_mut()
            .zip(sqr_err.iter())
            .for_each(|(mse, sqr_err)| {
                *mse += *sqr_err;
            });
    });
    let num_samples = num_samples.load(std::sync::atomic::Ordering::Relaxed) as f64;
    mses.iter_mut().for_each(|mse| {
        *mse /= num_samples;
    });
    mses
}

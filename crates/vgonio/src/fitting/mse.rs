use crate::{
    app::cache::{Cache, RawCache},
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
/// * `max_theta_o` - The maximum colatitude angle in degrees for the outgoing
///   directions. If set, the samples with the outgoing directions with the
///   colatitude angle greater than `max_theta_o` will be ignored.
/// * `cache` - The cache to use for loading the IOR database.
pub fn compute_brdf_mse(
    measured: &MeasuredBsdfData,
    models: &[Box<dyn MicrofacetBasedBrdfModel>],
    max_theta_o: Option<f64>,
    cache: &RawCache,
) -> Box<[f64]> {
    let partition = measured.params.receiver.partitioning();
    let wavelengths = measured
        .params
        .emitter
        .spectrum
        .values()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let (iors_i, iors_o) = (
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
    );

    let max_theta_o = max_theta_o.unwrap_or(95.0).to_radians();
    let mut num_samples = 0.0;
    let mut mses = vec![0.0; models.len()].into_boxed_slice();
    // Iterate over the samples and compute the mean squared error.
    measured.snapshots.iter().for_each(|snapshot| {
        let wi = spherical_to_cartesian(1.0, snapshot.wi.theta, snapshot.wi.phi);
        for (samples, patch) in snapshot.samples.iter().zip(partition.patches.iter()) {
            let sample = samples[0];
            let wo_sph = patch.center();
            if wo_sph.theta.as_f64() > max_theta_o {
                continue;
            }
            let wo = spherical_to_cartesian(1.0, wo_sph.theta, wo_sph.phi);
            let mut sqr_err = vec![0.0; models.len()].into_boxed_slice();
            // Compute the squared error for each model per sample.
            models
                .iter()
                .zip(sqr_err.iter_mut())
                .for_each(|(model, sqr_err)| {
                    *sqr_err = sqr(model.eval(wi, wo, &iors_i[0], &iors_o[0]) - sample as f64);
                });
            num_samples += 1.0;
            // Accumulate the squared error.
            mses.iter_mut()
                .zip(sqr_err.iter())
                .for_each(|(mse, sqr_err)| {
                    *mse += *sqr_err;
                });
        }
    });
    // Compute the mean squared error.
    mses.iter_mut().for_each(|mse| {
        *mse /= num_samples;
    });
    mses
}

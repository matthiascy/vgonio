use crate::{
    app::{cache::RawCache, cli::BrdfModel},
    measure::bsdf::MeasuredBsdfData,
    RangeByStepSizeInclusive,
};
use base::{
    math::{sph_to_cart, sqr, Vec3, Vec3A},
    optics::fresnel,
};
use bxdf::{
    brdf::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel},
    MicrofacetBasedBrdfModel,
};
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use std::sync::atomic::AtomicU64;
use wgpu::naga::TypeInner::Atomic;

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
pub fn compute_iso_brdf_mse(
    measured: &MeasuredBsdfData,
    max_theta_o: Option<f64>,
    model: BrdfModel,
    alpha: RangeByStepSizeInclusive<f64>,
    cache: &RawCache,
) -> Box<[f64]> {
    let count = alpha.step_count();
    let models = match model {
        BrdfModel::Beckmann => {
            let mut models = Vec::with_capacity(count);
            for i in 0..count {
                let alpha_x = i as f64 / count as f64 * alpha.span() + alpha.start;
                let alpha_y = i as f64 / count as f64 * alpha.span() + alpha.start;
                models.push(Box::new(BeckmannBrdfModel::new(alpha_x, alpha_y))
                    as Box<dyn MicrofacetBasedBrdfModel>);
            }
            models.into_boxed_slice()
        }
        BrdfModel::TrowbridgeReitz => {
            let mut models = Vec::with_capacity(count);
            for i in 0..count {
                let alpha_x = i as f64 / count as f64 * alpha.span() + alpha.start;
                let alpha_y = i as f64 / count as f64 * alpha.span() + alpha.start;
                models.push(Box::new(TrowbridgeReitzBrdfModel::new(alpha_x, alpha_y))
                    as Box<dyn MicrofacetBasedBrdfModel>);
            }
            models.into_boxed_slice()
        }
    };
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
    let num_samples = AtomicU64::new(0);

    // Maximum values of the measured samples and the modelled samples for each
    // snapshot.
    let (max_values_measured, max_values_model) = {
        let (max_measured, max_model): (Vec<_>, Vec<_>) = measured
            .snapshots
            .iter()
            .map(|snapshot| {
                let wi = sph_to_cart(snapshot.wi.theta, snapshot.wi.phi);
                let wo: Vec3 = fresnel::reflect(Vec3A::from(-wi), Vec3A::Z).into();
                (
                    snapshot.samples.iter().fold(0.0f64, |m, spectral_samples| {
                        m.max(spectral_samples[0] as f64)
                    }),
                    models
                        .iter()
                        .map(|model| model.eval(wi, wo, &iors_i[0], &iors_o[0]))
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            })
            .unzip();
        (
            max_measured.into_boxed_slice(),
            max_model.into_boxed_slice(),
        )
    };

    // Iterate over the samples and compute the mean squared error.
    let mut mses = measured
        .snapshots
        .iter()
        .enumerate()
        .map(|(snap_idx, snapshot)| {
            let wi = snapshot.wi.to_cartesian();
            snapshot
                .samples
                .iter()
                .zip(partition.patches.iter())
                .par_bridge()
                .map(|(s_samples, patch)| {
                    let sample = s_samples[0];
                    let wo_sph = patch.center();
                    if wo_sph.theta.as_f64() > max_theta_o {
                        return vec![0.0; models.len()].into_boxed_slice();
                    }
                    let wo = wo_sph.to_cartesian();
                    let mut sqr_err = vec![0.0; models.len()].into_boxed_slice();
                    // Compute the squared error for each model per sample.
                    models.iter().zip(sqr_err.iter_mut()).enumerate().for_each(
                        |(i, (model, sqr_err))| {
                            *sqr_err = sqr(model.eval(wi, wo, &iors_i[0], &iors_o[0])
                                / max_values_model[snap_idx][i]
                                - sample as f64 / max_values_measured[snap_idx]);
                        },
                    );
                    num_samples.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    sqr_err
                })
                .reduce(
                    || vec![0.0; models.len()].into_boxed_slice(),
                    sqr_err_accumulator,
                )
        })
        .fold(
            vec![0.0; models.len()].into_boxed_slice(),
            sqr_err_accumulator,
        );

    let n_samples = num_samples.load(std::sync::atomic::Ordering::Relaxed) as f64;
    // Compute the mean squared error.
    mses.iter_mut().for_each(|mse| {
        *mse /= n_samples;
    });
    mses
}

// Accumulates the squared error.
fn sqr_err_accumulator(mut acc: Box<[f64]>, sqr_err: Box<[f64]>) -> Box<[f64]> {
    acc.iter_mut()
        .zip(sqr_err.iter())
        .for_each(|(err, sqr_err)| {
            *err += *sqr_err;
        });
    acc
}

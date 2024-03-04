use crate::{
    app::{
        cache::{RawCache, RefractiveIndexRegistry},
        cli::BrdfModel,
    },
    measure::{
        bsdf::{BsdfSnapshot, MeasuredBsdfData, SpectralSamples},
        params::BsdfMeasurementParams,
    },
    partition::SphericalPartition,
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
use rand::seq::index::sample;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelBridge,
    ParallelIterator,
};
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
pub fn compute_iso_brdf_mse(
    measured: &MeasuredBsdfData,
    max_theta_o: Option<f64>,
    model: BrdfModel,
    alpha: RangeByStepSizeInclusive<f64>,
    cache: &RawCache,
) -> Box<[f64]> {
    let count = alpha.step_count();
    let partition = measured.params.receiver.partitioning();
    let (models, max_values): (Vec<_>, Vec<_>) = (0..count)
        .into_par_iter()
        .chunks(256)
        .map(|chunk| {
            chunk
                .into_iter()
                .map(|i| {
                    let alpha_x = i as f64 / count as f64 * alpha.span() + alpha.start;
                    let alpha_y = i as f64 / count as f64 * alpha.span() + alpha.start;
                    let m = match model {
                        BrdfModel::Beckmann => Box::new(BeckmannBrdfModel::new(alpha_x, alpha_y))
                            as Box<dyn MicrofacetBasedBrdfModel>,
                        BrdfModel::TrowbridgeReitz => {
                            Box::new(TrowbridgeReitzBrdfModel::new(alpha_x, alpha_y))
                                as Box<dyn MicrofacetBasedBrdfModel>
                        }
                    };
                    generate_analytical_brdf(&measured.params, &m, &cache.iors)
                })
                .collect::<Vec<_>>()
        })
        .reduce(
            || vec![],
            |mut acc, x| {
                acc.extend(x);
                acc
            },
        )
        .into_iter()
        .unzip();

    let max_theta_o = max_theta_o.unwrap_or(95.0).to_radians();
    // Maximum values of the measured samples for each snapshot. Only the first
    // spectral sample is considered.
    let max_measured = measured
        .snapshots
        .par_iter()
        .map(|snapshot| {
            snapshot.samples.iter().fold(0.0f64, |m, spectral_samples| {
                m.max(spectral_samples[0] as f64)
            })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    models
        .iter()
        .zip(max_values.iter())
        .par_bridge()
        .map(|(model, model_max_values)| {
            compute_mse(
                model,
                measured,
                &partition,
                model_max_values,
                &max_measured,
                max_theta_o,
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn compute_mse(
    model: &MeasuredBsdfData,
    measured: &MeasuredBsdfData,
    partition: &SphericalPartition,
    model_max: &[Box<[f64]>],
    measured_max: &Box<[f64]>,
    max_theta_o: f64,
) -> f64 {
    let mut n = 0;
    let mut mse = 0.0;

    for i in 0..model.snapshots.len() {
        for j in 0..partition.patches.len() {
            let patch = &partition.patches[j];
            let wo_sph = patch.center();
            if wo_sph.theta.as_f64() > max_theta_o {
                continue;
            }
            n += 1;
            let sample_model = model.snapshots[i].samples[j][0] as f64 / model_max[i][0];
            let sample_measured = measured.snapshots[i].samples[j][0] as f64 / measured_max[i];
            mse += sqr(sample_model - sample_measured);
        }
    }

    mse / n as f64
}

// TODO: use specific container instead of MeasuredBsdfData
pub(crate) fn generate_analytical_brdf(
    params: &BsdfMeasurementParams,
    target: &Box<dyn MicrofacetBasedBrdfModel>,
    iors: &RefractiveIndexRegistry,
) -> (MeasuredBsdfData, Box<[Box<[f64]>]>) {
    let mut brdf = MeasuredBsdfData {
        params: params.clone(),
        snapshots: Box::new([]),
        raw_snapshots: None,
    };
    let partition = brdf.params.receiver.partitioning();
    let mut snapshots = vec![];
    let meas_pts = brdf.params.emitter.generate_measurement_points();
    let wavelengths = params
        .emitter
        .spectrum
        .values()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let iors_i = iors
        .ior_of_spectrum(params.incident_medium, &wavelengths)
        .unwrap();
    let iors_t = iors
        .ior_of_spectrum(params.transmitted_medium, &wavelengths)
        .unwrap();
    let mut max_values = vec![vec![-1.0f64; wavelengths.len()].into_boxed_slice(); meas_pts.len()]
        .into_boxed_slice();
    for (wi_sph, current_max_values) in meas_pts.iter().zip(max_values.iter_mut()) {
        let wi = wi_sph.to_cartesian();
        let mut samples = vec![];
        let mut max_values_per_snapshot = vec![-1.0; wavelengths.len()];
        for patch in partition.patches.iter() {
            let wo_sph = patch.center();
            let wo = sph_to_cart(wo_sph.theta, wo_sph.phi);
            let spectral_samples = target
                .eval_spectrum(wi, wo, &iors_i, &iors_t)
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>();
            for (i, sample) in spectral_samples.iter().enumerate() {
                max_values_per_snapshot[i] = f64::max(max_values_per_snapshot[i], *sample as f64);
            }
            samples.push(SpectralSamples::from_vec(spectral_samples));
        }
        *current_max_values = max_values_per_snapshot.into_boxed_slice();
        snapshots.push(BsdfSnapshot {
            wi: *wi_sph,
            samples: samples.into_boxed_slice(),
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            trajectories: vec![],
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            hit_points: vec![],
        });
    }
    brdf.snapshots = snapshots.into_boxed_slice();
    (brdf, max_values)
}

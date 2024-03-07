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
    brdf::microfacet::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel},
    MicrofacetBasedBrdfModel,
};
use rand::seq::index::sample;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::{mem::MaybeUninit, sync::atomic::AtomicU64};

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
    normalize: bool,
) -> Box<[f64]> {
    // TODO: remove max_theta_o
    let count = alpha.step_count();
    let partition = measured.params.receiver.partitioning();
    const CHUNK_SIZE: usize = 32;
    let mut brdfs = {
        let mut brdfs = Box::new_uninit_slice(count);
        brdfs
            .chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(i, brdf_data)| {
                for j in 0..brdf_data.len() {
                    let alpha_x =
                        (i * CHUNK_SIZE + j) as f64 / count as f64 * alpha.span() + alpha.start;
                    let alpha_y =
                        (i * CHUNK_SIZE + j) as f64 / count as f64 * alpha.span() + alpha.start;
                    let m = match model {
                        BrdfModel::Beckmann => Box::new(BeckmannBrdfModel::new(alpha_x, alpha_y))
                            as Box<dyn MicrofacetBasedBrdfModel>,
                        BrdfModel::TrowbridgeReitz => {
                            Box::new(TrowbridgeReitzBrdfModel::new(alpha_x, alpha_y))
                                as Box<dyn MicrofacetBasedBrdfModel>
                        }
                    };
                    let (model_brdf, _) =
                        generate_analytical_brdf(&measured.params, &m, &cache.iors, normalize);
                    brdf_data[j].write(model_brdf);
                }
            });
        unsafe { brdfs.assume_init() }
    };

    println!(" Finished generating analytical BRDFs");

    let max_theta_o = max_theta_o.unwrap_or(95.0).to_radians();
    // Maximum values of the measured samples for each snapshot. Only the first
    // spectral sample is considered.
    let max_measured = if normalize {
        measured
            .snapshots
            .par_iter()
            .map(|snapshot| {
                snapshot
                    .samples
                    .iter()
                    .fold(0.0f64, |m, s| m.max(s[0] as f64))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    } else {
        vec![1.0; measured.snapshots.len()].into_boxed_slice()
    };

    let mses = brdfs
        .par_iter()
        .map(|(model)| compute_mse(model, measured, &max_measured, &partition, max_theta_o))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    mses
}

fn compute_mse(
    model: &MeasuredBsdfData,
    measured: &MeasuredBsdfData,
    measured_max_per_snapshot: &[f64],
    partition: &SphericalPartition,
    max_theta_o: f64,
) -> f64 {
    let count = AtomicU64::new(0);
    let patches = &partition.patches;
    let mse_sum = model
        .snapshots
        .par_iter()
        .zip(measured.snapshots.par_iter())
        .zip(measured_max_per_snapshot.par_iter())
        .map(|((model_snapshot, measured_snapshot), measured_max)| {
            if model_snapshot.wi.theta.as_f64() > max_theta_o {
                return 0.0;
            }
            model_snapshot
                .samples
                .iter()
                .zip(measured_snapshot.samples.iter())
                .zip(patches.iter())
                .map(|((model_samples, measured_samples), patch)| {
                    if patch.center().theta.as_f64() > max_theta_o {
                        0.0
                    } else {
                        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        sqr(model_samples[0] as f64 - measured_samples[0] as f64 / *measured_max)
                    }
                })
                .sum::<f64>()
        })
        .sum::<f64>();
    let n = count.load(std::sync::atomic::Ordering::Relaxed);
    mse_sum / n as f64
}

// TODO: use specific container instead of MeasuredBsdfData
pub(crate) fn generate_analytical_brdf(
    params: &BsdfMeasurementParams,
    target: &Box<dyn MicrofacetBasedBrdfModel>,
    iors: &RefractiveIndexRegistry,
    normalize: bool,
) -> (MeasuredBsdfData, Box<[Box<[f64]>]>) {
    let mut brdf = MeasuredBsdfData {
        params: params.clone(),
        snapshots: Box::new([]),
        raw_snapshots: None,
    };
    let partition = brdf.params.receiver.partitioning();
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
    let mut snapshots = Box::new_uninit_slice(meas_pts.len());
    meas_pts
        .par_iter()
        .zip(max_values.par_iter_mut())
        .zip(snapshots.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((wi_sph, snap_max_values), snapshot))| {
            let wi = wi_sph.to_cartesian();
            let mut samples = vec![];
            let mut max_values_per_snapshot = vec![-1.0; wavelengths.len()];
            for patch in partition.patches.iter() {
                let wo_sph = patch.center();
                let wo = sph_to_cart(wo_sph.theta, wo_sph.phi);
                let mut spectral_samples = target
                    .eval_spectrum(wi, wo, &iors_i, &iors_t)
                    .iter()
                    .map(|&x| x as f32)
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                for (i, sample) in spectral_samples.iter().enumerate() {
                    max_values_per_snapshot[i] =
                        f64::max(max_values_per_snapshot[i], *sample as f64);
                }
                samples.push(SpectralSamples::from_boxed_slice(spectral_samples));
            }
            if normalize {
                for sample in samples.iter_mut() {
                    for (s, max) in sample.iter_mut().zip(max_values_per_snapshot.iter()) {
                        *s /= *max as f32;
                    }
                }
            }
            *snap_max_values = max_values_per_snapshot.into_boxed_slice();
            snapshot.write(BsdfSnapshot {
                wi: *wi_sph,
                samples: samples.into_boxed_slice(),
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                trajectories: vec![],
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                hit_points: vec![],
            });
        });
    brdf.snapshots = unsafe { snapshots.assume_init() };
    (brdf, max_values)
}

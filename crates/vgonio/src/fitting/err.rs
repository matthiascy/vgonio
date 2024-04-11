use crate::{
    app::cache::{RawCache, RefractiveIndexRegistry},
    measure::{
        bsdf::{BsdfSnapshot, MeasuredBsdfData, SpectralSamples},
        data::SampledBrdf,
        params::BsdfMeasurementParams,
    },
    partition::SphericalPartition,
};
use base::{
    math::{sph_to_cart, sqr, Sph2, Vec3, Vec3A},
    medium::Medium,
    optics::fresnel,
    range::RangeByStepSizeInclusive,
    units::{deg, Degrees, Radians},
};
use bxdf::{
    brdf::{
        microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf, TrowbridgeReitzBrdfModel},
        Bxdf,
    },
    distro::{BeckmannDistribution, MicrofacetDistroKind},
    Scattering,
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::sync::atomic::AtomicU64;

/// Metrics to use for the error computation.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorMetric {
    /// Mean squared error.
    Mse,
    /// Most commonly used error metrics in non-linear least squares fitting.
    /// Which is the half of the sum of the squares of the differences between
    /// the measured data and the model.
    #[default]
    Nlls,
}

/// Compute the distance (error) between the measured data and the model.
///
/// # Arguments
///
/// * `measured` - The measured data.
/// * `model` - The model to compare with the measured data.
/// * `max_theta_o` - The maximum colatitude angle in degrees for the outgoing
///   directions. If set, the samples with the outgoing directions with the
///   colatitude angle greater than `max_theta_o` will be ignored.
/// * `cache` - The cache to use for loading the IOR database.
pub fn compute_iso_microfacet_brdf_err(
    measured: &MeasuredBsdfData,
    max_theta_o: Option<Degrees>,
    distro: MicrofacetDistroKind,
    alpha: RangeByStepSizeInclusive<f64>,
    cache: &RawCache,
    normalize: bool,
    metric: ErrorMetric,
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
                    let alpha_x = (i * CHUNK_SIZE + j) as f64 * alpha.step_size + alpha.start;
                    let alpha_y = (i * CHUNK_SIZE + j) as f64 * alpha.step_size + alpha.start;
                    let m = match distro {
                        MicrofacetDistroKind::Beckmann => {
                            Box::new(BeckmannBrdf::new(alpha_x, alpha_y))
                                as Box<dyn Bxdf<Params = [f64; 2]>>
                        }
                        MicrofacetDistroKind::TrowbridgeReitz => {
                            Box::new(TrowbridgeReitzBrdf::new(alpha_x, alpha_y))
                                as Box<dyn Bxdf<Params = [f64; 2]>>
                        }
                    };
                    let (model_brdf, _) =
                        generate_analytical_brdf(&measured.params, &*m, &cache.iors, normalize);
                    brdf_data[j].write(model_brdf);
                }
            });
        unsafe { brdfs.assume_init() }
    };

    println!(" Finished generating analytical BRDFs");

    let max_theta_o = max_theta_o.unwrap_or(deg!(90.0)).to_radians();
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
        .map(|(model)| {
            compute_distance(
                model,
                measured,
                &max_measured,
                &partition,
                max_theta_o,
                metric,
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    mses
}

fn compute_distance(
    normalized_model: &MeasuredBsdfData,
    measured: &MeasuredBsdfData,
    measured_max_per_snapshot: &[f64],
    partition: &SphericalPartition,
    max_theta_o: Radians,
    metric: ErrorMetric,
) -> f64 {
    let count = AtomicU64::new(0);
    let patches = &partition.patches;
    let sqr_err_sum = normalized_model
        .snapshots
        .par_iter()
        .zip(measured.snapshots.par_iter())
        .zip(measured_max_per_snapshot.par_iter())
        .map(|((model_snapshot, measured_snapshot), measured_max)| {
            if model_snapshot.wi.theta > max_theta_o {
                return 0.0;
            }
            model_snapshot
                .samples
                .iter()
                .zip(measured_snapshot.samples.iter())
                .zip(patches.iter())
                .map(|((model_samples, measured_samples), patch)| {
                    if patch.center().theta > max_theta_o {
                        0.0
                    } else {
                        count.fetch_add(
                            model_samples.len() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        let mut sum = 0.0;
                        for (model_sample, measured_sample) in
                            model_samples.iter().zip(measured_samples.iter())
                        {
                            sum +=
                                sqr(*model_sample as f64 - *measured_sample as f64 / *measured_max);
                        }
                        sum
                    }
                })
                .sum::<f64>()
        })
        .sum::<f64>();
    match metric {
        ErrorMetric::Mse => {
            let n = patches
                .iter()
                .map(|p| p.center().theta <= max_theta_o)
                .count()
                * measured.snapshots.len();
            sqr_err_sum / n as f64
        }
        ErrorMetric::Nlls => sqr_err_sum * 0.5,
    }
}

// TODO: all wavelengths are considered for the error computation
/// Returns the maximum values of each snapshot for the target BRDF.
pub(crate) fn generate_analytical_brdf_from_sampled_brdf(
    sampled_brdf: &SampledBrdf,
    target: &dyn Bxdf<Params = [f64; 2]>,
    iors: &RefractiveIndexRegistry,
    normalise: bool,
) -> SampledBrdf {
    let iors_i = iors
        .ior_of_spectrum(Medium::Air, &sampled_brdf.spectrum)
        .unwrap();
    let iors_t = iors
        .ior_of_spectrum(Medium::Aluminium, &sampled_brdf.spectrum)
        .unwrap();
    let mut samples = vec![0.0f32; sampled_brdf.samples.len()].into_boxed_slice();
    let mut max_values =
        vec![-1.0f32; sampled_brdf.spectrum.len() * sampled_brdf.wi_wo_pairs.len()]
            .into_boxed_slice();
    sampled_brdf
        .wi_wo_pairs
        .iter()
        .zip(max_values.chunks_mut(sampled_brdf.spectrum.len()))
        .for_each(|((wi, wos, offset), max_values)| {
            let offset = *offset as usize * sampled_brdf.spectrum.len();
            wos.iter().enumerate().for_each(|(i, wo)| {
                let samples_offset = offset + i * sampled_brdf.spectrum.len();
                let samples =
                    &mut samples[samples_offset..samples_offset + sampled_brdf.spectrum.len()];
                let wi = wi.to_cartesian();
                let wo = wo.to_cartesian();
                let spectral_samples =
                    Scattering::eval_reflectance_spectrum(target, &wi, &wo, &iors_i, &iors_t);
                samples
                    .iter_mut()
                    .zip(spectral_samples.iter())
                    .zip(max_values.iter_mut())
                    .for_each(|((sample, value), max)| {
                        *sample = *value as f32;
                        *max = f32::max(*max, *sample);
                    });
            })
        });
    if normalise {
        for (i, (_, wos, offset)) in sampled_brdf.wi_wo_pairs.iter().enumerate() {
            let offset = *offset as usize * sampled_brdf.spectrum.len();
            let max_values =
                &max_values[i * sampled_brdf.spectrum.len()..(i + 1) * sampled_brdf.spectrum.len()];
            for (sample, max) in samples[offset..offset + sampled_brdf.spectrum.len() * wos.len()]
                .iter_mut()
                .zip(max_values.iter())
            {
                *sample /= *max;
            }
        }
    }
    SampledBrdf {
        spectrum: sampled_brdf.spectrum.clone(),
        samples,
        max_values,
        wi_wo_pairs: sampled_brdf.wi_wo_pairs.clone(),
        num_pairs: sampled_brdf.num_pairs,
    }
}

pub fn compute_iso_sampled_brdf_err(
    sampled: &SampledBrdf,
    distro: MicrofacetDistroKind,
    alpha: RangeByStepSizeInclusive<f64>,
    cache: &RawCache,
    normalise: bool,
    metric: ErrorMetric,
) -> Box<[f64]> {
    let mut brdfs = {
        let mut brdfs = Box::new_uninit_slice(alpha.step_count());
        brdfs.iter_mut().enumerate().for_each(|(i, brdf)| {
            let alpha_x = i as f64 * alpha.step_size + alpha.start;
            let alpha_y = i as f64 * alpha.step_size + alpha.start;
            let m = match distro {
                MicrofacetDistroKind::Beckmann => Box::new(BeckmannBrdf::new(alpha_x, alpha_y))
                    as Box<dyn Bxdf<Params = [f64; 2]>>,
                MicrofacetDistroKind::TrowbridgeReitz => {
                    Box::new(TrowbridgeReitzBrdf::new(alpha_x, alpha_y))
                        as Box<dyn Bxdf<Params = [f64; 2]>>
                }
            };
            let model_brdf =
                generate_analytical_brdf_from_sampled_brdf(sampled, &*m, &cache.iors, normalise);
            brdf.write(model_brdf);
        });
        unsafe { brdfs.assume_init() }
    };

    brdfs
        .par_iter()
        .map(|model| compute_distance_from_sampled_brdf(model, sampled, metric))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn compute_distance_from_sampled_brdf(
    modelled: &SampledBrdf,
    measured: &SampledBrdf,
    error_metric: ErrorMetric,
) -> f64 {
    let sqr_err_sum = modelled
        .samples
        .iter()
        .step_by(modelled.spectrum.len())
        .zip(measured.samples.iter().step_by(measured.spectrum.len()))
        .map(|(model_samples, measured_samples)| {
            sqr(*model_samples as f64 - *measured_samples as f64)
        })
        .sum::<f64>();
    match error_metric {
        ErrorMetric::Mse => {
            let n = modelled.wi_wo_pairs.len() * modelled.spectrum.len();
            sqr_err_sum / n as f64
        }
        ErrorMetric::Nlls => sqr_err_sum * 0.5,
    }
}

// TODO: use specific container instead of MeasuredBsdfData
/// Generate the analytical BRDFs for the given model.
///
/// # Returns
///
/// The generated BRDFs and the maximum values of the spectral samples for each
/// snapshot and each wavelength. The first dimension of the returned array
/// corresponds to the snapshot index, the second dimension corresponds to the
/// wavelength index (Row-major order).
pub(crate) fn generate_analytical_brdf(
    params: &BsdfMeasurementParams,
    target: &dyn Bxdf<Params = [f64; 2]>,
    iors: &RefractiveIndexRegistry,
    normalise: bool,
) -> (MeasuredBsdfData, Box<[f64]>) {
    let mut brdf = MeasuredBsdfData {
        params: params.clone(),
        snapshots: Box::new([]),
        raw_snapshots: None,
        normalised: normalise,
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
    let mut max_values = vec![-1.0f64; wavelengths.len() * meas_pts.len()].into_boxed_slice();
    let mut snapshots = Box::new_uninit_slice(meas_pts.len());
    meas_pts
        .par_iter()
        .zip(max_values.par_chunks_mut(wavelengths.len()))
        .zip(snapshots.par_iter_mut())
        .chunks(32)
        .for_each(|chunk| {
            for ((wi_sph, max_values_per_snapshot), snapshot) in chunk {
                let wi = wi_sph.to_cartesian();
                let mut samples = vec![];
                for patch in partition.patches.iter() {
                    let wo_sph = patch.center();
                    let wo = sph_to_cart(wo_sph.theta, wo_sph.phi);
                    let mut spectral_samples =
                        Scattering::eval_reflectance_spectrum(target, &wi, &wo, &iors_i, &iors_t)
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
                if normalise {
                    for sample in samples.iter_mut() {
                        for (s, max) in sample.iter_mut().zip(max_values_per_snapshot.iter()) {
                            *s /= *max as f32;
                        }
                    }
                }
                snapshot.write(BsdfSnapshot {
                    wi: *wi_sph,
                    samples: samples.into_boxed_slice(),
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    trajectories: vec![],
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    hit_points: vec![],
                });
            }
        });
    brdf.snapshots = unsafe { snapshots.assume_init() };
    (brdf, max_values)
}

use crate::{
    app::cache::{RawCache, RefractiveIndexRegistry},
    measure::{
        bsdf::{BsdfSnapshot, MeasuredBsdfData},
        params::BsdfMeasurementParams,
    },
};
use base::{
    math::{sph_to_cart, sqr},
    medium::Medium,
    partition::SphericalPartition,
    range::RangeByStepSizeInclusive,
    units::{deg, Degrees, Radians},
    ErrorMetric,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::{ClausenBrdf, Origin},
        Bxdf,
    },
    distro::MicrofacetDistroKind,
    Scattering,
};
use jabr::array::DyArr;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::sync::atomic::AtomicU64;

// TODO: move under bxdf crate

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
    metric: ErrorMetric,
) -> Box<[f64]> {
    // TODO: remove max_theta_o
    let count = alpha.step_count();
    let partition = measured.params.receiver.partitioning();
    const CHUNK_SIZE: usize = 32;
    let brdfs = {
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
                    let model_brdf = generate_analytical_brdf(&measured.params, &*m, &cache.iors);
                    brdf_data[j].write(model_brdf);
                }
            });
        unsafe { brdfs.assume_init() }
    };

    println!(" Finished generating analytical BRDFs");

    let max_theta_o = max_theta_o.unwrap_or(deg!(90.0)).to_radians();

    let mses = brdfs
        .par_iter()
        .map(|model| compute_distance(model, measured, &partition, max_theta_o, metric))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    mses
}

fn compute_distance(
    normalized_model: &MeasuredBsdfData,
    measured: &MeasuredBsdfData,
    partition: &SphericalPartition,
    max_theta_o: Radians,
    metric: ErrorMetric,
) -> f64 {
    let count = AtomicU64::new(0);
    let patches = &partition.patches;
    let n_lambda = measured.params.emitter.spectrum.step_count();
    let sqr_err_sum = normalized_model
        .snapshots
        .par_iter()
        .zip(measured.snapshots.par_iter())
        .map(|(model_snapshot, measured_snapshot)| {
            if model_snapshot.wi.theta > max_theta_o {
                return 0.0;
            }
            model_snapshot
                .samples
                .as_slice()
                .chunks(n_lambda)
                .zip(measured_snapshot.samples.as_slice().chunks(n_lambda))
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
                            sum += sqr(*model_sample as f64 - *measured_sample as f64);
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

pub(crate) fn generate_analytical_brdf_from_clausen_brdf(
    brdf: &ClausenBrdf,
    target: &dyn Bxdf<Params = [f64; 2]>,
    iors: &RefractiveIndexRegistry,
) -> ClausenBrdf {
    let iors_i = iors
        .ior_of_spectrum(Medium::Air, brdf.spectrum.as_ref())
        .unwrap();
    let iors_t = iors
        .ior_of_spectrum(Medium::Aluminium, brdf.spectrum.as_ref())
        .unwrap();
    let n_spectrum = brdf.n_spectrum();
    let n_wo = brdf.n_wo();
    let mut samples = DyArr::zeros([brdf.n_wi(), n_wo, n_spectrum]);
    brdf.params
        .incoming
        .as_slice()
        .iter()
        .enumerate()
        .zip(brdf.params.outgoing.as_slice().chunks(n_wo))
        .for_each(|((i, wi), wos)| {
            let wi = wi.to_cartesian();
            wos.iter().enumerate().for_each(|(j, wo)| {
                let wo = wo.to_cartesian();
                let spectral_samples =
                    Scattering::eval_reflectance_spectrum(target, &wi, &wo, &iors_i, &iors_t);
                for (k, sample) in spectral_samples.iter().enumerate() {
                    samples[[i, j, k]] = *sample as f32;
                }
            });
        });
    ClausenBrdf::new(
        Origin::Analytical,
        brdf.incident_medium,
        brdf.transmitted_medium,
        brdf.params.clone(),
        brdf.spectrum.clone(),
        samples,
    )
}

pub fn compute_iso_clausen_brdf_err(
    sampled: &ClausenBrdf,
    distro: MicrofacetDistroKind,
    alpha: RangeByStepSizeInclusive<f64>,
    cache: &RawCache,
    metric: ErrorMetric,
) -> Box<[f64]> {
    let modelled_brdfs = {
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
            let model_brdf = generate_analytical_brdf_from_clausen_brdf(sampled, &*m, &cache.iors);
            brdf.write(model_brdf);
        });
        unsafe { brdfs.assume_init() }
    };

    modelled_brdfs
        .par_iter()
        .map(|model| model.distance(sampled, metric))
        .collect::<Vec<_>>()
        .into_boxed_slice()
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
) -> MeasuredBsdfData {
    let partition = params.receiver.partitioning();
    let meas_pts = params.emitter.generate_measurement_points();
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
    let n_wavelength = wavelengths.len();
    let n_wi = meas_pts.len();
    let n_patches = partition.n_patches();
    let mut snapshots = Box::new_uninit_slice(meas_pts.len());
    meas_pts
        .par_iter()
        .zip(snapshots.par_iter_mut())
        .chunks(32)
        .for_each(|chunk| {
            for (wi_sph, snapshot) in chunk {
                let wi = wi_sph.to_cartesian();
                let mut samples = DyArr::zeros([n_patches, n_wavelength]);
                for (i, patch) in partition.patches.iter().enumerate() {
                    let wo_sph = patch.center();
                    let wo = sph_to_cart(wo_sph.theta, wo_sph.phi);
                    let spectral_samples =
                        Scattering::eval_reflectance_spectrum(target, &wi, &wo, &iors_i, &iors_t)
                            .iter()
                            .map(|&x| x as f32)
                            .collect::<Vec<_>>()
                            .into_boxed_slice();
                    samples.as_mut_slice()[i * n_wavelength..(i + 1) * n_wavelength]
                        .copy_from_slice(&spectral_samples);
                }
                snapshot.write(BsdfSnapshot {
                    wi: *wi_sph,
                    samples,
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    trajectories: vec![],
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    hit_points: vec![].into_boxed_slice(),
                });
            }
        });
    MeasuredBsdfData {
        params: params.clone(),
        snapshots: unsafe { snapshots.assume_init() },
        raw_snapshots: None,
    }
}

//! Error computation for fitting measured data to a model.

use base::{
    optics::ior::RefractiveIndexRegistry, range::RangeByStepSizeInclusive, units::Radians,
    ErrorMetric,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::AnalyticalFit,
        Bxdf,
    },
    distro::MicrofacetDistroKind,
};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

/// Compute the distance (error) between the measured data and the model.
pub fn compute_microfacet_brdf_err<M: AnalyticalFit + Sync>(
    measured: &M,
    distro: MicrofacetDistroKind,
    alpha: RangeByStepSizeInclusive<f64>,
    iors: &RefractiveIndexRegistry,
    theta_limit: Radians,
    metric: ErrorMetric,
) -> Box<[f64]> {
    let count = alpha.step_count();
    let equals_half_pi = approx::abs_diff_eq!(
        theta_limit.as_f64(),
        std::f64::consts::FRAC_PI_2,
        epsilon = 1e-6
    );

    if equals_half_pi {
        log::debug!("compute full distance");
    } else {
        log::debug!("compute filtered distance");
    }

    const CHUNK_SIZE: usize = 32;
    let mut errs = Box::new_uninit_slice(count);
    errs.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, err_chunks)| {
            for j in 0..err_chunks.len() {
                let alpha_x = (i * CHUNK_SIZE + j) as f64 * alpha.step_size + alpha.start;
                let alpha_y = (i * CHUNK_SIZE + j) as f64 * alpha.step_size + alpha.start;
                let m = match distro {
                    MicrofacetDistroKind::Beckmann => Box::new(BeckmannBrdf::new(alpha_x, alpha_y))
                        as Box<dyn Bxdf<Params = [f64; 2]>>,
                    MicrofacetDistroKind::TrowbridgeReitz => {
                        Box::new(TrowbridgeReitzBrdf::new(alpha_x, alpha_y))
                            as Box<dyn Bxdf<Params = [f64; 2]>>
                    },
                };
                let modelled = measured.new_analytical_from_self(&*m, iors);
                if equals_half_pi {
                    err_chunks[j].write(measured.distance(&modelled, metric));
                } else {
                    err_chunks[j].write(measured.filtered_distance(&modelled, metric, theta_limit));
                }
            }
        });
    unsafe { errs.assume_init() }
}

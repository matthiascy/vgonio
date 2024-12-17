//! Error computation for fitting measured data to a model.

use crate::fitting::Weighting;
use base::{optics::ior::IorRegistry, range::StepRangeIncl, units::Radians, ErrorMetric};
use bxdf::{
    brdf::{
        analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR},
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
    alpha: StepRangeIncl<f64>,
    iors: &IorRegistry,
    theta_limit: Radians,
    metric: ErrorMetric,
    rmetric: Weighting,
) -> Box<[f64]> {
    let count = alpha.step_count();
    let equals_half_pi = approx::abs_diff_eq!(
        theta_limit.as_f64(),
        std::f64::consts::FRAC_PI_2,
        epsilon = 1e-6
    );

    const CHUNK_SIZE: usize = 32;
    let mut errs = Box::new_uninit_slice(count);
    errs.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, err_chunks)| {
            for j in 0..err_chunks.len() {
                let alpha_x = (i * CHUNK_SIZE + j) as f64 * alpha.step_size + alpha.start;
                let alpha_y = (i * CHUNK_SIZE + j) as f64 * alpha.step_size + alpha.start;
                let m = match distro {
                    MicrofacetDistroKind::Beckmann => {
                        Box::new(MicrofacetBrdfBK::new(alpha_x, alpha_y))
                            as Box<dyn Bxdf<Params = [f64; 2]>>
                    },
                    MicrofacetDistroKind::TrowbridgeReitz => {
                        Box::new(MicrofacetBrdfTR::new(alpha_x, alpha_y))
                            as Box<dyn Bxdf<Params = [f64; 2]>>
                    },
                };
                let modelled = measured.new_analytical_from_self(&*m, iors);
                if equals_half_pi {
                    err_chunks[j].write(measured.distance(&modelled, metric, rmetric));
                } else {
                    err_chunks[j].write(measured.filtered_distance(
                        &modelled,
                        metric,
                        rmetric,
                        theta_limit,
                    ));
                }
            }
        });
    unsafe { errs.assume_init() }
}

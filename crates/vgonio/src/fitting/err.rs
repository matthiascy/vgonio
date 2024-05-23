use crate::app::cache::RawCache;
use base::{range::RangeByStepSizeInclusive, ErrorMetric};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::AnalyticalFit,
        Bxdf,
    },
    distro::MicrofacetDistroKind,
};

/// Compute the distance (error) between the measured data and the model.
pub fn compute_microfacet_brdf_err(
    measured: &impl AnalyticalFit,
    distro: MicrofacetDistroKind,
    alpha: RangeByStepSizeInclusive<f64>,
    cache: &RawCache,
    metric: ErrorMetric,
) -> Box<[f64]> {
    let count = alpha.step_count();
    const CHUNK_SIZE: usize = 32;
    let mut errs = Box::new_uninit_slice(count);
    errs.chunks_mut(CHUNK_SIZE)
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
                    }
                };
                let modelled = measured.new_analytical_from_self(&*m, &cache.iors);
                err_chunks[j].write(measured.distance(&modelled, metric));
            }
        });
    unsafe { errs.assume_init() }
}

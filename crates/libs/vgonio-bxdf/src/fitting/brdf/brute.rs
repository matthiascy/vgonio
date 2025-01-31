use crate::brdf::analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use vgonio_core::{
    bxdf::{AnalyticalBrdf, BrdfProxy, MicrofacetDistroKind},
    optics::IorReg,
    units::Radians,
    utils::range::StepRangeIncl,
    AnyMeasuredBrdf, ErrorMetric, Weighting,
};

/// Compute the distance between a measured BRDF and a modelled BRDF.
pub fn compute_distance_between_measured_and_modelled(
    measured: &BrdfProxy,
    distro: MicrofacetDistroKind,
    metric: ErrorMetric,
    weighting: Weighting,
    alphax: f64,
    alphay: f64,
    max_theta_i: Radians,
    max_theta_o: Radians,
) -> f64 {
    let m = match distro {
        MicrofacetDistroKind::Beckmann => Box::new(MicrofacetBrdfBK::new(alphax, alphay))
            as Box<dyn AnalyticalBrdf<Params = [f64; 2]>>,
        MicrofacetDistroKind::TrowbridgeReitz => Box::new(MicrofacetBrdfTR::new(alphax, alphay))
            as Box<dyn AnalyticalBrdf<Params = [f64; 2]>>,
    };
    let modelled = measured.generate_analytical(&*m);
    let filtering = !(max_theta_i >= Radians::HALF_PI && max_theta_o >= Radians::HALF_PI);
    if filtering {
        measured.distance_filtered(
            &modelled,
            metric,
            weighting,
            max_theta_i.as_f32(),
            max_theta_o.as_f32(),
        )
    } else {
        measured.distance(&modelled, metric, weighting)
    }
}

/// Brute force fitting for isotropic microfacet BRDFs.
pub fn brdf_fitting_brute_force_isotropic<F: AnyMeasuredBrdf>(
    measured: &F,
    distro: MicrofacetDistroKind,
    metric: ErrorMetric,
    max_theta_i: Radians,
    max_theta_o: Radians,
    weighting: Weighting,
    alpha: StepRangeIncl<f64>,
    iors: &IorReg,
) -> Box<[f64]> {
    log::debug!("compute isotropic distance");
    let count = alpha.step_count();
    let mut errs = Box::new_uninit_slice(count);
    let cpu_count = ((std::thread::available_parallelism().unwrap().get()) / 2).max(1);
    let chunk_size = errs.len().div_ceil(cpu_count);
    errs.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, err_chunks)| {
            for j in 0..err_chunks.len() {
                let alpha = (i * chunk_size + j) as f64 * alpha.step_size + alpha.start;
                err_chunks[j].write(compute_distance_between_measured_and_modelled(
                    &measured.proxy(iors),
                    distro,
                    metric,
                    weighting,
                    alpha,
                    alpha,
                    max_theta_i,
                    max_theta_o,
                ));
            }
        });

    // Return the computed distances
    unsafe { errs.assume_init() }
}

/// Brute force fitting for anisotropic microfacet BRDFs.
pub fn brdf_fitting_brute_force_anisotropic<F: AnyMeasuredBrdf>(
    measured: &F,
    distro: MicrofacetDistroKind,
    metric: ErrorMetric,
    max_theta_i: Radians,
    max_theta_o: Radians,
    weighting: Weighting,
    alphax: StepRangeIncl<f64>,
    alphay: StepRangeIncl<f64>,
    iors: &IorReg,
) -> Box<[f64]> {
    log::debug!("compute anisotropic distance");
    let count = alphax.step_count() * alphay.step_count();
    let mut errs = Box::new_uninit_slice(count);
    // Limit the number of threads to 1/2 of the available parallelism to avoid
    // occupying too much resources.
    let num_threads = ((std::thread::available_parallelism().unwrap().get()) / 2).max(1);
    let chunk_size = count / num_threads;
    errs.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, err_chunks)| {
            for j in 0..err_chunks.len() {
                let index = i * chunk_size + j;
                let alpha_x_idx = index / alphay.step_count();
                let alpha_y_idx = index % alphay.step_count();
                let alpha_x = alphax.start + alpha_x_idx as f64 * alphax.step_size;
                let alpha_y = alphay.start + alpha_y_idx as f64 * alphay.step_size;
                err_chunks[j].write(compute_distance_between_measured_and_modelled(
                    &measured.proxy(iors),
                    distro,
                    metric,
                    weighting,
                    alpha_x,
                    alpha_y,
                    max_theta_i,
                    max_theta_o,
                ));
            }
        });

    // Return the computed distances
    unsafe { errs.assume_init() }
}

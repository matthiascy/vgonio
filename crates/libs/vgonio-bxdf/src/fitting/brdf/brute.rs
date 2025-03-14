use crate::brdf::analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cust::module::Module;
use vgonio_core::{
    bxdf::{AnalyticalBrdf, BrdfProxy, MicrofacetDistroKind},
    units::Radians,
    ErrorMetric, Weighting,
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
    gpu_threads_per_block: Option<u32>,
    gpu_modules: Option<&HashMap<&'static str, Module>>,
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
        measured.distance(
            &modelled,
            metric,
            weighting,
            gpu_threads_per_block,
            gpu_modules,
        )
    }
}

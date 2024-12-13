use base::{
    optics::ior::IorRegistry, range::StepRangeIncl, units::Radians, ErrorMetric, Isotropy,
    Weighting,
};

use crate::distro::MicrofacetDistroKind;

use super::AnalyticalFit2;

pub fn brdf_fitting_brute_force<F: AnalyticalFit2>(
    brdf: &F,
    distro: MicrofacetDistroKind,
    metric: ErrorMetric,
    isotropy: Isotropy,
    weighting: Weighting,
    alphax: StepRangeIncl<f64>,
    alphay: StepRangeIncl<f64>,
    iors: &IorRegistry,
    max_theta_i: Radians,
    max_theta_o: Radians,
) {
    todo!("Implement this")
}

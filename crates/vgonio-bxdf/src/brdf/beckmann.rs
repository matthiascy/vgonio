use base::math::{Sph2, Vec3};

use crate::{
    dist::BeckmannDistribution, impl_common_methods, MicrofacetBasedBrdfModel,
    MicrofacetBasedBrdfModelFittingModel, MicrofacetBasedBsdfModelKind,
    MicrofacetDistributionModel,
};

/// Beckmann microfacet BRDF model.
/// See [Beckmann Distribution](crate::dist::BeckmannDistribution).
#[derive(Debug, Clone, Copy)]
pub struct BeckmannBrdfModel {
    /// Roughness parameter of the originated from microfacet distribution
    /// function.
    pub alpha_x: f64,
    /// Roughness parameter of the originated from microfacet distribution
    pub alpha_y: f64,
}

impl BeckmannBrdfModel {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannBrdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

impl MicrofacetBasedBrdfModel for BeckmannBrdfModel {
    fn kind(&self) -> MicrofacetBasedBsdfModelKind { MicrofacetBasedBsdfModelKind::Beckmann }

    impl_common_methods!();

    fn eval(&self, wi: Vec3, wo: Vec3) -> f64 {
        debug_assert!(wi.is_normalized(), "incident direction is not normalized");
        debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
        let wh = (wi + wo).normalize();
        let dist = BeckmannDistribution::new(self.alpha_x, self.alpha_y);
        let d = dist.eval_adf(wh.z as f64, wh.y.atan2(wh.x) as f64);
        let g = dist.eval_msf1(wh, wi) * dist.eval_msf1(wh, wo);
        // TODO: eval fresnel
        let f = 1.0;
        (d * g * f) / (4.0 * wi.z * wo.z)
    }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBrdfModel> { Box::new(*self) }
}

impl MicrofacetBasedBrdfModelFittingModel for BeckmannBrdfModel {
    fn partial_derivatives(&self, wos: &[Vec3], wis: &[Vec3]) -> Box<[f64]> {
        debug_assert!(
            wos.len() == wis.len(),
            "incident and outgoing directions have different length"
        );
        let mut result = Box::new_uninit_slice(wos.len() * 2);
        for i in 0..wos.len() {
            let wo = wos[i];
            let wi = wis[i];
            debug_assert!(wi.is_normalized(), "incident direction is not normalized");
            debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
            let wh = (wi + wo).normalize();
            // TODO: eval fresnel
            let f = 1.0;
        }
    }
}

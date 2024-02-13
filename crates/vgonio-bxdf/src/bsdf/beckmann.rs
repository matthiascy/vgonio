use base::math::{Sph2, Vec3};

use crate::{
    impl_common_methods, MicrofacetBasedBsdfModel, MicrofacetBasedBsdfModelKind,
    MicrofactBasedBsdfModelFittingModel,
};

/// Beckmann microfacet BSDF model.
/// See [Beckmann Distribution](crate::dist::BeckmannDistribution).
#[derive(Debug, Clone, Copy)]
pub struct BeckmannBsdfModel {
    /// Roughness parameter of the originated from microfacet distribution
    /// function.
    pub alpha_x: f64,
    /// Roughness parameter of the originated from microfacet distribution
    pub alpha_y: f64,
}

impl BeckmannBsdfModel {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannBsdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

impl MicrofacetBasedBsdfModel for BeckmannBsdfModel {
    fn kind(&self) -> MicrofacetBasedBsdfModelKind { MicrofacetBasedBsdfModelKind::Beckmann }

    impl_common_methods!();

    fn eval(&self, wi: Sph2, wo: Sph2) -> f64 { todo!() }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBsdfModel> { Box::new(*self) }
}

impl MicrofactBasedBsdfModelFittingModel for BeckmannBsdfModel {
    fn partial_derivative(&self, wo: Vec3, wi: Vec3) -> Vec3 { todo!() }

    fn partial_derivatives(&self, wo: Vec3, wi: Vec3) -> (Vec3, Vec3) { todo!() }
}

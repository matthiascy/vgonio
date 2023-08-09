use crate::fitting::{FittingModel, MicrofacetAreaDistributionModel};

/// Trowbridge-Reitz(GGX) microfacet area distribution function.
///
/// $$ D(\mathbf{m}) = \frac{\alpha^2}{\pi \cos^4 \theta_m (\alpha^2 + \tan^2
/// \theta_m)^2} $$
///
/// where $\alpha$ is the width parameter of the NDF, $\theta_m$ is the angle
/// between the microfacet normal and the normal of the surface.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzIsotropicMadf {
    /// Width parameter of the NDF.
    pub width: f64,
}

impl MicrofacetAreaDistributionModel for TrowbridgeReitzIsotropicMadf {
    fn model(&self) -> FittingModel { FittingModel::TrowbridgeReitz }

    fn is_isotropic(&self) -> bool { true }

    fn params(&self) -> [f64; 2] { [self.width, self.width] }

    fn set_params(&mut self, params: [f64; 2]) { self.width = params[0]; }

    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64 {
        let alpha2 = self.width * self.width;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        alpha2 / (std::f64::consts::PI * cos_theta_m4 * (alpha2 + tan_theta_m2).powi(2))
    }

    fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel> { Box::new(*self) }
}

pub type GGXMadfIsotropic = TrowbridgeReitzIsotropicMadf;

#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzMadfAnisotropic {
    /// Width parameter along the horizontal axis of the NDF.
    pub width_u: f32,
    /// Width parameter along the vertical axis of the NDF.
    pub width_v: f32,
}

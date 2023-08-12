use crate::fitting::{
    FittingModel, MicrofacetAreaDistributionModel, MicrofacetMaskingShadowingModel,
};

/// Trowbridge-Reitz(GGX) microfacet area distribution function.
///
/// $$ D(\mathbf{m}) = \frac{\alpha^2}{\pi \cos^4 \theta_m (\alpha^2 + \tan^2
/// \theta_m)^2} $$
///
/// where $\alpha$ is the width parameter of the NDF, $\theta_m$ is the angle
/// between the microfacet normal and the normal of the surface.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzMadf {
    /// Width parameter of the NDF.
    pub width: f64,
}

impl MicrofacetAreaDistributionModel for TrowbridgeReitzMadf {
    fn name(&self) -> &'static str { "Trowbridge-Reitz ADF" }

    fn fitting_model(&self) -> FittingModel { FittingModel::TrowbridgeReitz }

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

/// Trowbridge-Reitz model is also known as GGX model.
pub type GGXMadfIsotropic = TrowbridgeReitzMadf;

/// Anisotropic Trowbridge-Reitz microfacet area distribution function.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzMadfAnisotropic {
    /// Width parameter along the horizontal axis of the NDF.
    pub width_u: f32,
    /// Width parameter along the vertical axis of the NDF.
    pub width_v: f32,
}

/// Trowbridge-Reitz microfacet masking-shadowing function.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzMmsf {
    /// Width parameter of the MSF.
    pub width: f64,
}

impl MicrofacetMaskingShadowingModel for TrowbridgeReitzMmsf {
    fn name(&self) -> &'static str { "Trowbridge-Reitz MSF" }

    fn fitting_model(&self) -> FittingModel { FittingModel::TrowbridgeReitz }

    fn param(&self) -> f64 { self.width }

    fn set_param(&mut self, param: f64) { self.width = param; }

    fn eval_with_cos_theta_v(&self, cos_theta_v: f64) -> f64 {
        let alpha2 = self.width * self.width;
        let cos_theta_v2 = cos_theta_v * cos_theta_v;
        let tan_theta_v2 = (1.0 - cos_theta_v2) / cos_theta_v2;
        2.0 / (1.0 + (1.0 + alpha2 * tan_theta_v2).sqrt())
    }

    fn clone_box(&self) -> Box<dyn MicrofacetMaskingShadowingModel> { Box::new(*self) }
}

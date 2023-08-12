use crate::fitting::{
    FittingModel, MicrofacetAreaDistributionModel, MicrofacetMaskingShadowingModel,
};

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoMadf {
    /// Roughness parameter of the NDF.
    pub roughness: f64,
}

impl MicrofacetAreaDistributionModel for BeckmannSpizzichinoMadf {
    fn name(&self) -> &'static str { "Beckmann-Spizzichino ADF" }

    fn fitting_model(&self) -> FittingModel { FittingModel::BeckmannSpizzichino }

    fn is_isotropic(&self) -> bool { true }

    fn params(&self) -> [f64; 2] { [self.roughness, self.roughness] }

    fn set_params(&mut self, params: [f64; 2]) { self.roughness = params[0]; }

    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64 {
        let alpha2 = self.roughness * self.roughness;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        (-tan_theta_m2 / alpha2).exp() / (std::f64::consts::PI * alpha2 * cos_theta_m4)
    }

    fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel> { Box::new(*self) }
}

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoMadfAnisotropic {
    /// Roughness parameter along the horizontal axis of the NDF.
    pub roughness_u: f32,
    /// Roughness parameter along the vertical axis of the NDF.
    pub roughness_v: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoMmsf {
    /// Roughness parameter of the MSF.
    pub roughness: f64,
}

impl MicrofacetMaskingShadowingModel for BeckmannSpizzichinoMmsf {
    fn name(&self) -> &'static str { "Beckmann-Spizzichino MSF" }

    fn fitting_model(&self) -> FittingModel { FittingModel::BeckmannSpizzichino }

    fn param(&self) -> f64 { self.roughness }

    fn set_param(&mut self, param: f64) { self.roughness = param; }

    fn eval_with_cos_theta_v(&self, cos_theta_v: f64) -> f64 {
        let alpha = self.roughness;
        let tan_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt() / cos_theta_v;
        let cot_theta_v = 1.0 / tan_theta_v;
        let a = cot_theta_v / alpha;
        let denominator =
            1.0 + libm::erf(a) + (-a * a).exp() * alpha * tan_theta_v / std::f64::consts::PI.sqrt();
        2.0 / denominator
    }

    fn clone_box(&self) -> Box<dyn MicrofacetMaskingShadowingModel> { Box::new(*self) }
}

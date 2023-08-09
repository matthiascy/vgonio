use crate::fitting::{FittingModel, MicrofacetAreaDistributionModel};

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoIsotropicMadf {
    /// Roughness parameter of the NDF.
    pub roughness: f64,
}

impl MicrofacetAreaDistributionModel for BeckmannSpizzichinoIsotropicMadf {
    fn model(&self) -> FittingModel { FittingModel::BeckmannSpizzichino }

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

use crate::fitting::{
    impl_get_set_scale, impl_microfacet_area_distribution_model_for_anisotropic_model,
    impl_microfacet_area_distribution_model_for_isotropic_model,
    AnisotropicMicrofacetAreaDistributionModel, IsotropicMicrofacetAreaDistributionModel,
    MicrofacetAreaDistributionModel, MicrofacetMaskingShadowingModel, MicrofacetModelFamily,
    ReflectionModelFamily,
};
use chrono::format::Item;
use vgcore::math::{rcp_f32, rcp_f64};

/// Beckmann-Spizzichino microfacet area distribution function.
///
/// Based on the Gaussian distribution of microfacet slopes. If σ is the
/// RMS slope of the microfacets, then the alpha parameter of the Beckmann
/// distribution is given by: $\alpha = \sqrt{2} \sigma$.
#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoNDF {
    /// Roughness parameter of the NDF.
    pub alpha: f64,
    #[cfg(feature = "scaled-ndf-fitting")]
    /// The scale factor of the NDF.
    pub scale: Option<f64>,
}

impl BeckmannSpizzichinoNDF {
    pub fn default() -> Self {
        BeckmannSpizzichinoNDF {
            alpha: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: None,
        }
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    pub fn default_with_scale() -> Self {
        BeckmannSpizzichinoNDF {
            alpha: 0.1,
            scale: Some(1.0),
        }
    }
}

impl IsotropicMicrofacetAreaDistributionModel for BeckmannSpizzichinoNDF {
    fn name(&self) -> &'static str {
        #[cfg(feature = "scaled-ndf-fitting")]
        if self.scale.is_none() {
            "Beckmann-Spizzichino NDF"
        } else {
            "Scaled Beckmann-Spizzichino NDF"
        }
        #[cfg(not(feature = "scaled-ndf-fitting"))]
        "Beckmann-Spizzichino NDF"
    }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    }

    fn param(&self) -> f64 { self.alpha }

    fn set_param(&mut self, param: f64) { self.alpha = param; }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn scale(&self) -> Option<f64> { self.scale }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn set_scale(&mut self, scale: f64) {
        #[cfg(debug_assertions)]
        if self.scale.is_none() {
            panic!("Trying to set the scale on a non-scaled Beckmann Spizzichino NDF");
        }
        self.scale.replace(scale);
    }

    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64 {
        let alpha2 = self.alpha * self.alpha;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        (-tan_theta_m2 / alpha2).exp() / (std::f64::consts::PI * alpha2 * cos_theta_m4)
    }

    fn calc_param_pd(&self, cos_theta_ms: &[f64]) -> Vec<f64> {
        let alpha = self.alpha;
        let alpha2 = alpha * alpha;
        let denominator = std::f64::consts::PI * alpha.powi(5);
        cos_theta_ms
            .iter()
            .map(|cos_theta_m| {
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let sec_theta_m2 = 1.0 / cos_theta_m2;
                let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / cos_theta_m2 - cos_theta_m2
                };
                let numerator = 2.0
                    * (-tan_theta_m2 / alpha2).exp()
                    * (tan_theta_m2 - alpha2)
                    * sec_theta_m2
                    * sec_theta_m2;
                numerator / denominator
            })
            .collect()
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn calc_param_pd_scaled(&self, cos_theta_ms: &[f64]) -> Vec<f64> {
        let alpha = self.alpha;
        let alpha2 = alpha * alpha;
        let scale = self.scale.expect("Model is not scalable");
        let d_alpha_denominator = std::f64::consts::PI * alpha.powi(5);
        let d_scale_denominator = std::f64::consts::PI * alpha2;
        cos_theta_ms
            .iter()
            .flat_map(|cos_theta_m| {
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let sec_theta_m2 = 1.0 / cos_theta_m2;
                let sec_theta_m4 = sec_theta_m2 * sec_theta_m2;
                let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / cos_theta_m2 - cos_theta_m2
                };
                let d_alpha = {
                    let numerator = 2.0
                        * (-tan_theta_m2 / alpha2).exp()
                        * (tan_theta_m2 - alpha2)
                        * sec_theta_m4
                        * scale;

                    numerator / d_alpha_denominator
                };

                let d_scale = {
                    let numerator = (-tan_theta_m2 / alpha2).exp() * sec_theta_m4;
                    numerator / d_scale_denominator
                };
                [d_alpha, d_scale]
            })
            .collect()
    }

    fn clone_box(&self) -> Box<dyn IsotropicMicrofacetAreaDistributionModel> { todo!() }
}

impl_microfacet_area_distribution_model_for_isotropic_model!(BeckmannSpizzichinoNDF);

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoAnisotropicNDF {
    /// Roughness parameter along the horizontal axis of the NDF.
    pub alpha_x: f64,
    /// Roughness parameter along the vertical axis of the NDF.
    pub alpha_y: f64,
    #[cfg(feature = "scaled-ndf-fitting")]
    /// The scale factor of the NDF.
    pub scale: Option<f64>,
}

impl BeckmannSpizzichinoAnisotropicNDF {
    pub fn default() -> Self {
        BeckmannSpizzichinoAnisotropicNDF {
            alpha_x: 0.1,
            alpha_y: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: None,
        }
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    pub fn default_with_scale() -> Self {
        BeckmannSpizzichinoAnisotropicNDF {
            alpha_x: 0.1,
            alpha_y: 0.1,
            scale: Some(1.0),
        }
    }
}

// TODO: Verify the correctness of the implementation
impl AnisotropicMicrofacetAreaDistributionModel for BeckmannSpizzichinoAnisotropicNDF {
    fn name(&self) -> &'static str {
        #[cfg(feature = "scaled-ndf-fitting")]
        if self.scale.is_none() {
            "Beckmann-Spizzichino NDF (anisotropic)"
        } else {
            "Scaled Beckmann-Spizzichino NDF (anisotropic)"
        }
        #[cfg(not(feature = "scaled-ndf-fitting"))]
        "Beckmann-Spizzichino NDF (anisotropic)"
    }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    }

    fn params(&self) -> [f64; 2] { [self.alpha_x, self.alpha_y] }

    fn set_params(&mut self, params: [f64; 2]) {
        self.alpha_x = params[0];
        self.alpha_y = params[1];
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    impl_get_set_scale!(self);

    /// Evaluates the NDF for the given
    fn eval_with_cos_theta_phi_m(&self, cos_theta_m: f64, cos_phi_m: f64) -> f64 {
        let alpha_x2 = self.alpha_x * self.alpha_x;
        let alpha_y2 = self.alpha_y * self.alpha_y;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) * rcp_f64(cos_theta_m2);
        let sec_theta_m4 = rcp_f64(cos_theta_m2 * cos_theta_m2);
        let cos_phi_m2 = cos_phi_m * cos_phi_m;
        let sin_phi_m2 = (1.0 - cos_phi_m2).max(0.0);
        let denominator =
            std::f64::consts::PI * self.alpha_x * self.alpha_y * cos_theta_m2 * cos_theta_m2;
        let numerator = (-tan_theta_m2 * (cos_phi_m2 / alpha_x2 + sin_phi_m2 / alpha_y2)).exp();
        numerator * sec_theta_m4 / denominator
    }

    fn calc_params_pd(&self, cos_theta_phi_ms: &[(f64, f64)]) -> Vec<f64> {
        let alpha_x2 = self.alpha_x * self.alpha_x;
        let alpha_x4 = alpha_x2 * alpha_x2;
        let alpha_y2 = self.alpha_y * self.alpha_y;
        let alpha_y4 = alpha_y2 * alpha_y2;
        cos_theta_phi_ms
            .iter()
            .flat_map(|(cos_theta_m, cos_phi_m)| {
                let cos_phi_m2 = cos_phi_m * cos_phi_m;
                let sin_phi_m2 = (1.0 - cos_phi_m2).max(0.0);
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
                let tan_theta_m2 = (1.0 - cos_theta_m2) * rcp_f64(cos_theta_m2);
                let sec_theta_m4 = rcp_f64(cos_theta_m4);
                let exp = (-tan_theta_m2 * (cos_phi_m2 / alpha_x2 + sin_phi_m2 / alpha_y2)).exp();
                let d_alpha_x = {
                    let numerator =
                        exp * sec_theta_m4 * (2.0 * cos_phi_m2 * tan_theta_m2 - alpha_x2);
                    numerator / (std::f64::consts::PI * alpha_x4 * alpha_y2)
                };
                let d_alpha_y = {
                    let numerator =
                        exp * sec_theta_m4 * (2.0 * sin_phi_m2 * tan_theta_m2 - alpha_y2);
                    numerator / (std::f64::consts::PI * alpha_x2 * alpha_y4)
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect()
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn calc_params_pd_scaled(&self, cos_theta_phi_ms: &[(f64, f64)]) -> Vec<f64> {
        let scale = self.scale.expect("Model is not scalable");
        let alpha_x = self.alpha_x;
        let alpha_y = self.alpha_y;
        let alpha_x2 = alpha_x * alpha_x;
        let alpha_x4 = alpha_x2 * alpha_x2;
        let alpha_y2 = alpha_y * alpha_y;
        let alpha_y4 = alpha_y2 * alpha_y2;
        cos_theta_phi_ms
            .iter()
            .flat_map(|(cos_theta_m, cos_phi_m)| {
                let cos_phi_m2 = cos_phi_m * cos_phi_m;
                let sin_phi_m2 = (1.0 - cos_phi_m2).max(0.0);
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
                let tan_theta_m2 = (1.0 - cos_theta_m2) * rcp_f64(cos_theta_m2);
                let sec_theta_m4 = rcp_f64(cos_theta_m4);
                let exp = (-tan_theta_m2 * (cos_phi_m2 / alpha_x2 + sin_phi_m2 / alpha_y2)).exp();
                let d_alpha_x = {
                    let numerator =
                        scale * exp * sec_theta_m4 * (2.0 * cos_phi_m2 * tan_theta_m2 - alpha_x2);
                    numerator / (std::f64::consts::PI * alpha_x4 * alpha_y2)
                };
                let d_alpha_y = {
                    let numerator =
                        scale * exp * sec_theta_m4 * (2.0 * sin_phi_m2 * tan_theta_m2 - alpha_y2);
                    numerator / (std::f64::consts::PI * alpha_x2 * alpha_y4)
                };
                let d_scale = {
                    let numerator = exp * sec_theta_m4;
                    numerator / (std::f64::consts::PI * alpha_x * alpha_y)
                };
                [d_alpha_x, d_alpha_y, d_scale]
            })
            .collect()
    }

    fn clone_box(&self) -> Box<dyn AnisotropicMicrofacetAreaDistributionModel> { Box::new(*self) }
}

impl_microfacet_area_distribution_model_for_anisotropic_model!(BeckmannSpizzichinoAnisotropicNDF);

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoMSF {
    /// Roughness parameter of the MSF.
    pub alpha: f64,
}

impl MicrofacetMaskingShadowingModel for BeckmannSpizzichinoMSF {
    fn name(&self) -> &'static str { "Beckmann-Spizzichino MSF" }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    }

    fn param(&self) -> f64 { self.alpha }

    fn set_param(&mut self, param: f64) { self.alpha = param; }

    fn eval_with_cos_theta_v(&self, cos_theta_v: f64) -> f64 {
        let alpha = self.alpha;
        let tan_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt() / cos_theta_v;
        let cot_theta_v = 1.0 / tan_theta_v;
        let a = cot_theta_v / alpha;
        let denominator =
            1.0 + libm::erf(a) + (-a * a).exp() * alpha * tan_theta_v / std::f64::consts::PI.sqrt();
        2.0 / denominator
    }

    fn clone_box(&self) -> Box<dyn MicrofacetMaskingShadowingModel> { Box::new(*self) }
}

#[test]
fn beckmann_spizzichino_family() {
    let madf = BeckmannSpizzichinoNDF::default();
    assert_eq!(
        madf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    );
    let mmsf = BeckmannSpizzichinoMSF { alpha: 0.1 };
    assert_eq!(
        mmsf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    );
}

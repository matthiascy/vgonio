#[cfg(feature = "scaled-ndf-fitting")]
use crate::fitting::impl_get_set_scale;
use crate::fitting::{
    impl_microfacet_area_distribution_model_for_anisotropic_model,
    impl_microfacet_area_distribution_model_for_isotropic_model,
    AnisotropicMicrofacetAreaDistributionModel, IsotropicMicrofacetAreaDistributionModel,
    MicrofacetAreaDistributionModel, MicrofacetMaskingShadowingModel, MicrofacetModelFamily,
    ReflectionModelFamily,
};
use vgcore::math::rcp_f64;

/// Trowbridge-Reitz(GGX) microfacet area distribution function.
///
/// $$ D(\mathbf{m}) = \frac{\alpha^2}{\pi \cos^4 \theta_m (\alpha^2 + \tan^2
/// \theta_m)^2} $$
///
/// where $\alpha$ is the width parameter of the NDF, $\theta_m$ is the angle
/// between the microfacet normal and the normal of the surface.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzNDF {
    /// Width parameter of the NDF.
    pub alpha: f64,
    #[cfg(feature = "scaled-ndf-fitting")]
    /// The scale factor of the NDF.
    pub scale: Option<f64>,
}

impl TrowbridgeReitzNDF {
    pub fn new(alpha: f64, #[cfg(feature = "scaled-ndf-fitting")] scale: Option<f64>) -> Self {
        TrowbridgeReitzNDF {
            alpha,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale,
        }
    }
    pub fn default() -> Self {
        TrowbridgeReitzNDF {
            alpha: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: None,
        }
    }

    pub fn default_with_scale() -> Self {
        TrowbridgeReitzNDF {
            alpha: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: Some(1.0),
        }
    }
}

impl IsotropicMicrofacetAreaDistributionModel for TrowbridgeReitzNDF {
    fn name(&self) -> &'static str {
        #[cfg(feature = "scaled-ndf-fitting")]
        if self.scale.is_none() {
            "Trowbridge-Reitz NDF"
        } else {
            "Scaled Trowbridge-Reitz NDF"
        }
        #[cfg(not(feature = "scaled-ndf-fitting"))]
        "Trowbridge-Reitz NDF"
    }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    }

    fn param(&self) -> f64 { self.alpha }

    fn set_param(&mut self, param: f64) { self.alpha = param; }

    #[cfg(feature = "scaled-ndf-fitting")]
    impl_get_set_scale!(self);

    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64 {
        let alpha2 = self.alpha * self.alpha;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        alpha2 / (std::f64::consts::PI * cos_theta_m4 * (alpha2 + tan_theta_m2).powi(2))
    }

    fn calc_param_pd(&self, cos_theta_ms: &[f64]) -> Vec<f64> {
        let alpha = self.alpha;
        let alpha2 = alpha * alpha;
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
                let numerator = 2.0 * alpha * (tan_theta_m2 - alpha2) * sec_theta_m2 * sec_theta_m2;
                let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);

                numerator / denominator
            })
            .collect()
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn calc_param_pd_scaled(&self, cos_theta_ms: &[f64]) -> Vec<f64> {
        let alpha = self.alpha;
        let alpha2 = alpha * alpha;
        let scale = self.scale.expect("Model is not scalable");
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

                // Derivative of the Trowbridge-Reitz distribution with respect to alpha
                let d_alpha = {
                    let numerator = 2.0 * alpha * scale * (tan_theta_m2 - alpha2) * sec_theta_m4;
                    let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);
                    numerator / denominator
                };

                let d_scale = {
                    let numerator = alpha2 * sec_theta_m4;
                    let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(2);
                    numerator / denominator
                };
                [d_alpha, d_scale]
            })
            .collect()
    }

    fn clone_box(&self) -> Box<dyn IsotropicMicrofacetAreaDistributionModel> { Box::new(*self) }
}

/// Trowbridge-Reitz model is also known as GGX model.
pub type GGXIsotropicNDF = TrowbridgeReitzNDF;

impl_microfacet_area_distribution_model_for_isotropic_model!(TrowbridgeReitzNDF);

/// Anisotropic Trowbridge-Reitz microfacet area distribution function.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzAnisotropicNDF {
    /// Width parameter along the horizontal axis of the NDF.
    pub alpha_x: f64,
    /// Width parameter along the vertical axis of the NDF.
    pub alpha_y: f64,
    #[cfg(feature = "scaled-ndf-fitting")]
    /// The scale factor of the NDF.
    pub scale: Option<f64>,
}

impl TrowbridgeReitzAnisotropicNDF {
    pub fn new(
        alpha_x: f64,
        alpha_y: f64,
        #[cfg(feature = "scaled-ndf-fitting")] scale: Option<f64>,
    ) -> Self {
        TrowbridgeReitzAnisotropicNDF {
            alpha_x,
            alpha_y,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale,
        }
    }

    pub fn default() -> Self {
        TrowbridgeReitzAnisotropicNDF {
            alpha_x: 0.5,
            alpha_y: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: None,
        }
    }

    pub fn default_with_scale() -> Self {
        TrowbridgeReitzAnisotropicNDF {
            alpha_x: 0.5,
            alpha_y: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: Some(1.0),
        }
    }
}

impl AnisotropicMicrofacetAreaDistributionModel for TrowbridgeReitzAnisotropicNDF {
    fn name(&self) -> &'static str {
        #[cfg(feature = "scaled-ndf-fitting")]
        if self.scale.is_none() {
            "Trowbridge-Reitz NDF (anisotropic)"
        } else {
            "Scaled Trowbridge-Reitz NDF (anisotropic)"
        }
        #[cfg(not(feature = "scaled-ndf-fitting"))]
        "Trowbridge-Reitz NDF (anisotropic)"
    }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    }

    fn params(&self) -> [f64; 2] { [self.alpha_x, self.alpha_y] }

    fn set_params(&mut self, params: [f64; 2]) {
        self.alpha_x = params[0];
        self.alpha_y = params[1];
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    impl_get_set_scale!(self);

    fn eval_with_cos_theta_phi_m(&self, cos_theta_m: f64, cos_phi_m: f64) -> f64 {
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        let sec_theta_m4 = rcp_f64(cos_theta_m4);
        let cos_phi_m2 = cos_phi_m * cos_phi_m;
        let sin_phi_m2 = 1.0 - cos_phi_m2;
        let alpha_x = self.alpha_x;
        let alpha_y = self.alpha_y;
        let alpha_x2 = self.alpha_x * self.alpha_x;
        let alpha_y2 = self.alpha_y * self.alpha_y;
        let rcp_alpha_x2 = rcp_f64(alpha_x2);
        let rcp_alpha_y2 = rcp_f64(alpha_y2);
        let denominator = std::f64::consts::PI
            * alpha_x
            * alpha_y
            * (1.0 + tan_theta_m2 * (cos_phi_m2 * rcp_alpha_x2 + sin_phi_m2 * rcp_alpha_y2))
                .powi(2);
        sec_theta_m4 / denominator
    }

    fn calc_params_pd(&self, cos_theta_phi_ms: &[(f64, f64)]) -> Vec<f64> {
        let alpha_x = self.alpha_x;
        let alpha_y = self.alpha_y;
        let alpha_x2 = alpha_x * alpha_x;
        let alpha_y2 = alpha_y * alpha_y;
        let alpha_x3 = alpha_x2 * alpha_x;
        let alpha_y3 = alpha_y2 * alpha_y;
        cos_theta_phi_ms
            .iter()
            .flat_map(|(cos_theta_m, cos_phi_m)| {
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let tan_theta_m2 = (1.0 - cos_theta_m2) * rcp_f64(cos_theta_m2);
                let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
                let sec_theta_m4 = rcp_f64(cos_theta_m4);
                let cos_phi_m2 = cos_phi_m * cos_phi_m;
                let sin_phi_m2 = 1.0 - cos_phi_m2;

                let alpha_denominator = std::f64::consts::PI
                    * (alpha_y2 * cos_phi_m2 * tan_theta_m2
                        + alpha_x2 * (alpha_y2 + sin_phi_m2 * tan_theta_m2))
                        .powi(3);

                // Derivative of the Trowbridge-Reitz distribution with respect to alpha_x
                let d_alpha_x = {
                    let numerator = alpha_x2
                        * alpha_y3
                        * sec_theta_m4
                        * (3.0 * alpha_y2 * cos_phi_m2 * tan_theta_m2
                            - alpha_x2 * (alpha_y2 + sin_phi_m2 * tan_theta_m2));
                    numerator / alpha_denominator
                };

                // Derivative of the Trowbridge-Reitz distribution with respect to alpha_y
                let d_alpha_y = {
                    let numerator = -alpha_x3
                        * alpha_y2
                        * sec_theta_m4
                        * (alpha_y2 * cos_phi_m2 * tan_theta_m2
                            + alpha_x2 * (alpha_y2 - 3.0 * sin_phi_m2 * tan_theta_m2));
                    numerator / alpha_denominator
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
        let alpha_y2 = alpha_y * alpha_y;
        let alpha_x3 = alpha_x2 * alpha_x;
        let alpha_y3 = alpha_y2 * alpha_y;

        cos_theta_phi_ms
            .iter()
            .flat_map(|(cos_theta_m, cos_phi_m)| {
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let tan_theta_m2 = (1.0 - cos_theta_m2) * rcp_f64(cos_theta_m2);
                let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
                let sec_theta_m4 = rcp_f64(cos_theta_m4);
                let cos_phi_m2 = cos_phi_m * cos_phi_m;
                let sin_phi_m2 = 1.0 - cos_phi_m2;

                let alpha_denominator = std::f64::consts::PI
                    * (alpha_y2 * cos_phi_m2 * tan_theta_m2
                        + alpha_x2 * (alpha_y2 + sin_phi_m2 * tan_theta_m2))
                        .powi(3);

                // Derivative of the Trowbridge-Reitz distribution with respect to alpha_x
                let d_alpha_x = {
                    let numerator = scale
                        * alpha_x2
                        * alpha_y3
                        * sec_theta_m4
                        * (3.0 * alpha_y2 * cos_phi_m2 * tan_theta_m2
                            - alpha_x2 * (alpha_y2 + sin_phi_m2 * tan_theta_m2));
                    numerator / alpha_denominator
                };

                // Derivative of the Trowbridge-Reitz distribution with respect to alpha_y
                let d_alpha_y = {
                    let numerator = -scale
                        * alpha_x3
                        * alpha_y2
                        * sec_theta_m4
                        * (alpha_y2 * cos_phi_m2 * tan_theta_m2
                            + alpha_x2 * (alpha_y2 - 3.0 * sin_phi_m2 * tan_theta_m2));
                    numerator / alpha_denominator
                };

                let d_scale = {
                    let denominator = std::f64::consts::PI
                        * alpha_x
                        * alpha_y
                        * (1.0 + tan_theta_m2 * (cos_phi_m2 / alpha_x2 + sin_phi_m2 / alpha_y2))
                            .powi(2);

                    sec_theta_m4 / denominator
                };

                [d_alpha_x, d_alpha_y, d_scale]
            })
            .collect()
    }

    fn clone_box(&self) -> Box<dyn AnisotropicMicrofacetAreaDistributionModel> { Box::new(*self) }
}

impl_microfacet_area_distribution_model_for_anisotropic_model!(TrowbridgeReitzAnisotropicNDF);

/// Trowbridge-Reitz microfacet masking-shadowing function.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzMSF {
    /// Width parameter of the MSF.
    pub width: f64,
}

impl MicrofacetMaskingShadowingModel for TrowbridgeReitzMSF {
    fn name(&self) -> &'static str { "Trowbridge-Reitz MSF" }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    }

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

#[test]
fn trowbridge_reitz_family() {
    let mndf = TrowbridgeReitzNDF::default();
    assert_eq!(
        mndf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    );
    let mmsf = TrowbridgeReitzMSF { width: 0.1 };
    assert_eq!(
        mmsf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    );
}

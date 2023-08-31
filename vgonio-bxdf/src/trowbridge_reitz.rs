use crate::{
    impl_microfacet_distribution_common_methods, MicrofacetDistributionFittingModel,
    MicrofacetDistributionModel, MicrofacetDistributionModelKind,
};
use std::fmt::Debug;
use vgcore::{
    math::{cube, rcp_f64, sqr},
    Isotropy,
};

/// Trowbridge-Reitz(GGX) microfacet distribution function.
///
/// $$ D(\mathbf{m}) = \frac{\alpha^2}{\pi \cos^4 \theta_m (\alpha^2 + \tan^2
/// \theta_m)^2} $$
///
/// where $\alpha$ is the width parameter of the NDF, $\theta_m$ is the angle
/// between the microfacet normal and the normal of the surface.
///
/// In case of anisotropic distribution, the NDF is defined as
///
/// $$ D(\mathbf{m}) = \frac{\alpha_x \alpha_y}{\pi \cos^4 \theta_m (\alpha_x^2
/// \cos^2 \phi_m + \alpha_y^2 \sin^2 \phi_m) (\alpha^2 + \tan^2 \theta_m)^2} $$
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzDistribution {
    /// Parameter of the microfacet area distribution function along the
    /// horizontal axis.
    pub alpha_x: f64,
    /// Parameter of the microfacet area distribution function along the
    /// vertical axis.
    pub alpha_y: f64,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        TrowbridgeReitzDistribution {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }

    pub fn new_isotropic(alpha: f64) -> Self {
        TrowbridgeReitzDistribution {
            alpha_x: alpha.max(1.0e-6),
            alpha_y: alpha.max(1.0e-6),
        }
    }
}

impl MicrofacetDistributionModel for TrowbridgeReitzDistribution {
    fn kind(&self) -> MicrofacetDistributionModelKind {
        MicrofacetDistributionModelKind::TrowbridgeReitz
    }

    impl_microfacet_distribution_common_methods!();

    fn eval_adf(&self, cos_theta: f64, cos_phi: f64) -> f64 {
        let cos_theta2 = sqr(cos_theta);
        let tan_theta2 = (1.0 - cos_theta2) / cos_theta2;
        if tan_theta2.is_infinite() {
            return 0.0;
        }
        let cos_theta4 = sqr(cos_theta2);
        if cos_theta4 < 1.0e-16 {
            return 0.0;
        }
        let alpha_xy = self.alpha_x * self.alpha_y;
        let cos_phi2 = sqr(cos_phi);
        let sin_phi2 = 1.0 - cos_phi2;
        let e = tan_theta2 * (cos_phi2 / sqr(self.alpha_x) + sin_phi2 / sqr(self.alpha_y));
        rcp_f64(alpha_xy * std::f64::consts::PI * cos_theta4 * sqr(1.0 + e))
    }

    fn eval_msf(&self, cos_theta_i: f64, cos_theta_o: f64, cos_theta_h: f64) -> f64 { todo!() }

    fn clone_box(&self) -> Box<dyn MicrofacetDistributionModel> { Box::new(*self) }
}

impl MicrofacetDistributionFittingModel for TrowbridgeReitzDistribution {
    fn adf_partial_derivatives(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Vec<f64> {
        debug_assert!(
            cos_thetas.len() == cos_phis.len(),
            "The number of cos_thetas and cos_phis must be the same."
        );
        let alpha_x2 = sqr(self.alpha_x);
        let alpha_y2 = sqr(self.alpha_y);
        cos_thetas
            .iter()
            .zip(cos_phis.iter())
            .flat_map(|(cos_theta, cos_phi)| {
                let cos_theta2 = sqr(*cos_theta);
                let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
                let sec_theta4 = rcp_f64(sqr(cos_theta2));
                let cos_phi2 = sqr(*cos_phi);
                let sin_phi2 = 1.0 - cos_phi2;
                let rcp_alpha_denom = rcp_f64(
                    std::f64::consts::PI
                        * cube(
                            alpha_y2 * cos_phi2 * tan_theta2
                                + alpha_x2 * (alpha_y2 + sin_phi2 * tan_theta2),
                        ),
                );
                let d_alpha_x = {
                    let numerator = alpha_x2
                        * cube(self.alpha_y)
                        * sec_theta4
                        * (3.0 * alpha_y2 * cos_phi2 * tan_theta2
                            - alpha_x2 * (alpha_y2 + sin_phi2 * tan_theta2));
                    numerator * rcp_alpha_denom
                };
                let d_alpha_y = {
                    let numerator = -cube(self.alpha_x)
                        * alpha_y2
                        * sec_theta4
                        * (alpha_y2 * cos_phi2 * tan_theta2
                            + alpha_x2 * (alpha_y2 - 3.0 * sin_phi2 * tan_theta2));
                    numerator * rcp_alpha_denom
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect()
    }

    fn msf_partial_derivatives(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Vec<f64> { todo!() }
}
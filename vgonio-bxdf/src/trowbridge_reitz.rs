use crate::{
    impl_microfacet_distribution_common_methods, MicrofacetDistributionFittingModel,
    MicrofacetDistributionModel, MicrofacetDistributionModelKind,
};
use std::fmt::Debug;
use vgcore::math::{cube, rcp_f64, sqr, Vec3};

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

    fn eval_msf1(&self, m: Vec3, v: Vec3) -> f64 {
        if m.dot(v) <= 0.0 {
            return 0.0;
        } else {
            let cos_theta_v2 = sqr(v.y as f64);
            let tan_theta_v2 = (1.0 - cos_theta_v2) * rcp_f64(cos_theta_v2);
            2.0 * rcp_f64(1.0 + (1.0 + tan_theta_v2 * self.alpha_x * self.alpha_y).sqrt())
        }
    }

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
        let alpha_x2_alpha_y2 = alpha_x2 * alpha_y2;
        let alpha_x2_alpha_y3 = alpha_x2 * cube(self.alpha_y);
        let alpha_x3_alpha_y2 = cube(self.alpha_x) * alpha_y2;
        cos_thetas
            .iter()
            .zip(cos_phis.iter())
            .flat_map(|(cos_theta, cos_phi)| {
                if cos_theta.abs() < 1.0e-6 {
                    // avoid 90 degree
                    return [0.0, 0.0];
                }
                let cos_theta2 = sqr(*cos_theta);
                let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
                let sec_theta4 = rcp_f64(sqr(cos_theta2));
                let cos_phi2 = sqr(*cos_phi);
                let sin_phi2 = 1.0 - cos_phi2;
                let rcp_alpha_denom = rcp_f64(
                    std::f64::consts::PI
                        * cube(
                            alpha_x2_alpha_y2
                                + tan_theta2 * (alpha_x2 * sin_phi2 + alpha_y2 * cos_phi2),
                        ),
                );
                let d_alpha_x = {
                    let numerator = -alpha_x2_alpha_y3
                        * sec_theta4
                        * (alpha_x2_alpha_y2
                            + tan_theta2 * (alpha_x2 * sin_phi2 - 3.0 * alpha_y2 * cos_phi2));
                    numerator * rcp_alpha_denom
                };
                let d_alpha_y = {
                    let numerator = -alpha_x3_alpha_y2
                        * sec_theta4
                        * (alpha_x2_alpha_y2
                            + tan_theta2 * (alpha_y2 * cos_phi2 - 3.0 * alpha_x2 * sin_phi2));
                    numerator * rcp_alpha_denom
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect()
    }

    fn msf_partial_derivative(&self, m: Vec3, i: Vec3, o: Vec3) -> f64 {
        let cos_theta_i2 = sqr(i.y as f64);
        let tan_theta_i2 = (1.0 - cos_theta_i2) * rcp_f64(cos_theta_i2);
        let cos_theta_o2 = sqr(o.y as f64);
        let tan_theta_o2 = (1.0 - cos_theta_o2) * rcp_f64(cos_theta_o2);
        let denominator = sqr(1.0 + (1.0 + tan_theta_i2 * sqr(self.alpha_x)).sqrt())
            * sqr(1.0 + (1.0 + tan_theta_o2 * sqr(self.alpha_x)).sqrt());
        let numerator = -4.0
            * self.alpha_x
            * ((1.0 + (1.0 + tan_theta_i2 * sqr(self.alpha_x)).sqrt()) * tan_theta_o2
                / sqr(1.0 + (1.0 + sqr(self.alpha_x) * tan_theta_o2).sqrt())
                + (1.0 + (1.0 + tan_theta_o2 * sqr(self.alpha_x)).sqrt()) * tan_theta_i2
                    / sqr(1.0 + (1.0 + sqr(self.alpha_x) * tan_theta_i2).sqrt()));
        numerator * rcp_f64(denominator)
    }
}

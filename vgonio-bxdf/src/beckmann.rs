use crate::{
    impl_microfacet_distribution_common_methods, MicrofacetDistributionFittingModel,
    MicrofacetDistributionModel, MicrofacetDistributionModelKind,
};
use vgcore::{
    math::{cube, rcp_f64, sqr, Vec3},
    Isotropy,
};

/// Beckmann microfacet distribution function.
///
/// Beckmann distribution is based on the Gaussian distribution of microfacet
/// slopes. If σ is the RMS slope of the microfacets, then the alpha
/// parameter of the Beckmann distribution is given by: $\alpha = \sqrt{2}
/// \sigma$.
#[derive(Debug, Copy, Clone)]
pub struct BeckmannDistribution {
    /// Roughness parameter of the MDF (αx = √2σx).
    pub alpha_x: f64,
    /// Roughness parameter of the MDF (αy = √2σy).
    pub alpha_y: f64,
}

impl BeckmannDistribution {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannDistribution {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }

    pub fn new_isotropic(alpha: f64) -> Self {
        BeckmannDistribution {
            alpha_x: alpha.max(1.0e-6),
            alpha_y: alpha.max(1.0e-6),
        }
    }
}

impl MicrofacetDistributionModel for BeckmannDistribution {
    fn kind(&self) -> MicrofacetDistributionModelKind { MicrofacetDistributionModelKind::Beckmann }

    impl_microfacet_distribution_common_methods!();

    fn eval_adf(&self, cos_theta: f64, cos_phi: f64) -> f64 {
        let cos_theta2 = sqr(cos_theta);
        let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
        let cos_theta4 = sqr(cos_theta2);
        if cos_theta4 < 1.0e-16 {
            return 0.0;
        }
        let cos_phi2 = sqr(cos_phi);
        let sin_phi2 = 1.0 - cos_phi2;
        let e = tan_theta2
            * (cos_phi2 * rcp_f64(sqr(self.alpha_x)) + sin_phi2 * rcp_f64(sqr(self.alpha_y)));
        (-e).exp() * rcp_f64(std::f64::consts::PI * cos_theta4 * self.alpha_x * self.alpha_y)
    }

    fn eval_msf1(&self, m: Vec3, v: Vec3) -> f64 {
        if m.dot(v) <= 0.0 {
            0.0
        } else {
            let cos_theta_v2 = sqr(v.y as f64);
            let tan_theta_v2 = (1.0 - cos_theta_v2) * rcp_f64(cos_theta_v2);
            let tan_theta_v = tan_theta_v2.sqrt();
            let a = rcp_f64(self.alpha_x * tan_theta_v);
            2.0 * rcp_f64(
                1.0 + libm::erf(a) + (-sqr(a)).exp() * rcp_f64(std::f64::consts::PI.sqrt() * a),
            )
        }
    }

    fn clone_box(&self) -> Box<dyn MicrofacetDistributionModel> { Box::new(*self) }
}

impl MicrofacetDistributionFittingModel for BeckmannDistribution {
    fn adf_partial_derivatives(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Vec<f64> {
        debug_assert!(
            cos_thetas.len() == cos_phis.len(),
            "The number of cosines of the zenith angle and the number of cosines of the azimuth \
             angle must be the same."
        );
        let alpha_x2 = sqr(self.alpha_x);
        let alpha_y2 = sqr(self.alpha_y);
        let rcp_alpha_x2 = rcp_f64(alpha_x2);
        let rcp_alpha_y2 = rcp_f64(alpha_y2);
        let d_alpha_x_rcp_denom = rcp_f64(std::f64::consts::PI * sqr(alpha_x2) * self.alpha_y);
        let d_alpha_y_rcp_denom = rcp_f64(std::f64::consts::PI * self.alpha_x * sqr(alpha_y2));
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
                let exp = (-tan_theta2 * (cos_phi2 * rcp_alpha_x2 + sin_phi2 * rcp_alpha_y2)).exp();
                let d_alpha_x = {
                    let numerator = exp * sec_theta4 * (2.0 * cos_phi2 * tan_theta2 - alpha_x2);
                    numerator * d_alpha_x_rcp_denom
                };
                let d_alpha_y = {
                    let numerator = exp * sec_theta4 * (2.0 * sin_phi2 * tan_theta2 - alpha_y2);
                    numerator * d_alpha_y_rcp_denom
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect()
    }

    /// Compute the partial derivative of the masking-shadowing function with
    /// respect to the roughness parameters, αx and αy.
    ///
    /// NOTE: Currently, it's using isotropic Beckmann masking-shadowing
    /// function.
    fn msf_partial_derivative(&self, m: Vec3, i: Vec3, o: Vec3) -> f64 {
        if m.dot(i) <= 0.0 || m.dot(o) <= 0.0 {
            return 0.0;
        }
        // TODO
        let cos_theta_i = i.y as f64;
        let cos_theta_o = o.y as f64;
        let cos_theta_i2 = sqr(cos_theta_i);
        let cos_theta_o2 = sqr(cos_theta_o);
        let cot_theta_i2 = cos_theta_i2 * rcp_f64(1.0 - cos_theta_i2);
        let cot_theta_i = cot_theta_i2.sqrt();
        let cot_theta_o2 = cos_theta_o2 * rcp_f64(1.0 - cos_theta_o2);
        let cot_theta_o = cot_theta_o2.sqrt();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let cot_theta_i_over_alpha = cot_theta_i * rcp_f64(self.alpha_x);
        let cot_theta_o_over_alpha = cot_theta_o * rcp_f64(self.alpha_x);
        let nominator = -4.0
            * std::f64::consts::PI
            * cot_theta_i
            * cot_theta_o
            * ((cot_theta_i2 + cot_theta_o2) / sqr(self.alpha_x)).exp()
            * (2.0 * self.alpha_x
                + sqrt_pi
                    * cot_theta_i
                    * sqr(cot_theta_i_over_alpha).exp()
                    * libm::erf(cot_theta_i_over_alpha)
                + sqrt_pi
                    * cot_theta_o
                    * sqr(cot_theta_o_over_alpha).exp()
                    * (libm::erf(cot_theta_o_over_alpha) + 1.0)
                + sqrt_pi * cot_theta_i * sqr(cot_theta_i_over_alpha).exp());
        let denominator = sqr(self.alpha_x
            + sqrt_pi
                * cot_theta_i
                * sqr(cot_theta_i_over_alpha).exp()
                * (libm::erf(cot_theta_i_over_alpha) + 1.0))
            * sqr(self.alpha_x
                + sqrt_pi
                    * cot_theta_o
                    * sqr(cot_theta_o_over_alpha).exp()
                    * (libm::erf(cot_theta_o_over_alpha) + 1.0));
        nominator / denominator
    }
}

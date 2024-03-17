use crate::distro::{MicrofacetDistribution, MicrofacetDistroKind};
use base::{
    math::{cbr, cos_theta, rcp_f64, sqr, Vec3},
    Isotropy,
};

/// Beckmann microfacet distribution function.
///
/// Beckman distribution is based on the Gaussian distribution of microfacet
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
    /// Creates a new Beckmann distribution with the given roughness parameters.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannDistribution {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }

    /// Creates a new isotropic Beckmann distribution with the given roughness.
    pub fn new_isotropic(alpha: f64) -> Self {
        BeckmannDistribution {
            alpha_x: alpha.max(1.0e-6),
            alpha_y: alpha.max(1.0e-6),
        }
    }

    /// Returns whether the distribution is isotropic or anisotropic.
    #[inline]
    pub fn is_isotropic(&self) -> bool { (self.alpha_x - self.alpha_y).abs() < 1.0e-6 }
}

impl MicrofacetDistribution for BeckmannDistribution {
    fn params(&self) -> Self::Params { [self.alpha_x, self.alpha_y] }

    fn set_params(&mut self, params: &Self::Params) {
        self.alpha_x = params[0];
        self.alpha_y = params[1];
    }

    fn kind(&self) -> MicrofacetDistroKind { MicrofacetDistroKind::Beckmann }

    fn isotropy(&self) -> Isotropy {
        if self.is_isotropic() {
            Isotropy::Isotropic
        } else {
            Isotropy::Anisotropic
        }
    }

    fn eval_ndf(&self, cos_theta: f64, cos_phi: f64) -> f64 {
        let cos_theta2 = sqr(cos_theta);
        let e = if cos_theta2 < 1.0e-16 {
            1.0
        } else {
            let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
            let cos_phi2 = sqr(cos_phi);
            let sin_phi2 = 1.0 - cos_phi2;
            (-tan_theta2
                * (cos_phi2 * rcp_f64(sqr(self.alpha_x)) + sin_phi2 * rcp_f64(sqr(self.alpha_y))))
            .exp()
        };
        let cos_theta4 = sqr(cos_theta2);
        if cos_theta4 < 1.0e-16 {
            return 0.0;
        }
        e * rcp_f64(std::f64::consts::PI * cos_theta4 * self.alpha_x * self.alpha_y)
    }

    fn eval_msf1(&self, m: &Vec3, v: &Vec3) -> f64 {
        // TODO: anisotropic
        if m.dot(*v) <= 0.0 {
            0.0
        } else {
            let cos_theta_v2 = sqr(cos_theta(v) as f64);
            if cos_theta_v2 < 1.0e-8 {
                return 0.0;
            }
            let tan_theta_v2 = (1.0 - cos_theta_v2) * rcp_f64(cos_theta_v2);
            let tan_theta_v = if tan_theta_v2 < 1.0e-8 {
                0.0
            } else {
                tan_theta_v2.sqrt()
            };
            let alpha_tan_theta_v = self.alpha_x * tan_theta_v;
            let a = if alpha_tan_theta_v < 1.0e-8 {
                0.0
            } else {
                rcp_f64(alpha_tan_theta_v)
            };
            2.0 * rcp_f64(
                1.0 + libm::erf(a)
                    + (-sqr(a)).exp() * alpha_tan_theta_v * rcp_f64(std::f64::consts::PI.sqrt()),
            )
        }
    }

    fn clone_box(&self) -> Box<dyn MicrofacetDistribution<Params = Self::Params>> {
        Box::new(*self)
    }

    #[cfg(feature = "fitting")]
    fn pd_ndf(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Box<[f64]> {
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
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[cfg(feature = "fitting")]
    fn pd_ndf_iso(&self, cos_thetas: &[f64]) -> Box<[f64]> {
        let mut results = Box::new_uninit_slice(cos_thetas.len());
        let alpha2 = sqr(self.alpha_x);
        let alpha5 = cbr(self.alpha_x) * alpha2;
        for (pd, cos_theta) in results.iter_mut().zip(cos_thetas.iter()) {
            if cos_theta.abs() < 1.0e-6 {
                pd.write(0.0);
                continue;
            }
            let cos_theta2 = sqr(*cos_theta);
            let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
            if tan_theta2.is_infinite() {
                pd.write(0.0);
                continue;
            }
            let sec_theta4 = rcp_f64(sqr(cos_theta2));
            let exp = (-tan_theta2 * rcp_f64(alpha2)).exp();
            let d_alpha = {
                let numerator = 2.0 * exp * sec_theta4 * (tan_theta2 - alpha2);
                numerator * rcp_f64(std::f64::consts::PI * alpha5)
            };
            pd.write(d_alpha);
        }
        unsafe { results.assume_init() }
    }

    #[cfg(feature = "fitting")]
    /// Compute the partial derivative of the masking-shadowing function with
    /// respect to the roughness parameters, αx and αy.
    ///
    /// NOTE: Currently, it's using isotropic Beckmann masking-shadowing
    /// function.
    fn pd_msf(&self, m: Vec3, i: Vec3, o: Vec3) -> f64 {
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

use crate::{
    impl_microfacet_distribution_common_methods, MicrofacetDistributionFittingModel,
    MicrofacetDistributionModel, MicrofacetDistributionModelKind,
};
use vgcore::{
    math::{cube, rcp_f64, sqr},
    Isotropy,
};

/// Beckmann-Spizzichino microfacet distribution function.
///
/// Beckmann-Spizzichino distribution is based on the Gaussian distribution of
/// microfacet slopes. If σ is the RMS slope of the microfacets, then the alpha
/// parameter of the Beckmann distribution is given by: $\alpha = \sqrt{2}
/// \sigma$.
#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoDistribution {
    /// Roughness parameter of the MDF (αx = √2σx).
    pub alpha_x: f64,
    /// Roughness parameter of the MDF (αy = √2σy).
    pub alpha_y: f64,
}

impl BeckmannSpizzichinoDistribution {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannSpizzichinoDistribution {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }

    pub fn new_isotropic(alpha: f64) -> Self {
        BeckmannSpizzichinoDistribution {
            alpha_x: alpha.max(1.0e-6),
            alpha_y: alpha.max(1.0e-6),
        }
    }
}

impl MicrofacetDistributionModel for BeckmannSpizzichinoDistribution {
    fn kind(&self) -> MicrofacetDistributionModelKind {
        MicrofacetDistributionModelKind::BeckmannSpizzichino
    }

    impl_microfacet_distribution_common_methods!();

    fn eval_adf(&self, cos_theta: f64, cos_phi: f64) -> f64 {
        let cos_theta2 = sqr(cos_theta);
        let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
        let cos_theta4 = sqr(cos_theta2);
        if cos_theta4 < 1.0e-16 {
            return 0.0;
        }
        let alpha_xy = self.alpha_x * self.alpha_y;
        let cos_phi2 = sqr(cos_phi);
        let sin_phi2 = 1.0 - cos_phi2;
        let e = tan_theta2 * (cos_phi2 / sqr(self.alpha_x) + sin_phi2 / sqr(self.alpha_y));
        (-e).exp() * rcp_f64(std::f64::consts::PI * sqr(alpha_xy) * cos_theta4)
    }

    fn eval_msf(&self, cos_theta_i: f64, cos_theta_o: f64, cos_theta_h: f64) -> f64 { todo!() }

    fn clone_box(&self) -> Box<dyn MicrofacetDistributionModel> { Box::new(*self) }
}

impl MicrofacetDistributionFittingModel for BeckmannSpizzichinoDistribution {
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
        let alpha_x5 = sqr(alpha_x2) * self.alpha_x;
        let alpha_y5 = sqr(alpha_y2) * self.alpha_y;
        cos_thetas
            .iter()
            .zip(cos_phis.iter())
            .flat_map(|(cos_theta, cos_phi)| {
                let cos_theta2 = sqr(*cos_theta);
                let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
                let sec_theta4 = rcp_f64(sqr(cos_theta2));
                let cos_phi2 = sqr(*cos_phi);
                let sin_phi2 = 1.0 - cos_phi2;
                let exp = {
                    let e = -tan_theta2 * (cos_phi2 * rcp_alpha_x2 + sin_phi2 * rcp_alpha_y2);
                    e.exp()
                };
                let d_alpha_x = {
                    let numerator = 2.0 * exp * (sin_phi2 * tan_theta2 - alpha_x2) * sec_theta4;
                    numerator * rcp_f64(std::f64::consts::PI * alpha_x5 * alpha_y2)
                };
                let d_alpha_y = {
                    let numerator = 2.0 * exp * (cos_phi2 * tan_theta2 - alpha_y2) * sec_theta4;
                    numerator * rcp_f64(std::f64::consts::PI * alpha_x2 * alpha_y5)
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect()
    }

    fn msf_partial_derivatives(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Vec<f64> { todo!() }
}

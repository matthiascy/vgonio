use base::math::{rcp_f64, sqr, Vec3};

use crate::{
    dist::TrowbridgeReitzDistribution, impl_common_methods, MicrofacetBasedBrdfFittingModel,
    MicrofacetBasedBrdfModel, MicrofacetBasedBrdfModelKind, MicrofacetDistributionModel,
};

/// Trowbridge-Reitz(GGX) microfacet BRDF model.
/// See [Trowbridge-Reitz
/// Distribution](crate::dist::TrowbridgeReitzDistribution).
#[derive(Debug, Clone, Copy)]
pub struct TrowbridgeReitzBrdfModel {
    /// Roughness parameter of the originated from microfacet distribution
    pub alpha_x: f64,
    /// Roughness parameter of the originated from microfacet distribution
    pub alpha_y: f64,
}

impl TrowbridgeReitzBrdfModel {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        TrowbridgeReitzBrdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

impl MicrofacetBasedBrdfModel for TrowbridgeReitzBrdfModel {
    fn kind(&self) -> MicrofacetBasedBrdfModelKind { MicrofacetBasedBrdfModelKind::TrowbridgeReitz }

    impl_common_methods!();

    fn eval(&self, wi: Vec3, wo: Vec3) -> f64 {
        debug_assert!(wi.is_normalized(), "incident direction is not normalized");
        debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
        let wh = (wi + wo).normalize();
        let dist = TrowbridgeReitzDistribution::new(self.alpha_x, self.alpha_y);
        let d = dist.eval_adf(wh.z as f64, wh.y.atan2(wh.x) as f64);
        let g = dist.eval_msf1(wh, wi) * dist.eval_msf1(wh, wo);
        // TODO: eval_fresnel
        let f = 1.0;
        (d * g * f) / (4.0 * wi.z as f64 * wo.z as f64)
    }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBrdfModel> { Box::new(*self) }
}

impl MicrofacetBasedBrdfFittingModel for TrowbridgeReitzBrdfModel {
    fn partial_derivatives(&self, wis: &[Vec3], wos: &[Vec3]) -> Box<[f64]> {
        let mut result = Box::new_uninit_slice(wis.len() * wos.len() * 2);
        // TODO: eval_fresnel
        let f = 1.0;
        for i in 0..wis.len() {
            let wi = wis[i];
            for j in 0..wos.len() {
                let wo = wos[j];
                debug_assert!(wi.is_normalized(), "incident direction is not normalized");
                debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
                let wh = (wi + wo).normalize();
                let cos_theta_i = wi.z as f64;
                let cos_theta_o = wo.z as f64;
                let cos_theta_h = wh.z as f64;
                let cos_theta_h2 = cos_theta_h as f64 * cos_theta_h as f64;
                let tan_theta_h2 = (1.0 - cos_theta_h2) * rcp_f64(cos_theta_h2);
                let cos_phi_h = wh.y.atan2(wh.x) as f64;
                let cos_phi_h2 = cos_phi_h * cos_phi_h;
                let sin_phi_h2 = 1.0 - cos_phi_h2;
                let alpha_x2 = self.alpha_x * self.alpha_x;
                let alpha_y2 = self.alpha_y * self.alpha_y;
                let alpha_x4 = alpha_x2 * alpha_x2;
                let alpha_y4 = alpha_y2 * alpha_y2;
                let a = 1.0
                    + (cos_phi_h2 * rcp_f64(alpha_x2) + sin_phi_h2 * rcp_f64(alpha_y2))
                        * tan_theta_h2;

                let phi_h = wh.y.atan2(wh.x) as f64;
                let phi_i = wi.y.atan2(wi.x) as f64;
                let phi_hi = (phi_h - phi_i).abs() as f64;
                let cos_phi_hi = phi_hi.cos();
                let sin_phi_hi = phi_hi.sin();
                let cos_theta_hi = wi.dot(wh) as f64;
                let tan_theta_hi2 =
                    (1.0 - cos_theta_hi * cos_theta_hi) / (cos_theta_hi * cos_theta_hi);
                let b = (1.0
                    + alpha_x2 * cos_phi_hi * tan_theta_hi2
                    + alpha_y2 * sin_phi_hi * tan_theta_hi2)
                    .sqrt();

                let phi_o = wo.y.atan2(wo.x) as f64;
                let phi_ho = (phi_h - phi_o).abs();
                let cos_phi_ho = phi_ho.cos();
                let sin_phi_ho = phi_ho.sin();
                let cos_theta_ho = wo.dot(wh) as f64;
                let tan_theta_ho2 =
                    (1.0 - cos_theta_ho * cos_theta_ho) / (cos_theta_ho * cos_theta_ho);
                let c = (1.0
                    + alpha_x2 * cos_phi_ho * tan_theta_ho2
                    + alpha_y2 * sin_phi_ho * tan_theta_ho2)
                    .sqrt();

                let sec_theta_h4 = rcp_f64(cos_theta_h2 * cos_theta_h2);
                let sec_theta_i = rcp_f64(cos_theta_i);
                let sec_theta_o = rcp_f64(cos_theta_o);
                let cos_phi_h2 = cos_phi_h * cos_phi_h;

                let one_plus_b = 1.0 + b;
                let one_plus_c = 1.0 + c;
                let one_plus_b_one_plus_c = one_plus_b * one_plus_c;
                let one_plus_b_one_plus_c_sqr = sqr(one_plus_b_one_plus_c);
                let common_denominator =
                    std::f64::consts::PI * a * a * a * one_plus_b_one_plus_c_sqr;
                let tan_theta_ho2_over_c = tan_theta_ho2 * rcp_f64(c);
                let tan_theta_hi2_over_b = tan_theta_hi2 * rcp_f64(b);
                let sec_theta_h4_i_o = sec_theta_h4 * sec_theta_i * sec_theta_o;

                let nominator_x = f
                    * sec_theta_h4_i_o
                    * (-cos_phi_ho * a * one_plus_b * tan_theta_ho2_over_c
                        - cos_phi_hi * a * one_plus_c * tan_theta_hi2_over_b
                        + 4.0
                            * cos_phi_h2
                            * tan_theta_h2
                            * one_plus_b_one_plus_c
                            * rcp_f64(alpha_x4)
                        - a * one_plus_b_one_plus_c * rcp_f64(alpha_x2));
                let denominator_x = common_denominator * self.alpha_y;

                let nominator_y = f
                    * sec_theta_h4_i_o
                    * (-sin_phi_ho * a * one_plus_b * tan_theta_ho2_over_c
                        - sin_phi_hi * a * one_plus_c * tan_theta_hi2_over_b
                        + 4.0
                            * sin_phi_h2
                            * tan_theta_h2
                            * one_plus_b_one_plus_c
                            * rcp_f64(alpha_y4)
                        - a * one_plus_b_one_plus_c * rcp_f64(alpha_y2));

                let denominator_y = common_denominator * self.alpha_x;

                let dfr_dalpha_x = nominator_x * rcp_f64(denominator_x);
                let dfr_dalpha_y = nominator_y * rcp_f64(denominator_y);
                result[i * wos.len() * 2 + j * 2].write(dfr_dalpha_x);
                result[i * wos.len() * 2 + j * 2 + 1].write(dfr_dalpha_y);
            }
        }
        unsafe { result.assume_init() }
    }
}

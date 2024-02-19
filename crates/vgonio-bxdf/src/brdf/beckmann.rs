use base::{
    math::{rcp_f64, sqr, Vec3},
    optics::{fresnel, ior::RefractiveIndex},
};
use libm::erf;

use crate::{
    dist::BeckmannDistribution, impl_common_methods, MicrofacetBasedBrdfFittingModel,
    MicrofacetBasedBrdfModel, MicrofacetBasedBrdfModelKind, MicrofacetDistributionModel,
};

/// Beckmann microfacet BRDF model.
/// See [Beckmann Distribution](crate::dist::BeckmannDistribution).
#[derive(Debug, Clone, Copy)]
pub struct BeckmannBrdfModel {
    /// Roughness parameter of the originated from microfacet distribution
    /// function.
    pub alpha_x: f64,
    /// Roughness parameter of the originated from microfacet distribution
    pub alpha_y: f64,
}

impl BeckmannBrdfModel {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannBrdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

impl MicrofacetBasedBrdfModel for BeckmannBrdfModel {
    fn kind(&self) -> MicrofacetBasedBrdfModelKind { MicrofacetBasedBrdfModelKind::Beckmann }

    impl_common_methods!();

    fn eval(&self, wi: Vec3, wo: Vec3, ior_i: &RefractiveIndex, ior_t: &RefractiveIndex) -> f64 {
        debug_assert!(wi.is_normalized(), "incident direction is not normalized");
        debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
        let wh = (wi + wo).normalize();
        let dist = BeckmannDistribution::new(self.alpha_x, self.alpha_y);
        let d = dist.eval_adf(wh.z as f64, wh.y.atan2(wh.x) as f64);
        let g = dist.eval_msf1(wh, wi) * dist.eval_msf1(wh, wo);
        // TODO: test medium type
        let f = fresnel::reflectance_dielectric_conductor(wi.z.abs(), ior_i.eta, ior_t.eta, ior_t.k)
            as f64;
        (d * g * f) / (4.0 * wi.z as f64 * wo.z as f64)
    }

    fn eval_spectrum(
        &self,
        wi: Vec3,
        wo: Vec3,
        iors_i: &[RefractiveIndex],
        iors_t: &[RefractiveIndex],
    ) -> Box<[f64]> {
        debug_assert!(wi.is_normalized(), "incident direction is not normalized");
        debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
        debug_assert_eq!(
            iors_i.len(),
            iors_t.len(),
            "length of iors_i and iors_t must be equal"
        );
        // TODO: test medium type
        let mut result = Box::new_uninit_slice(iors_i.len());
        for i in 0..iors_i.len() {
            let ior_i = iors_i[i];
            let ior_t = iors_t[i];
            let wh = (wi + wo).normalize();
            let dist = BeckmannDistribution::new(self.alpha_x, self.alpha_y);
            let d = dist.eval_adf(wh.z as f64, wh.y.atan2(wh.x) as f64);
            let g = dist.eval_msf1(wh, wi) * dist.eval_msf1(wh, wo);
            let f = fresnel::reflectance_dielectric_conductor(
                wi.z.abs(),
                ior_i.eta,
                ior_t.eta,
                ior_t.k,
            ) as f64;
            result[i].write((d * g * f) / (4.0 * wi.z as f64 * wo.z as f64));
        }
        unsafe { result.assume_init() }
    }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBrdfModel> { Box::new(*self) }
}

impl MicrofacetBasedBrdfFittingModel for BeckmannBrdfModel {
    fn partial_derivatives(
        &self,
        wos: &[Vec3],
        wis: &[Vec3],
        ior_i: &RefractiveIndex,
        ior_t: &RefractiveIndex,
    ) -> Box<[f64]> {
        let mut result = Box::new_uninit_slice(wis.len() * wos.len() * 2);
        // TODO: test medium type
        for i in 0..wis.len() {
            let wi = wis[i];
            for j in 0..wos.len() {
                let wo = wos[j];
                debug_assert!(wi.is_normalized(), "incident direction is not normalized");
                debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
                let wh = (wi + wo).normalize();

                let f = fresnel::reflectance_dielectric_conductor(
                    wi.z.abs(),
                    ior_i.eta,
                    ior_t.eta,
                    ior_t.k,
                ) as f64;

                let alpha_x2 = self.alpha_x * self.alpha_x;
                let alpha_y2 = self.alpha_y * self.alpha_y;
                let alpha_x4 = alpha_x2 * alpha_x2;
                let alpha_y4 = alpha_y2 * alpha_y2;
                let rcp_alpha_x2 = rcp_f64(alpha_x2);
                let rcp_alpha_y2 = rcp_f64(alpha_y2);
                let rcp_alpha_x4 = rcp_f64(alpha_x4);
                let rcp_alpha_y4 = rcp_f64(alpha_y4);
                let cos_theta_i = wi.z as f64;
                let sec_theta_i = rcp_f64(cos_theta_i);
                let cos_theta_o = wo.z as f64;
                let sec_theta_o = rcp_f64(cos_theta_o);
                let cos_theta_h = wh.z as f64;
                let cos_theta_h2 = cos_theta_h * cos_theta_h;
                let sec_theta_h4 = rcp_f64(cos_theta_h2 * cos_theta_h2);
                let sin_theta_h2 = 1.0 - cos_theta_h2;
                let tan_theta_h2 = (1.0 - cos_theta_h2) / cos_theta_h2;
                let cos_phi_h = wh.y.atan2(wh.x);
                let cos_phi_h2 = (cos_phi_h * cos_phi_h) as f64;
                let sin_phi_h2 = 1.0 - cos_phi_h2;
                let c = (-tan_theta_h2 * cos_phi_h2 * rcp_f64(alpha_x2)
                    + sin_phi_h2 * rcp_f64(alpha_y2))
                .exp();

                let phi_h = wh.y.atan2(wh.x) as f64;
                let phi_i = wi.y.atan2(wi.x) as f64;
                let phi_o = wo.y.atan2(wo.x) as f64;
                let phi_hi = (phi_h - phi_i).abs();
                let phi_ho = (phi_h - phi_o).abs();
                let cos_phi_hi = phi_hi.cos();
                let sin_phi_hi = phi_hi.sin();
                let cos_phi_ho = phi_ho.cos();
                let sin_phi_ho = phi_ho.sin();
                let cos_phi_hi2 = sqr(cos_phi_hi);
                let sin_phi_hi2 = sqr(sin_phi_hi);
                let cos_phi_ho2 = sqr(cos_phi_ho);
                let sin_phi_ho2 = sqr(sin_phi_ho);
                let tan_theta_hi = sin_phi_hi * rcp_f64(cos_phi_hi);
                let tan_theta_ho = sin_phi_ho * rcp_f64(cos_phi_ho);

                let ai = (alpha_x2 * cos_phi_hi2 + alpha_y2 * sin_phi_hi2).sqrt();
                let ao = (alpha_x2 * cos_phi_ho2 + alpha_y2 * sin_phi_ho2).sqrt();
                let rcp_ai = rcp_f64(ai);
                let rcp_ao = rcp_f64(ao);
                let bi = rcp_f64(ai) * rcp_f64(tan_theta_hi);
                let bo = rcp_f64(ao) * rcp_f64(tan_theta_ho);
                let bi2 = sqr(bi);
                let bo2 = sqr(bo);
                let erf_bi = erf(bi);
                let erf_bo = erf(bo);
                let sqrt_pi = std::f64::consts::PI.sqrt();
                let d_i = (-bi2).exp() * ai * tan_theta_hi * rcp_f64(sqrt_pi);
                let d_o = (-bo2).exp() * ao * tan_theta_ho * rcp_f64(sqrt_pi);
                let gi = 1.0 + erf_bi + d_i;
                let go = 1.0 + erf_bo + d_o;
                let gi2 = sqr(gi);
                let go2 = sqr(go);
                let nominator_x = f
                    * c
                    * sec_theta_h4
                    * sec_theta_i
                    * sec_theta_o
                    * (-(-bo2).exp() * cos_phi_ho2 * gi * tan_theta_ho * rcp_ao
                        - (-bi2).exp() * cos_phi_hi2 * go * tan_theta_hi * rcp_ai
                        + (sqrt_pi * 2.0 * sin_theta_h2 * rcp_alpha_x4 - sqrt_pi * rcp_alpha_x2)
                            * gi
                            * go);
                let denominator_x = sqrt_pi * sqrt_pi * sqrt_pi * self.alpha_y * gi2 * go2;

                let nominator_y = f
                    * c
                    * sec_theta_h4
                    * sec_theta_i
                    * sec_theta_o
                    * (-(-bo2).exp() * sin_phi_ho2 * gi * tan_theta_ho * rcp_ao
                        - (-bi2).exp() * sin_phi_hi2 * go * tan_theta_hi * rcp_ai
                        - (sqrt_pi * gi * go * rcp_alpha_y2)
                        + (sqrt_pi * 2.0 * sin_theta_h2 * tan_theta_h2 * gi * go * rcp_alpha_y4));
                let denominator_y = sqrt_pi * sqrt_pi * sqrt_pi * self.alpha_x * gi2 * go2;

                let dfr_dalpha_x = nominator_x * rcp_f64(denominator_x);
                let dfr_dalpha_y = nominator_y * rcp_f64(denominator_y);
                result[i * 2 * wos.len() + j * 2].write(dfr_dalpha_x);
                result[i * 2 * wos.len() + j * 2 + 1].write(dfr_dalpha_y);
            }
        }
        unsafe { result.assume_init() }
    }
}

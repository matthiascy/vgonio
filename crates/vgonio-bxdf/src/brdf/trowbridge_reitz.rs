use std::fmt::{Debug, Formatter};
use base::{
    math::{cartesian_to_spherical, rcp_f64, spherical_to_cartesian, sqr, Vec3},
    optics::{fresnel, ior::RefractiveIndex},
};
use libm::log;
use log::log;

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

    fn eval(&self, wi: Vec3, wo: Vec3, ior_i: &RefractiveIndex, ior_t: &RefractiveIndex) -> f64 {
        debug_assert!(wi.is_normalized(), "incident direction is not normalized");p
        debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
        let wh = (wi + wo).normalize();
        let dist = TrowbridgeReitzDistribution::new(self.alpha_x, self.alpha_y);
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
            "the number of refractive indices for incident and transmitted media must be the same"
        );
        let mut result = Vec::with_capacity(iors_i.len());
        for i in 0..iors_i.len() {
            let ior_i = &iors_i[i];
            let ior_t = &iors_t[i];
            result.push(self.eval(wi, wo, ior_i, ior_t));
        }
        result.into_boxed_slice()
    }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBrdfModel> { Box::new(*self) }
}

impl MicrofacetBasedBrdfFittingModel for TrowbridgeReitzBrdfModel {
    fn partial_derivatives(
        &self,
        wis: &[Vec3],
        wos: &[Vec3],
        ior_i: &RefractiveIndex,
        ior_t: &RefractiveIndex,
    ) -> Box<[f64]> {
        let mut result = Box::new_uninit_slice(wis.len() * wos.len() * 2);
        log::debug!("---- ---- TrowbridgeReitzBrdfModel::partial_derivatives");
        // TODO: test medium type
        for i in 0..wis.len() {
            let wi = wis[i];
            for j in 0..wos.len() {
                let wo = wos[j];
                debug_assert!(wi.is_normalized(), "incident direction is not normalized");
                debug_assert!(wo.is_normalized(), "outgoing direction is not normalized");
                let wh = (wi + wo).normalize();
                let cos_theta_h = wh.z;
                let cos_theta_h2 = sqr(cos_theta_h as f64);
                let cos_theta_h4 = sqr(cos_theta_h2);

                if cos_theta_h4 < 1.0e-6 {
                    result[i * wos.len() * 2 + j * 2].write(0.0);
                    result[i * wos.len() * 2 + j * 2 + 1].write(0.0);
                    continue;
                }

                let tan_theta_h2 = (1.0 - cos_theta_h2) * rcp_f64(cos_theta_h2);
                if tan_theta_h2.is_infinite() {
                    result[i * wos.len() * 2 + j * 2].write(0.0);
                    result[i * wos.len() * 2 + j * 2 + 1].write(0.0);
                    continue;
                }

                let cos_theta_i = wi.z;
                let cos_theta_o = wo.z;
                let f = fresnel::reflectance_dielectric_conductor(
                    cos_theta_i.abs(),
                    ior_i.eta,
                    ior_t.eta,
                    ior_t.k,
                ) as f64;

                let alpha_x2 = self.alpha_x * self.alpha_x;
                let alpha_y2 = self.alpha_y * self.alpha_y;
                let alpha_x4 = alpha_x2 * alpha_x2;
                let alpha_y4 = alpha_y2 * alpha_y2;

                let (_, _, phi_h) = cartesian_to_spherical(wh, 1.0);
                let (_, _, phi_o) = cartesian_to_spherical(wo, 1.0);
                let (_, _, phi_i) = cartesian_to_spherical(wi, 1.0);
                let phi_hi = (phi_h - phi_i).abs().as_f64();
                let cos_phi_hi = phi_hi.cos();
                let sin_phi_hi = phi_hi.sin();
                let cos_theta_hi = wi.dot(wh) as f64;
                let tan_theta_hi2 =
                    (1.0 - cos_theta_hi * cos_theta_hi) * rcp_f64(cos_theta_hi * cos_theta_hi);

                let ahi = {
                    let squared = 1.0
                        + alpha_x2 * cos_phi_hi * tan_theta_hi2
                        + alpha_y2 * sin_phi_hi * tan_theta_hi2;
                    if squared < 0.0 {
                        0.0
                    } else {
                        squared.sqrt()
                    }
                };

                let phi_ho = (phi_h - phi_o).abs().as_f64();
                let cos_phi_ho = phi_ho.cos();
                let sin_phi_ho = phi_ho.sin();
                let cos_theta_ho = wo.dot(wh) as f64;
                let tan_theta_ho2 =
                    (1.0 - cos_theta_ho * cos_theta_ho) * rcp_f64(cos_theta_ho * cos_theta_ho);

                let aho = {
                    let squared = 1.0
                        + alpha_x2 * cos_phi_ho * tan_theta_ho2
                        + alpha_y2 * sin_phi_ho * tan_theta_ho2;
                    if squared < 0.0 {
                        0.0
                    } else {
                        squared.sqrt()
                    }
                };

                let cos_phi_h = phi_h.cos() as f64;
                let sin_phi_h = phi_h.sin() as f64;
                let b = 1.0
                    + (sqr(cos_phi_h) * rcp_f64(alpha_x2) + sqr(sin_phi_h) * rcp_f64(alpha_y2))
                        * tan_theta_h2;

                let one_plus_ahi = 1.0 + ahi;
                let one_plus_aho = 1.0 + aho;
                let one_plus_ahi2 = sqr(one_plus_ahi);
                let one_plus_aho2 = sqr(one_plus_aho);

                let common = std::f64::consts::PI
                    * b
                    * b
                    * b
                    * one_plus_ahi2
                    * one_plus_aho2
                    * cos_theta_h4
                    * cos_theta_i as f64
                    * cos_theta_o as f64;

                let coeff_dfr_dalpha_x = f * rcp_f64(common * self.alpha_y);

                let cos_phi_h2 = sqr(cos_phi_h);
                let rcp_aho = if aho.abs() < 1.0e-6 { 0.0 } else { 1.0 / aho };
                let rcp_ahi = if ahi.abs() < 1.0e-6 { 0.0 } else { 1.0 / ahi };
                let dfr_dalpha_x = coeff_dfr_dalpha_x
                    * (-cos_phi_ho * b * one_plus_ahi * tan_theta_ho2 * rcp_aho
                        - cos_phi_hi * b * one_plus_aho * tan_theta_hi2 * rcp_ahi
                        + (4.0 * cos_phi_h2 * tan_theta_h2 * rcp_f64(alpha_x4)
                            - b * rcp_f64(alpha_x2))
                            * one_plus_ahi
                            * one_plus_aho);

                let coeff_dfr_dalpha_y = f * rcp_f64(common * self.alpha_x);
                let sin_phi_h2 = sqr(sin_phi_h);
                let dfr_dalpha_y = coeff_dfr_dalpha_y
                    * (-sin_phi_ho * b * one_plus_ahi * tan_theta_ho2 * rcp_aho
                        - sin_phi_hi * b * one_plus_aho * tan_theta_hi2 * rcp_ahi
                        + (4.0 * sin_phi_h2 * tan_theta_h2 * rcp_f64(alpha_y4)
                            - b * rcp_f64(alpha_y2))
                            * one_plus_ahi
                            * one_plus_aho);

                assert_ne!(
                    dfr_dalpha_x.is_infinite() || dfr_dalpha_y.is_infinite(),
                    true,
                    "inf: alpha_x = {}, alpha_y = {}, phi_h = {}, phi_o = {}, phi_i = {}, phi_hi \
                     = {}, phi_ho = {}, cos_theta_h = {}, cos_theta_i = {}, cos_theta_o = {}, f = \
                     {}, cos_phi_hi = {}, sin_phi_hi = {}, cos_theta_hi = {}, tan_theta_hi2 = {}, \
                     ahi2 = {}, ahi = {}, cos_phi_ho = {}, sin_phi_ho = {}, cos_theta_ho = {}, \
                     tan_theta_ho2 = {}, aho = {}, cos_phi_h = {}, sin_phi_h = {}, b = {}, \
                     one_plus_ahi = {}, one_plus_aho = {}, one_plus_ahi2 = {}, one_plus_aho2 = \
                     {}, common = {}, coeff_dfr_dalpha_x = {}, dfr_dalpha_x = {}, \
                     coeff_dfr_dalpha_y = {}, dfr_dalpha_y = {} at wi = {:?} - <{}, {}> , wo = \
                     {:?} - <{}, {}>, wh = {:?} - <{}, {}>",
                    self.alpha_x,
                    self.alpha_y,
                    phi_h,
                    phi_o,
                    phi_i,
                    phi_hi,
                    phi_ho,
                    cos_theta_h,
                    cos_theta_i,
                    cos_theta_o,
                    f,
                    cos_phi_hi,
                    sin_phi_hi,
                    cos_theta_hi,
                    tan_theta_hi2,
                    1.0 + alpha_x2 * cos_phi_hi * tan_theta_hi2
                        + alpha_y2 * sin_phi_hi * tan_theta_hi2,
                    ahi,
                    cos_phi_ho,
                    sin_phi_ho,
                    cos_theta_ho,
                    tan_theta_ho2,
                    aho,
                    cos_phi_h,
                    sin_phi_h,
                    b,
                    one_plus_ahi,
                    one_plus_aho,
                    one_plus_ahi2,
                    one_plus_aho2,
                    common,
                    coeff_dfr_dalpha_x,
                    dfr_dalpha_x,
                    coeff_dfr_dalpha_y,
                    dfr_dalpha_y,
                    wi,
                    cartesian_to_spherical(wi, 1.0).1.to_degrees(),
                    cartesian_to_spherical(wi, 1.0).2.to_degrees(),
                    wo,
                    cartesian_to_spherical(wo, 1.0).1.to_degrees(),
                    cartesian_to_spherical(wo, 1.0).2.to_degrees(),
                    wh,
                    cartesian_to_spherical(wh, 1.0).1.to_degrees(),
                    cartesian_to_spherical(wh, 1.0).2.to_degrees(),
                );

                result[i * wos.len() * 2 + j * 2].write(dfr_dalpha_x);
                result[i * wos.len() * 2 + j * 2 + 1].write(dfr_dalpha_y);
            }
        }
        unsafe { result.assume_init() }
    }
}

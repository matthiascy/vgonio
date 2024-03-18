use base::{
    math::{cart_to_sph, cos_theta, rcp_f64, sqr, Vec3},
    optics::{fresnel, ior::Ior},
};
use std::fmt::Debug;

use crate::{
    brdf::{microfacet::MicrofacetBrdf, Bxdf, BxdfFamily},
    distro::{MicrofacetDistribution, MicrofacetDistroKind, TrowbridgeReitzDistribution},
};

/// Microfacet BRDF model
pub type TrowbridgeReitzBrdf = MicrofacetBrdf<TrowbridgeReitzDistribution>;

impl TrowbridgeReitzBrdf {
    /// Creates a new microfacet BRDF model with the given roughness parameters.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        MicrofacetBrdf::from(TrowbridgeReitzDistribution::new(alpha_x, alpha_y))
    }
}

impl Bxdf for TrowbridgeReitzBrdf {
    type Params = <TrowbridgeReitzDistribution as MicrofacetDistribution>::Params;

    fn family(&self) -> BxdfFamily { BxdfFamily::Microfacet }

    fn distro(&self) -> Option<MicrofacetDistroKind> { Some(MicrofacetDistroKind::TrowbridgeReitz) }

    fn isotropic(&self) -> bool { self.distro.is_isotropic() }

    fn params(&self) -> Self::Params { self.distro.params() }

    fn set_params(&mut self, params: &Self::Params) { self.distro.set_params(params) }

    fn eval(&self, wi: &Vec3, wo: &Vec3) -> f64 {
        debug_assert!(wi.is_normalized(), "Incident direction is not normalized.");
        debug_assert!(wo.is_normalized(), "Outgoing direction is not normalized.");
        let cos_theta_i = cos_theta(wi);
        let cos_theta_o = cos_theta(wo);
        let cos_theta_io = (cos_theta_i * cos_theta_o) as f64;
        if cos_theta_io <= 1e-16 {
            return 0.0;
        }
        let wh = (*wi + *wo).normalize();
        let wh_sph = cart_to_sph(wh);
        let d = self
            .distro
            .eval_ndf(wh_sph.theta.as_f64().cos(), wh_sph.phi.as_f64().cos());
        let g = self.distro.eval_msf1(wh, *wi) * self.distro.eval_msf1(wh, *wo);
        (d * g) / (4.0 * cos_theta_io)
    }

    fn eval_hd(&self, wh: &Vec3, wd: &Vec3) -> f64 { todo!() }

    fn evalp(&self, wi: &Vec3, wo: &Vec3) -> f64 { todo!() }

    fn evalp_hd(&self, wh: &Vec3, wd: &Vec3) -> f64 { todo!() }

    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> f64 { todo!() }

    fn sample(&self, u: f32, v: f32, wo: &Vec3) -> f64 { todo!() }

    fn pdf(&self, wi: &Vec3, wo: &Vec3) -> f64 { todo!() }

    #[cfg(feature = "fitting")]
    fn pd(&self, wis: &[Vec3], wos: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]> {
        let mut result = Box::new_uninit_slice(wis.len() * wos.len() * 2);
        let [alpha_x, alpha_y] = self.distro.params();
        for i in 0..wis.len() {
            let wi = wis[i];
            for j in 0..wos.len() {
                let wo = wos[j];
                debug_assert!(wi.is_normalized(), "Incident direction is not normalized");
                debug_assert!(wo.is_normalized(), "Outgoing direction is not normalized");
                let wh = (wi + wo).normalize();
                let cos_theta_h = cos_theta(&wh);
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
                let f = fresnel::reflectance(cos_theta(&-wi), ior_i, ior_t) as f64;
                let cos_theta_i = cos_theta(&wi);
                let cos_theta_o = cos_theta(&wo);
                let alpha_x2 = sqr(alpha_x);
                let alpha_y2 = sqr(alpha_y);
                let alpha_x4 = sqr(alpha_x2);
                let alpha_y4 = sqr(alpha_y2);

                let phi_h = cart_to_sph(wh).phi;
                let phi_o = cart_to_sph(wo).phi;
                let phi_i = cart_to_sph(wi).phi;
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

                let coeff_dfr_dalpha_x = f * rcp_f64(common * alpha_y);

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

                let coeff_dfr_dalpha_y = f * rcp_f64(common * alpha_x);
                let sin_phi_h2 = sqr(sin_phi_h);
                let dfr_dalpha_y = coeff_dfr_dalpha_y
                    * (-sin_phi_ho * b * one_plus_ahi * tan_theta_ho2 * rcp_aho
                        - sin_phi_hi * b * one_plus_aho * tan_theta_hi2 * rcp_ahi
                        + (4.0 * sin_phi_h2 * tan_theta_h2 * rcp_f64(alpha_y4)
                            - b * rcp_f64(alpha_y2))
                            * one_plus_ahi
                            * one_plus_aho);

                result[i * wos.len() * 2 + j * 2].write(dfr_dalpha_x);
                result[i * wos.len() * 2 + j * 2 + 1].write(dfr_dalpha_y);
            }
        }
        unsafe { result.assume_init() }
    }

    #[cfg(feature = "fitting")]
    fn pd_iso(&self, wis: &[Vec3], wos: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]> {
        debug_assert!(self.distro.is_isotropic());
        let mut result = Box::new_uninit_slice(wis.len() * wos.len());
        for i in 0..wis.len() {
            let wi = wis[i];
            for j in 0..wos.len() {
                let wo = wos[j];
                debug_assert!(wi.is_normalized(), "Incident direction is not normalized");
                debug_assert!(wo.is_normalized(), "Outgoing direction is not normalized");
                let wh = (wi + wo).normalize();
                let cos_theta_h = cos_theta(&wh).abs();
                let cos_theta_h2 = sqr(cos_theta_h as f64);
                let cos_theta_h4 = sqr(cos_theta_h2);
                let cos_theta_i = wi.z.abs();
                let cos_theta_o = wo.z.abs();

                if cos_theta_h4 < 1e-16 || cos_theta_i.abs() < 1e-16 || cos_theta_o.abs() < 1e-16 {
                    result[i * wos.len() + j].write(0.0);
                    continue;
                }

                let tan_theta_h2 = (1.0 - cos_theta_h2) * rcp_f64(cos_theta_h2);
                if tan_theta_h2.is_infinite() {
                    result[i * wos.len() + j].write(0.0);
                    continue;
                }

                let f = fresnel::reflectance(cos_theta(&-wi), ior_i, ior_t) as f64;

                let alpha = self.params()[0];
                let alpha2 = sqr(alpha);

                let cos_theta_hi = wi.dot(wh).abs() as f64;
                let cos_theta_hi2 = sqr(cos_theta_hi);
                let tan_theta_hi2 = (1.0 - cos_theta_hi2) * rcp_f64(cos_theta_hi2);
                let ai = (1.0 + alpha2 * tan_theta_hi2).sqrt();

                let cos_theta_ho = wo.dot(wh).abs() as f64;
                let cos_theta_ho2 = sqr(cos_theta_ho);
                let tan_theta_ho2 = (1.0 - cos_theta_ho2) * rcp_f64(cos_theta_ho2);
                let ao = (1.0 + alpha2 * tan_theta_ho2).sqrt();

                let one_plus_ai = 1.0 + ai;
                let one_plus_ao = 1.0 + ao;
                let one_plus_ai2 = sqr(one_plus_ai);
                let one_plus_ao2 = sqr(one_plus_ao);

                let part_one = -f
                    * rcp_f64(
                        std::f64::consts::PI
                            * cos_theta_h4
                            * cos_theta_i as f64
                            * cos_theta_o as f64,
                    );

                let nominator =
                    (alpha.powi(3) * one_plus_ai * (2.0 + 2.0 * ao + 3.0 * alpha2 * tan_theta_ho2))
                        + (alpha.powi(5)
                            * tan_theta_hi2
                            * (3.0 + 3.0 * ao + 4.0 * alpha2 * tan_theta_ho2))
                        - (alpha
                            * tan_theta_h2
                            * (alpha2 * one_plus_ao * tan_theta_hi2
                                + one_plus_ai2 * (2.0 + 2.0 * ao + alpha2 * tan_theta_ho2)));
                let denominator =
                    one_plus_ai2 * one_plus_ao2 * (alpha2 + tan_theta_h2).powi(3) * ai * ao;
                result[i * wos.len() + j].write(part_one * nominator * rcp_f64(denominator));
            }
        }
        unsafe { result.assume_init() }
    }
}

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

// impl MicrofacetBrdfModel for TrowbridgeReitzBrdfModel {
//     fn kind(&self) -> MicrofacetBrdfKind {
// MicrofacetBrdfKind::TrowbridgeReitz }
//
//     impl_common_methods!();
//
//     fn eval(
//         &self,
//         wi: Vec3,
//         wo: Vec3,
//         ior_i: &RefractiveIndexRecord,
//         ior_t: &RefractiveIndexRecord,
//     ) -> f64 {
//         debug_assert!(wi.is_normalized(), "incident direction is not
// normalized");         debug_assert!(wo.is_normalized(), "outgoing direction
// is not normalized");         let cos_theta_i = wi.z;
//         let cos_theta_o = wo.z;
//         let cos_theta_io = (cos_theta_i * cos_theta_o) as f64;
//         if cos_theta_io <= 1e-16 {
//             return 0.0;
//         }
//         let wh = (wi + wo).normalize();
//         let dist = TrowbridgeReitzDistribution::new(self.alpha_x,
// self.alpha_y);         let wh_sph = cart_to_sph(wh);
//         let d = dist.eval_adf(wh_sph.theta.as_f64().cos(),
// wh_sph.phi.as_f64().cos());         let g = dist.eval_msf1(wh, wi) *
// dist.eval_msf1(wh, wo);         // TODO: test medium type
//         let f = fresnel::reflectance_dielectric_conductor(
//             cos_theta_i.abs(),
//             ior_i.eta,
//             ior_t.eta,
//             ior_t.k,
//         ) as f64;
//         (d * g * f) / (4.0 * cos_theta_io)
//     }
//
//     fn eval_spectrum(
//         &self,
//         wi: Vec3,
//         wo: Vec3,
//         iors_i: &[RefractiveIndexRecord],
//         iors_t: &[RefractiveIndexRecord],
//     ) -> Box<[f64]> {
//         debug_assert!(wi.is_normalized(), "incident direction is not
// normalized");         debug_assert!(wo.is_normalized(), "outgoing direction
// is not normalized");         debug_assert_eq!(
//             iors_i.len(),
//             iors_t.len(),
//             "the number of refractive indices for incident and transmitted
// media must be the same"         );
//         let mut result = Vec::with_capacity(iors_i.len());
//         for i in 0..iors_i.len() {
//             let ior_i = &iors_i[i];
//             let ior_t = &iors_t[i];
//             result.push(self.eval(wi, wo, ior_i, ior_t));
//         }
//         result.into_boxed_slice()
//     }
//
//     fn clone_box(&self) -> Box<dyn MicrofacetBrdfModel> { Box::new(*self) }
// }
//
// impl MicrofacetBrdfFittingModel for TrowbridgeReitzBrdfModel {
//     fn partial_derivatives(
//         &self,
//         wis: &[Vec3],
//         wos: &[Vec3],
//         ior_i: &RefractiveIndexRecord,
//         ior_t: &RefractiveIndexRecord,
//     ) -> Box<[f64]> {
//         let mut result = Box::new_uninit_slice(wis.len() * wos.len() * 2);
//         // TODO: test medium type
//         for i in 0..wis.len() {
//             let wi = wis[i];
//             for j in 0..wos.len() {
//                 let wo = wos[j];
//                 debug_assert!(wi.is_normalized(), "incident direction is not
// normalized");                 debug_assert!(wo.is_normalized(), "outgoing
// direction is not normalized");                 let wh = (wi +
// wo).normalize();                 let cos_theta_h = wh.z;
//                 let cos_theta_h2 = sqr(cos_theta_h as f64);
//                 let cos_theta_h4 = sqr(cos_theta_h2);
//
//                 if cos_theta_h4 < 1.0e-6 {
//                     result[i * wos.len() * 2 + j * 2].write(0.0);
//                     result[i * wos.len() * 2 + j * 2 + 1].write(0.0);
//                     continue;
//                 }
//
//                 let tan_theta_h2 = (1.0 - cos_theta_h2) *
// rcp_f64(cos_theta_h2);                 if tan_theta_h2.is_infinite() {
//                     result[i * wos.len() * 2 + j * 2].write(0.0);
//                     result[i * wos.len() * 2 + j * 2 + 1].write(0.0);
//                     continue;
//                 }
//
//                 let cos_theta_i = wi.z;
//                 let cos_theta_o = wo.z;
//                 let f = fresnel::reflectance_dielectric_conductor(
//                     cos_theta_i.abs(),
//                     ior_i.eta,
//                     ior_t.eta,
//                     ior_t.k,
//                 ) as f64;
//
//                 let alpha_x2 = self.alpha_x * self.alpha_x;
//                 let alpha_y2 = self.alpha_y * self.alpha_y;
//                 let alpha_x4 = alpha_x2 * alpha_x2;
//                 let alpha_y4 = alpha_y2 * alpha_y2;
//
//                 let phi_h = cart_to_sph(wh).phi;
//                 let phi_o = cart_to_sph(wo).phi;
//                 let phi_i = cart_to_sph(wi).phi;
//                 let phi_hi = (phi_h - phi_i).abs().as_f64();
//                 let cos_phi_hi = phi_hi.cos();
//                 let sin_phi_hi = phi_hi.sin();
//                 let cos_theta_hi = wi.dot(wh) as f64;
//                 let tan_theta_hi2 =
//                     (1.0 - cos_theta_hi * cos_theta_hi) *
// rcp_f64(cos_theta_hi * cos_theta_hi);
//
//                 let ahi = {
//                     let squared = 1.0
//                         + alpha_x2 * cos_phi_hi * tan_theta_hi2
//                         + alpha_y2 * sin_phi_hi * tan_theta_hi2;
//                     if squared < 0.0 {
//                         0.0
//                     } else {
//                         squared.sqrt()
//                     }
//                 };
//
//                 let phi_ho = (phi_h - phi_o).abs().as_f64();
//                 let cos_phi_ho = phi_ho.cos();
//                 let sin_phi_ho = phi_ho.sin();
//                 let cos_theta_ho = wo.dot(wh) as f64;
//                 let tan_theta_ho2 =
//                     (1.0 - cos_theta_ho * cos_theta_ho) *
// rcp_f64(cos_theta_ho * cos_theta_ho);
//
//                 let aho = {
//                     let squared = 1.0
//                         + alpha_x2 * cos_phi_ho * tan_theta_ho2
//                         + alpha_y2 * sin_phi_ho * tan_theta_ho2;
//                     if squared < 0.0 {
//                         0.0
//                     } else {
//                         squared.sqrt()
//                     }
//                 };
//
//                 let cos_phi_h = phi_h.cos() as f64;
//                 let sin_phi_h = phi_h.sin() as f64;
//                 let b = 1.0
//                     + (sqr(cos_phi_h) * rcp_f64(alpha_x2) + sqr(sin_phi_h) *
//                       rcp_f64(alpha_y2))
//                         * tan_theta_h2;
//
//                 let one_plus_ahi = 1.0 + ahi;
//                 let one_plus_aho = 1.0 + aho;
//                 let one_plus_ahi2 = sqr(one_plus_ahi);
//                 let one_plus_aho2 = sqr(one_plus_aho);
//
//                 let common = std::f64::consts::PI
//                     * b
//                     * b
//                     * b
//                     * one_plus_ahi2
//                     * one_plus_aho2
//                     * cos_theta_h4
//                     * cos_theta_i as f64
//                     * cos_theta_o as f64;
//
//                 let coeff_dfr_dalpha_x = f * rcp_f64(common * self.alpha_y);
//
//                 let cos_phi_h2 = sqr(cos_phi_h);
//                 let rcp_aho = if aho.abs() < 1.0e-6 { 0.0 } else { 1.0 / aho
// };                 let rcp_ahi = if ahi.abs() < 1.0e-6 { 0.0 } else { 1.0 /
// ahi };                 let dfr_dalpha_x = coeff_dfr_dalpha_x
//                     * (-cos_phi_ho * b * one_plus_ahi * tan_theta_ho2 *
//                       rcp_aho
//                         - cos_phi_hi * b * one_plus_aho * tan_theta_hi2 *
//                           rcp_ahi
//                         + (4.0 * cos_phi_h2 * tan_theta_h2 *
//                           rcp_f64(alpha_x4)
//                             - b * rcp_f64(alpha_x2))
//                             * one_plus_ahi
//                             * one_plus_aho);
//
//                 let coeff_dfr_dalpha_y = f * rcp_f64(common * self.alpha_x);
//                 let sin_phi_h2 = sqr(sin_phi_h);
//                 let dfr_dalpha_y = coeff_dfr_dalpha_y
//                     * (-sin_phi_ho * b * one_plus_ahi * tan_theta_ho2 *
//                       rcp_aho
//                         - sin_phi_hi * b * one_plus_aho * tan_theta_hi2 *
//                           rcp_ahi
//                         + (4.0 * sin_phi_h2 * tan_theta_h2 *
//                           rcp_f64(alpha_y4)
//                             - b * rcp_f64(alpha_y2))
//                             * one_plus_ahi
//                             * one_plus_aho);
//
//                 assert_ne!(
//                     dfr_dalpha_x.is_infinite() || dfr_dalpha_y.is_infinite(),
//                     true,
//                     "inf: alpha_x = {}, alpha_y = {}, phi_h = {}, phi_o = {},
// phi_i = {}, phi_hi \                      = {}, phi_ho = {}, cos_theta_h =
// {}, cos_theta_i = {}, cos_theta_o = {}, f = \                      {},
// cos_phi_hi = {}, sin_phi_hi = {}, cos_theta_hi = {}, tan_theta_hi2 = {}, \
//                      ahi2 = {}, ahi = {}, cos_phi_ho = {}, sin_phi_ho = {},
// cos_theta_ho = {}, \                      tan_theta_ho2 = {}, aho = {},
// cos_phi_h = {}, sin_phi_h = {}, b = {}, \                      one_plus_ahi =
// {}, one_plus_aho = {}, one_plus_ahi2 = {}, one_plus_aho2 = \
// {}, common = {}, coeff_dfr_dalpha_x = {}, dfr_dalpha_x = {}, \
// coeff_dfr_dalpha_y = {}, dfr_dalpha_y = {} at wi = {:?} - <{}, {}> , wo = \
//                      {:?} - <{}, {}>, wh = {:?} - <{}, {}>",
//                     self.alpha_x,
//                     self.alpha_y,
//                     phi_h,
//                     phi_o,
//                     phi_i,
//                     phi_hi,
//                     phi_ho,
//                     cos_theta_h,
//                     cos_theta_i,
//                     cos_theta_o,
//                     f,
//                     cos_phi_hi,
//                     sin_phi_hi,
//                     cos_theta_hi,
//                     tan_theta_hi2,
//                     1.0 + alpha_x2 * cos_phi_hi * tan_theta_hi2
//                         + alpha_y2 * sin_phi_hi * tan_theta_hi2,
//                     ahi,
//                     cos_phi_ho,
//                     sin_phi_ho,
//                     cos_theta_ho,
//                     tan_theta_ho2,
//                     aho,
//                     cos_phi_h,
//                     sin_phi_h,
//                     b,
//                     one_plus_ahi,
//                     one_plus_aho,
//                     one_plus_ahi2,
//                     one_plus_aho2,
//                     common,
//                     coeff_dfr_dalpha_x,
//                     dfr_dalpha_x,
//                     coeff_dfr_dalpha_y,
//                     dfr_dalpha_y,
//                     wi,
//                     cart_to_sph(wi).theta.to_degrees(),
//                     cart_to_sph(wi).phi.to_degrees(),
//                     wo,
//                     cart_to_sph(wo).theta.to_degrees(),
//                     cart_to_sph(wo).phi.to_degrees(),
//                     wh,
//                     cart_to_sph(wh).theta.to_degrees(),
//                     cart_to_sph(wh).phi.to_degrees(),
//                 );
//
//                 result[i * wos.len() * 2 + j * 2].write(dfr_dalpha_x);
//                 result[i * wos.len() * 2 + j * 2 + 1].write(dfr_dalpha_y);
//             }
//         }
//         unsafe { result.assume_init() }
//     }
//
//     fn partial_derivatives_isotropic(
//         &self,
//         wis: &[Vec3],
//         wos: &[Vec3],
//         ior_i: &RefractiveIndexRecord,
//         ior_t: &RefractiveIndexRecord,
//     ) -> Box<[f64]> {
//         let mut result = Box::new_uninit_slice(wis.len() * wos.len());
//         // TODO: test medium type
//         for i in 0..wis.len() {
//             let wi = wis[i];
//             for j in 0..wos.len() {
//                 let wo = wos[j];
//                 debug_assert!(wi.is_normalized(), "incident direction is not
// normalized");                 debug_assert!(wo.is_normalized(), "outgoing
// direction is not normalized");                 let wh = (wi +
// wo).normalize();                 let cos_theta_h = wh.z.abs();
//                 let cos_theta_h2 = sqr(cos_theta_h as f64);
//                 let cos_theta_h4 = sqr(cos_theta_h2);
//                 let cos_theta_i = wi.z.abs();
//                 let cos_theta_o = wo.z.abs();
//
//                 if cos_theta_h4 < 1e-16 || cos_theta_i.abs() < 1e-16 ||
// cos_theta_o.abs() < 1e-16 {                     result[i * wos.len() +
// j].write(0.0);                     continue;
//                 }
//
//                 let tan_theta_h2 = (1.0 - cos_theta_h2) *
// rcp_f64(cos_theta_h2);                 if tan_theta_h2.is_infinite() {
//                     result[i * wos.len() + j].write(0.0);
//                     continue;
//                 }
//
//                 let f = fresnel::reflectance_dielectric_conductor(
//                     cos_theta_i.abs(),
//                     ior_i.eta,
//                     ior_t.eta,
//                     ior_t.k,
//                 ) as f64;
//
//                 let alpha = self.alpha_x;
//                 let alpha2 = sqr(self.alpha_x);
//
//                 let cos_theta_hi = wi.dot(wh).abs() as f64;
//                 let cos_theta_hi2 = sqr(cos_theta_hi);
//                 let tan_theta_hi2 = (1.0 - cos_theta_hi2) *
// rcp_f64(cos_theta_hi2);                 let ai = (1.0 + alpha2 *
// tan_theta_hi2).sqrt();
//
//                 let cos_theta_ho = wo.dot(wh).abs() as f64;
//                 let cos_theta_ho2 = sqr(cos_theta_ho);
//                 let tan_theta_ho2 = (1.0 - cos_theta_ho2) *
// rcp_f64(cos_theta_ho2);                 let ao = (1.0 + alpha2 *
// tan_theta_ho2).sqrt();
//
//                 let one_plus_ai = 1.0 + ai;
//                 let one_plus_ao = 1.0 + ao;
//                 let one_plus_ai2 = sqr(one_plus_ai);
//                 let one_plus_ao2 = sqr(one_plus_ao);
//
//                 let part_one = -f
//                     * rcp_f64( std::f64::consts::PI
//                             * cos_theta_h4
//                             * cos_theta_i as f64
//                             * cos_theta_o as f64,
//                     );
//
//                 let nominator =
//                     (alpha.powi(3) * one_plus_ai * (2.0 + 2.0 * ao + 3.0 *
// alpha2 * tan_theta_ho2))
//                         + (alpha.powi(5)
//                             * tan_theta_hi2
//                             * (3.0 + 3.0 * ao + 4.0 * alpha2 *
//                               tan_theta_ho2))
//                         - (alpha
//                             * tan_theta_h2
//                             * (alpha2 * one_plus_ao * tan_theta_hi2
//                                 + one_plus_ai2 * (2.0 + 2.0 * ao + alpha2 *
//                                   tan_theta_ho2)));
//                 let denominator =
//                     one_plus_ai2 * one_plus_ao2 * (alpha2 +
// tan_theta_h2).powi(3) * ai * ao;                 result[i * wos.len() +
// j].write(part_one * nominator * rcp_f64(denominator));             }
//         }
//         unsafe { result.assume_init() }
//     }
//
//     fn as_ref(&self) -> &dyn MicrofacetBrdfModel { self }
// }

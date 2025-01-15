use crate::math::{cart_to_sph, cos_theta, Vec3};
use std::fmt::Debug;

#[cfg(feature = "bxdf_fit")]
use crate::{
    math::{rcp_f64, sqr},
    optics::{fresnel, ior::Ior},
};

use crate::bxdf::{
    brdf::{analytical::microfacet::MicrofacetBrdf, Bxdf, BxdfFamily},
    distro::{MicrofacetDistribution, MicrofacetDistroKind, TrowbridgeReitzDistribution},
};

/// Microfacet BRDF model
pub type MicrofacetBrdfTR = MicrofacetBrdf<TrowbridgeReitzDistribution>;

impl MicrofacetBrdfTR {
    /// Creates a new microfacet BRDF model with the given roughness parameters.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        MicrofacetBrdf::from(TrowbridgeReitzDistribution::new(alpha_x, alpha_y))
    }
}

impl Bxdf for MicrofacetBrdfTR {
    type Params = <TrowbridgeReitzDistribution as MicrofacetDistribution>::Params;

    fn family(&self) -> BxdfFamily { BxdfFamily::Microfacet }

    fn distro(&self) -> Option<MicrofacetDistroKind> { Some(MicrofacetDistroKind::TrowbridgeReitz) }

    fn is_isotropic(&self) -> bool { self.distro.is_isotropic() }

    fn params(&self) -> Self::Params { self.distro.params() }

    fn set_params(&mut self, params: &Self::Params) { self.distro.set_params(params) }

    fn eval(&self, i: &Vec3, o: &Vec3) -> f64 {
        debug_assert!(i.is_normalized(), "Incident direction is not normalized.");
        debug_assert!(o.is_normalized(), "Outgoing direction is not normalized.");
        let cos_theta_i = cos_theta(i);
        let cos_theta_o = cos_theta(o);
        let cos_theta_io = (cos_theta_i * cos_theta_o) as f64;
        if cos_theta_io <= 1e-16 {
            return 0.0;
        }
        let h = (*i + *o).normalize();
        let wh = cart_to_sph(h);
        let d = self
            .distro
            .eval_ndf(wh.theta.as_f64().cos(), wh.phi.as_f64().cos());
        let g = self.distro.eval_msf1(h, *i) * self.distro.eval_msf1(h, *o);
        (d * g) / (4.0 * cos_theta_io)
    }

    fn eval_hd(&self, h: &Vec3, d: &Vec3) -> f64 { todo!() }

    fn evalp(&self, i: &Vec3, o: &Vec3) -> f64 { todo!() }

    fn evalp_hd(&self, h: &Vec3, d: &Vec3) -> f64 { todo!() }

    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> f64 { todo!() }

    fn sample(&self, u: f32, v: f32, o: &Vec3) -> f64 { todo!() }

    fn pdf(&self, i: &Vec3, o: &Vec3) -> f64 { todo!() }

    #[cfg(feature = "bxdf_fit")]
    fn pds(&self, i: &[Vec3], o: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]> {
        let mut result = Box::new_uninit_slice(i.len() * o.len() * 2);
        for j in 0..i.len() {
            for k in 0..o.len() {
                let pd = self.pd(&i[j], &o[k], ior_i, ior_t);
                result[j * o.len() * 2 + k * 2].write(pd[0]);
                result[j * o.len() * 2 + k * 2 + 1].write(pd[1]);
            }
        }
        unsafe { result.assume_init() }
    }

    #[cfg(feature = "bxdf_fit")]
    fn pd(&self, i: &Vec3, o: &Vec3, ior_i: &Ior, ior_t: &Ior) -> [f64; 2] {
        debug_assert!(i.is_normalized(), "Incident direction is not normalized");
        debug_assert!(o.is_normalized(), "Outgoing direction is not normalized");
        let [alpha_x, alpha_y] = self.distro.params();
        let h = (*i + *o).normalize();
        let cos_theta_h = cos_theta(&h);
        let cos_theta_h2 = sqr(cos_theta_h as f64);
        let cos_theta_h4 = sqr(cos_theta_h2);

        if cos_theta_h4 < 1.0e-6 {
            return [0.0, 0.0];
        }

        let tan_theta_h2 = (1.0 - cos_theta_h2) * rcp_f64(cos_theta_h2);
        if tan_theta_h2.is_infinite() {
            return [0.0, 0.0];
        }
        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;
        let cos_theta_i = cos_theta(&i);
        let cos_theta_o = cos_theta(&o);
        let alpha_x2 = sqr(alpha_x);
        let alpha_y2 = sqr(alpha_y);
        let alpha_x4 = sqr(alpha_x2);
        let alpha_y4 = sqr(alpha_y2);

        let phi_h = cart_to_sph(h).phi;
        let phi_o = cart_to_sph(*o).phi;
        let phi_i = cart_to_sph(*i).phi;
        let phi_hi = (phi_h - phi_i).abs().as_f64();
        let cos_phi_hi = phi_hi.cos();
        let sin_phi_hi = phi_hi.sin();
        let cos_theta_hi = i.dot(h) as f64;
        let tan_theta_hi2 =
            (1.0 - cos_theta_hi * cos_theta_hi) * rcp_f64(cos_theta_hi * cos_theta_hi);

        let ahi = {
            let squared =
                1.0 + alpha_x2 * cos_phi_hi * tan_theta_hi2 + alpha_y2 * sin_phi_hi * tan_theta_hi2;
            if squared < 0.0 {
                0.0
            } else {
                squared.sqrt()
            }
        };

        let phi_ho = (phi_h - phi_o).abs().as_f64();
        let cos_phi_ho = phi_ho.cos();
        let sin_phi_ho = phi_ho.sin();
        let cos_theta_ho = o.dot(h) as f64;
        let tan_theta_ho2 =
            (1.0 - cos_theta_ho * cos_theta_ho) * rcp_f64(cos_theta_ho * cos_theta_ho);

        let aho = {
            let squared =
                1.0 + alpha_x2 * cos_phi_ho * tan_theta_ho2 + alpha_y2 * sin_phi_ho * tan_theta_ho2;
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
                + (4.0 * cos_phi_h2 * tan_theta_h2 * rcp_f64(alpha_x4) - b * rcp_f64(alpha_x2))
                    * one_plus_ahi
                    * one_plus_aho);

        let coeff_dfr_dalpha_y = f * rcp_f64(common * alpha_x);
        let sin_phi_h2 = sqr(sin_phi_h);
        let dfr_dalpha_y = coeff_dfr_dalpha_y
            * (-sin_phi_ho * b * one_plus_ahi * tan_theta_ho2 * rcp_aho
                - sin_phi_hi * b * one_plus_aho * tan_theta_hi2 * rcp_ahi
                + (4.0 * sin_phi_h2 * tan_theta_h2 * rcp_f64(alpha_y4) - b * rcp_f64(alpha_y2))
                    * one_plus_ahi
                    * one_plus_aho);
        [dfr_dalpha_x, dfr_dalpha_y]
    }

    #[cfg(feature = "bxdf_fit")]
    fn pds_iso(&self, i: &[Vec3], o: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]> {
        debug_assert!(self.distro.is_isotropic());
        let mut result = Box::new_uninit_slice(i.len() * o.len());
        for j in 0..i.len() {
            let wi = i[j];
            for k in 0..o.len() {
                let wo = o[k];
                result[j * o.len() + k].write(self.pd_iso(&wi, &wo, ior_i, ior_t));
            }
        }
        unsafe { result.assume_init() }
    }

    #[cfg(feature = "bxdf_fit")]
    fn pd_iso(&self, i: &Vec3, o: &Vec3, ior_i: &Ior, ior_t: &Ior) -> f64 {
        debug_assert!(i.is_normalized(), "Incident direction is not normalized");
        debug_assert!(o.is_normalized(), "Outgoing direction is not normalized");
        let wh = (*i + *o).normalize();
        let cos_theta_h = cos_theta(&wh).abs();
        let cos_theta_h2 = sqr(cos_theta_h as f64);
        let cos_theta_h4 = sqr(cos_theta_h2);
        let cos_theta_i = i.z.abs();
        let cos_theta_o = o.z.abs();

        if cos_theta_h4 < 1e-16 || cos_theta_i.abs() < 1e-16 || cos_theta_o.abs() < 1e-16 {
            return 0.0;
        }

        let tan_theta_h2 = (1.0 - cos_theta_h2) * rcp_f64(cos_theta_h2);
        if tan_theta_h2.is_infinite() {
            return 0.0;
        }

        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;

        let alpha = self.params()[0];
        let alpha2 = sqr(alpha);

        let cos_theta_hi = i.dot(wh).abs() as f64;
        let cos_theta_hi2 = sqr(cos_theta_hi);
        let tan_theta_hi2 = (1.0 - cos_theta_hi2) * rcp_f64(cos_theta_hi2);
        let ai = (1.0 + alpha2 * tan_theta_hi2).sqrt();

        let cos_theta_ho = o.dot(wh).abs() as f64;
        let cos_theta_ho2 = sqr(cos_theta_ho);
        let tan_theta_ho2 = (1.0 - cos_theta_ho2) * rcp_f64(cos_theta_ho2);
        let ao = (1.0 + alpha2 * tan_theta_ho2).sqrt();

        let one_plus_ai = 1.0 + ai;
        let one_plus_ao = 1.0 + ao;
        let one_plus_ai2 = sqr(one_plus_ai);
        let one_plus_ao2 = sqr(one_plus_ao);

        let part_one = -f
            * rcp_f64(
                std::f64::consts::PI * cos_theta_h4 * cos_theta_i as f64 * cos_theta_o as f64,
            );

        let nominator =
            (alpha.powi(3) * one_plus_ai * (2.0 + 2.0 * ao + 3.0 * alpha2 * tan_theta_ho2))
                + (alpha.powi(5) * tan_theta_hi2 * (3.0 + 3.0 * ao + 4.0 * alpha2 * tan_theta_ho2))
                - (alpha
                    * tan_theta_h2
                    * (alpha2 * one_plus_ao * tan_theta_hi2
                        + one_plus_ai2 * (2.0 + 2.0 * ao + alpha2 * tan_theta_ho2)));
        let denominator = one_plus_ai2 * one_plus_ao2 * (alpha2 + tan_theta_h2).powi(3) * ai * ao;
        part_one * nominator * rcp_f64(denominator)
    }

    fn clone_box(&self) -> Box<dyn Bxdf<Params = Self::Params>> { Box::new(self.clone()) }
}

/// Trowbridge-Reitz (GGX) microfacet BRDF model.
/// See [Trowbridge-Reitz
/// Distribution](crate::dist::TrowbridgeReitzDistribution).
#[derive(Debug, Clone, Copy)]
pub struct TrowbridgeReitzBrdfModel {
    /// Roughness parameter of originated from microfacet distribution
    pub alpha_x: f64,
    /// Roughness parameter of originated from microfacet distribution
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

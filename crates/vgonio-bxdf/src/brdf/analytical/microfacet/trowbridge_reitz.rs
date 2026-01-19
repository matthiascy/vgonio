#[rustfmt::skip]
//! Trowbridge-Reitz (GGX) microfacet BRDF implementation.
//!
//! Normal distribution:
//!
//! $$
//! D(m) = \\frac{\\alpha_x \\alpha_y}{\\pi \\cos^4(\\theta_m) \\left( \\alpha_x^2 \\cos^2(\\phi_m) + \\alpha_y^2 \\sin^2(\\phi_m) \\right) \\left( 1 + \\tan^2(\\theta_m) \\right)^2}
//! $$
//!
//! with Smith masking-shadowing `G1` and half-vector sampling per Heitz (2014).

use std::{f64::consts::PI, fmt::Debug};
use vgn_core::math::{cart_to_sph, cos_theta, Vec3};

use crate::{
    brdf::{analytical::microfacet::MicrofacetBrdf, hd2io, AnalyticalBrdf},
    distro::{MicrofacetDistribution, MicrofacetDistroKind, TrowbridgeReitzDistribution},
    BrdfFamily,
};
#[cfg(feature = "fitting")]
use vgn_core::{
    math::{rcp_f64, sqr},
    optics::{fresnel, Ior},
};

#[rustfmt::skip]
/// Microfacet BRDF model using the Trowbridge-Reitz (GGX) normal distribution.
///
/// Evaluates
///
/// $$
/// f_r(i,o) = \\frac{D(h) * G(i,o,h)}{4 * cos(\\theta_i) * cos(\\theta_o)}
/// $$
///
/// use `Scattering::eval_reflectance` (or multiply by Fresnel manually) for
/// full energy conservation.
///
/// Based on Trowbridge and Reitz (1975) with Smith masking-shadowing (1967) and
/// sampling from Heitz (2014) / Walter et al.
pub type MicrofacetBrdfTR = MicrofacetBrdf<TrowbridgeReitzDistribution>;

impl MicrofacetBrdfTR {
    /// Creates a new microfacet BRDF model with the given roughness parameters.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        MicrofacetBrdf::from(TrowbridgeReitzDistribution::new(alpha_x, alpha_y))
    }
}

impl AnalyticalBrdf for MicrofacetBrdfTR {
    type Params = <TrowbridgeReitzDistribution as MicrofacetDistribution>::Params;

    fn family(&self) -> BrdfFamily { BrdfFamily::Microfacet }

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

    fn eval_hd(&self, h: &Vec3, d: &Vec3) -> f64 {
        let (i, o) = hd2io(h, d);
        self.eval(&i, &o)
    }

    fn evalp(&self, i: &Vec3, o: &Vec3) -> f64 { self.eval(i, o) * cos_theta(i) as f64 }

    fn evalp_hd(&self, h: &Vec3, d: &Vec3) -> f64 {
        let (i, o) = hd2io(h, d);
        self.evalp(&i, &o)
    }

    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> f64 {
        let wh = sample_half_ggx(self.distro.params(), u, v);
        let wo = *o;
        if wo.z <= 0.0 {
            *pdf = 0.0;
            return 0.0;
        }
        let wi = (2.0 * wo.dot(wh)) * wh - wo;
        if wi.z <= 0.0 {
            *pdf = 0.0;
            return 0.0;
        }
        *i = wi;
        *pdf = self.pdf(&wi, o) as f32;
        self.evalp(&wi, o)
    }

    fn sample(&self, u: f32, v: f32, o: &Vec3) -> f64 {
        let mut wi = Vec3::ZERO;
        let mut pdf = 0.0;
        self.evalp_is(u, v, o, &mut wi, &mut pdf)
    }

    fn pdf(&self, i: &Vec3, o: &Vec3) -> f64 {
        if i.z <= 0.0 || o.z <= 0.0 {
            return 0.0;
        }
        let wh = (*i + *o).normalize();
        let cos_theta_h = wh.z as f64;
        if cos_theta_h <= 0.0 {
            return 0.0;
        }
        let wh_sph = cart_to_sph(wh);
        let d = self
            .distro
            .eval_ndf(wh_sph.theta.as_f64().cos(), wh_sph.phi.as_f64().cos());
        let pdf_wh = d * cos_theta_h;
        let denom = 4.0 * o.dot(wh) as f64;
        if denom.abs() < 1.0e-9 {
            return 0.0;
        }
        pdf_wh / denom
    }

    #[cfg(feature = "fitting")]
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

    #[cfg(feature = "fitting")]
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

        let coeff_dfr_dalpha_x = rcp_f64(common * alpha_y);

        let cos_phi_h2 = sqr(cos_phi_h);
        let rcp_aho = if aho.abs() < 1.0e-6 { 0.0 } else { 1.0 / aho };
        let rcp_ahi = if ahi.abs() < 1.0e-6 { 0.0 } else { 1.0 / ahi };
        let dfr_dalpha_x = coeff_dfr_dalpha_x
            * (-cos_phi_ho * b * one_plus_ahi * tan_theta_ho2 * rcp_aho
                - cos_phi_hi * b * one_plus_aho * tan_theta_hi2 * rcp_ahi
                + (4.0 * cos_phi_h2 * tan_theta_h2 * rcp_f64(alpha_x4) - b * rcp_f64(alpha_x2))
                    * one_plus_ahi
                    * one_plus_aho);

        let coeff_dfr_dalpha_y = rcp_f64(common * alpha_x);
        let sin_phi_h2 = sqr(sin_phi_h);
        let dfr_dalpha_y = coeff_dfr_dalpha_y
            * (-sin_phi_ho * b * one_plus_ahi * tan_theta_ho2 * rcp_aho
                - sin_phi_hi * b * one_plus_aho * tan_theta_hi2 * rcp_ahi
                + (4.0 * sin_phi_h2 * tan_theta_h2 * rcp_f64(alpha_y4) - b * rcp_f64(alpha_y2))
                    * one_plus_ahi
                    * one_plus_aho);
        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;
        [f * dfr_dalpha_x, f * dfr_dalpha_y]
    }

    #[cfg(feature = "fitting")]
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

    #[cfg(feature = "fitting")]
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

        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;
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

    fn clone_box(&self) -> Box<dyn AnalyticalBrdf<Params = Self::Params>> { Box::new(self.clone()) }

    fn name(&self) -> &str { "Microfacet@TrowbridgeReitz" }
}

/// Samples a GGX half-vector by drawing isotropic slopes and stretching
/// them with `alpha_x/alpha_y`, following Heitz (2014).
fn sample_half_ggx([alpha_x, alpha_y]: [f64; 2], u1: f32, u2: f32) -> Vec3 {
    let u1 = u1.clamp(1.0e-6, 1.0 - 1.0e-6) as f64;
    let u2 = u2 as f64;
    let tan_theta2 = u1 / (1.0 - u1);
    let cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let slope_x = sin_theta * cos_phi / cos_theta;
    let slope_y = sin_theta * sin_phi / cos_theta;
    let mut wh = Vec3::new((alpha_x * slope_x) as f32, (alpha_y * slope_y) as f32, 1.0).normalize();
    if wh.z < 0.0 {
        wh = -wh;
    }
    wh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brdf::io2hd;
    use vgn_core::math::{cart_to_sph, cos_theta, Vec3};

    #[test]
    fn evalp_matches_eval_with_cos() {
        let brdf = MicrofacetBrdfTR::new(0.45, 0.45);
        let i = Vec3::new(0.0, 0.4, 0.916).normalize();
        let o = Vec3::new(0.2, 0.0, 0.98).normalize();
        let eval = brdf.eval(&i, &o);
        assert!((brdf.evalp(&i, &o) - eval * i.z as f64).abs() < 1.0e-6);
    }

    #[test]
    fn eval_hd_matches_eval() {
        let brdf = MicrofacetBrdfTR::new(0.25, 0.25);
        let i = Vec3::new(0.35, 0.15, 0.92).normalize();
        let o = Vec3::new(-0.15, 0.25, 0.95).normalize();
        let (h, d) = io2hd(&i, &o);
        assert!((brdf.eval_hd(&h, &d) - brdf.eval(&i, &o)).abs() < 1.0e-6);
        assert!((brdf.evalp_hd(&h, &d) - brdf.evalp(&i, &o)).abs() < 1.0e-6);
    }

    #[test]
    fn sampling_returns_valid_pdf() {
        let brdf = MicrofacetBrdfTR::new(0.5, 0.5);
        let o = Vec3::new(0.0, 0.3, 0.954).normalize();
        let mut wi = Vec3::ZERO;
        let mut pdf = 0.0;
        let value = brdf.evalp_is(0.21, 0.73, &o, &mut wi, &mut pdf);
        assert!(wi.z > 0.0);
        assert!(pdf > 0.0);
        assert!((wi.length_squared() - 1.0).abs() < 1.0e-4);
        assert!((value - brdf.evalp(&wi, &o)).abs() < 1.0e-6);
        let pdf_direct = brdf.pdf(&wi, &o);
        assert!((pdf_direct - pdf as f64).abs() < 1.0e-6);
    }

    #[test]
    fn eval_matches_manual_without_fresnel() {
        let brdf = MicrofacetBrdfTR::new(0.35, 0.55);
        let i = Vec3::new(0.25, -0.35, 0.9).normalize();
        let o = Vec3::new(-0.18, 0.32, 0.93).normalize();
        let h = (i + o).normalize();
        let wh = cart_to_sph(h);
        let d = brdf
            .distro
            .eval_ndf(wh.theta.as_f64().cos(), wh.phi.as_f64().cos());
        let g = brdf.distro.eval_msf1(h, i) * brdf.distro.eval_msf1(h, o);
        let expected = d * g / (4.0 * cos_theta(&i) as f64 * cos_theta(&o) as f64);
        let eval = brdf.eval(&i, &o);
        assert!((eval - expected).abs() < 1.0e-8);
    }

    #[test]
    fn eval_is_symmetric() {
        let brdf = MicrofacetBrdfTR::new(0.28, 0.63);
        let i = Vec3::new(0.22, 0.15, 0.962).normalize();
        let o = Vec3::new(-0.31, 0.05, 0.95).normalize();
        let f_io = brdf.eval(&i, &o);
        let f_oi = brdf.eval(&o, &i);
        assert!((f_io - f_oi).abs() < 1.0e-10);
    }

    #[test]
    fn anisotropic_sampling_pdf_consistent() {
        let brdf = MicrofacetBrdfTR::new(0.22, 0.67);
        let o = Vec3::new(0.12, -0.18, 0.975).normalize();
        let mut wi = Vec3::ZERO;
        let mut pdf = 0.0;
        let value = brdf.evalp_is(0.19, 0.81, &o, &mut wi, &mut pdf);
        assert!(wi.z > 0.0);
        assert!(pdf > 0.0);
        assert!((wi.length_squared() - 1.0).abs() < 1.0e-4);
        assert!((value - brdf.evalp(&wi, &o)).abs() < 1.0e-6);
        let pdf_direct = brdf.pdf(&wi, &o);
        assert!((pdf_direct - pdf as f64).abs() < 1.0e-6);
    }

    #[test]
    fn returns_zero_when_below_horizon() {
        let brdf = MicrofacetBrdfTR::new(0.3, 0.4);
        let o = Vec3::new(0.2, 0.1, 0.97).normalize();
        let i = Vec3::new(0.1, -0.2, -0.95).normalize();
        assert_eq!(0.0, brdf.pdf(&i, &o));
        assert_eq!(0.0, brdf.eval(&i, &o));
    }
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
    /// Creates a new Trowbridge-Reitz (GGX) microfacet BRDF model.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        TrowbridgeReitzBrdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

#[rustfmt::skip]
//! Beckmann microfacet BRDF implementation.
//!
//! Uses the Gaussian slope-based NDF
//!
//! $$
//! D(m) = \\frac{e^{-\\tan^2(\\theta_m) \\left(\\frac{\\cos^2(\\phi)}{\\alpha_x^2} + \\frac{\\sin^2(\\phi)}{\\alpha_y^2}\\right)}}{\\pi \\alpha_x \\alpha_y \\cos^4(\\theta_m)}
//! $$
//!
//! with Smith masking-shadowing and Heitz (2014) slope sampling.

use crate::{
    brdf::{analytical::microfacet::MicrofacetBrdf, hd2io, AnalyticalBrdf},
    distro::{BeckmannDistribution, MicrofacetDistribution, MicrofacetDistroKind},
    BrdfFamily,
};
#[cfg(feature = "fitting")]
use libm::erf;
use std::f64::consts::PI;
use vgn_core::math::{cart_to_sph, cos_theta, Vec3};
#[cfg(feature = "fitting")]
use vgn_core::{
    math::{cbr, rcp_f64, sqr},
    optics::{fresnel, Ior},
};

#[rustfmt::skip]
/// Microfacet BRDF model based on Beckmann distribution.
///
/// Evaluates
///
/// $$
/// f_r(i,o) = \\frac{D(h) * G(i,o,h)}{4 * cos(\\theta_i) * cos(\\theta_o)}
/// $$
///
/// without the Fresnel factor; callers can apply `F` separately. Uses the
/// Beckmann slope distribution from Beckmann and Spizzichino (1963),
/// Smith masking (1967), and sampling guidance from Heitz (2014).
pub type MicrofacetBrdfBK = MicrofacetBrdf<BeckmannDistribution>;

impl MicrofacetBrdfBK {
    /// Creates a new Beckmann microfacet BRDF model.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        MicrofacetBrdf::from(BeckmannDistribution::new(alpha_x, alpha_y))
    }
}

impl AnalyticalBrdf for MicrofacetBrdfBK {
    type Params = <BeckmannDistribution as MicrofacetDistribution>::Params;

    fn name(&self) -> &str { "Microfacet@Beckmann" }

    fn family(&self) -> BrdfFamily { BrdfFamily::Microfacet }

    fn distro(&self) -> Option<MicrofacetDistroKind> { Some(MicrofacetDistroKind::Beckmann) }

    fn is_isotropic(&self) -> bool { self.distro.is_isotropic() }

    fn params(&self) -> Self::Params { self.distro.params() }

    fn set_params(&mut self, params: &Self::Params) { self.distro.set_params(params); }

    fn eval(&self, i: &Vec3, o: &Vec3) -> f64 {
        debug_assert!(i.is_normalized(), "Incident direction is not normalized");
        debug_assert!(o.is_normalized(), "Outgoing direction is not normalized");
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
        let wh = sample_half_beckmann(self.distro.params(), u, v);
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
        // TODO: test medium type
        for j in 0..i.len() {
            for k in 0..o.len() {
                let pd = self.pd(&i[j], &o[k], ior_i, ior_t);
                result[j * 2 * o.len() + k * 2].write(pd[0]);
                result[j * 2 * o.len() + k * 2 + 1].write(pd[1]);
            }
        }
        unsafe { result.assume_init() }
    }

    #[cfg(feature = "fitting")]
    fn pd(&self, i: &Vec3, o: &Vec3, ior_i: &Ior, ior_t: &Ior) -> [f64; 2] {
        debug_assert!(i.is_normalized(), "incident direction is not normalized");
        debug_assert!(o.is_normalized(), "outgoing direction is not normalized");
        let [alpha_x, alpha_y] = self.distro.params();
        let h = (*i + *o).normalize();
        let alpha_x2 = sqr(alpha_x);
        let alpha_y2 = sqr(alpha_y);
        let alpha_x4 = sqr(alpha_x2);
        let alpha_y4 = sqr(alpha_y2);
        let rcp_alpha_x2 = rcp_f64(alpha_x2);
        let rcp_alpha_y2 = rcp_f64(alpha_y2);
        let rcp_alpha_x4 = rcp_f64(alpha_x4);
        let rcp_alpha_y4 = rcp_f64(alpha_y4);
        let cos_theta_i = i.z as f64;
        let sec_theta_i = rcp_f64(cos_theta_i);
        let cos_theta_o = o.z as f64;
        let sec_theta_o = rcp_f64(cos_theta_o);
        let cos_theta_h = h.z as f64;
        let cos_theta_h2 = cos_theta_h * cos_theta_h;
        let sec_theta_h4 = rcp_f64(cos_theta_h2 * cos_theta_h2);
        let sin_theta_h2 = 1.0 - cos_theta_h2;
        let tan_theta_h2 = (1.0 - cos_theta_h2) / cos_theta_h2;

        // Compute azimuthal angle phi_h first
        let phi_h = h.y.atan2(h.x) as f64;
        let cos_phi_h = phi_h.cos();
        let cos_phi_h2 = sqr(cos_phi_h);
        let sin_phi_h2 = 1.0 - cos_phi_h2;
        let c =
            (-tan_theta_h2 * cos_phi_h2 * rcp_f64(alpha_x2) + sin_phi_h2 * rcp_f64(alpha_y2)).exp();
        let phi_i = i.y.atan2(i.x) as f64;
        let phi_o = o.y.atan2(o.x) as f64;
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
        // erf(±inf) -> ±1 and exp(-inf) -> 0, so infinities are naturally handled.
        // Only guard against NaN from sqrt of negative.
        if bi.is_nan() || bo.is_nan() {
            return [0.0, 0.0];
        }
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

        let nominator_x = c
            * sec_theta_h4
            * sec_theta_i
            * sec_theta_o
            * (-(-bo2).exp() * cos_phi_ho2 * gi * tan_theta_ho * rcp_ao
                - (-bi2).exp() * cos_phi_hi2 * go * tan_theta_hi * rcp_ai
                + (sqrt_pi * 2.0 * sin_theta_h2 * rcp_alpha_x4 - sqrt_pi * rcp_alpha_x2) * gi * go);
        // let denominator_x = sqrt_pi * sqrt_pi * sqrt_pi * alpha_y * gi2 * go2;

        let nominator_y = c
            * sec_theta_h4
            * sec_theta_i
            * sec_theta_o
            * (-(-bo2).exp() * sin_phi_ho2 * gi * tan_theta_ho * rcp_ao
                - (-bi2).exp() * sin_phi_hi2 * go * tan_theta_hi * rcp_ai
                - (sqrt_pi * gi * go * rcp_alpha_y2)
                + (sqrt_pi * 2.0 * sin_theta_h2 * tan_theta_h2 * gi * go * rcp_alpha_y4));
        // let denominator_y = sqrt_pi * sqrt_pi * sqrt_pi * alpha_x * gi2 * go2;

        let rcp_gi = rcp_f64(gi);
        let rcp_go = rcp_f64(go);
        let dfr_dalpha_x = if nominator_x == 0.0 {
            0.0
        } else {
            nominator_x
                * rcp_gi
                * rcp_gi
                * rcp_go
                * rcp_go
                * rcp_f64(sqrt_pi * sqrt_pi * sqrt_pi * alpha_y)
        };
        let dfr_dalpha_y = if nominator_y == 0.0 {
            0.0
        } else {
            nominator_y
                * rcp_gi
                * rcp_gi
                * rcp_go
                * rcp_go
                * rcp_f64(sqrt_pi * sqrt_pi * sqrt_pi * alpha_x)
        };

        // Handle NaN/Inf gracefully instead of panicking
        let dfr_dalpha_x = if !dfr_dalpha_x.is_finite() {
            0.0
        } else {
            dfr_dalpha_x
        };
        let dfr_dalpha_y = if !dfr_dalpha_y.is_finite() {
            0.0
        } else {
            dfr_dalpha_y
        };

        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;
        [f * dfr_dalpha_x, f * dfr_dalpha_y]
    }

    #[cfg(feature = "fitting")]
    fn pds_iso(&self, vi: &[Vec3], vo: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]> {
        debug_assert!(self.distro.is_isotropic());
        let mut result = Box::new_uninit_slice(vi.len() * vo.len());
        for j in 0..vi.len() {
            for k in 0..vo.len() {
                result[j * vo.len() + k].write(self.pd_iso(&vi[j], &vo[k], ior_i, ior_t));
            }
        }
        unsafe { result.assume_init() }
    }

    #[cfg(feature = "fitting")]
    fn pd_iso(&self, i: &Vec3, o: &Vec3, ior_i: &Ior, ior_t: &Ior) -> f64 {
        debug_assert!(i.is_normalized(), "incident direction is not normalized");
        debug_assert!(o.is_normalized(), "outgoing direction is not normalized");
        let h = (*i + *o).normalize();
        let cos_theta_h = cos_theta(&h).abs();
        let cos_theta_h2 = sqr(cos_theta_h as f64);
        let cos_theta_h4 = sqr(cos_theta_h2);
        let cos_theta_i = cos_theta(&i).abs();
        let cos_theta_o = cos_theta(&o).abs();
        if cos_theta_h4 < 1e-16 || cos_theta_i < 1e-16 || cos_theta_o < 1e-16 {
            return 0.0;
        }
        let cos_theta_h4_i_o = cos_theta_h4 * cos_theta_i as f64 * cos_theta_o as f64;
        let tan_theta_h2 = (1.0 - cos_theta_h2).max(0.0) / cos_theta_h2;
        if tan_theta_h2 < 1e-16 {
            return 0.0;
        }
        let alpha = self.params()[0];
        let alpha2 = sqr(alpha);
        let alpha3 = alpha2 * alpha;
        let alpha5 = alpha2 * alpha3;

        let tan_theta_h2 = (1.0 - cos_theta_h2) / cos_theta_h2;
        let cos_theta_hi = i.dot(h).abs() as f64;
        let cos_theta_ho = o.dot(h).abs() as f64;
        let tan_theta_hi = (1.0 - cos_theta_hi * cos_theta_hi).max(0.0).sqrt() / cos_theta_hi;
        let tan_theta_ho = (1.0 - cos_theta_ho * cos_theta_ho).max(0.0).sqrt() / cos_theta_ho;

        let sqrt_pi = std::f64::consts::PI.sqrt();
        let alpha_over_sqrt_pi = alpha * rcp_f64(sqrt_pi);
        let ahi = rcp_f64(alpha * tan_theta_hi);
        let aho = rcp_f64(alpha * tan_theta_ho);
        let ehi = tan_theta_hi * (-sqr(ahi)).exp();
        let eho = tan_theta_ho * (-sqr(aho)).exp();
        let bhi = 1.0 + erf(ahi) + alpha_over_sqrt_pi * ehi;
        let bho = 1.0 + erf(aho) + alpha_over_sqrt_pi * eho;

        let nominator_part1 = 2.0 * sqrt_pi * bhi * bho * (tan_theta_h2 - alpha2);
        let nominator_part2 = alpha3 * (eho * bhi + ehi * bho);
        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;
        let nominator = f * (nominator_part1 - nominator_part2) * (-tan_theta_h2 / alpha2).exp();
        let denominator = cbr(sqrt_pi) * alpha5 * sqr(bhi) * sqr(bho) * cos_theta_h4_i_o;
        nominator * rcp_f64(denominator)
    }

    fn clone_box(&self) -> Box<dyn AnalyticalBrdf<Params = Self::Params>> { Box::new(self.clone()) }
}

/// Samples a Beckmann half-vector by drawing isotropic slopes and stretching
/// them with `alpha_x/alpha_y`, following the slope remapping of Heitz (2014).
fn sample_half_beckmann([alpha_x, alpha_y]: [f64; 2], u1: f32, u2: f32) -> Vec3 {
    let u1 = u1.clamp(1.0e-6, 1.0 - 1.0e-6) as f64;
    let u2 = u2 as f64;
    let tan_theta2 = -(1.0_f64) * (1.0 - u1).ln();
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
        let brdf = MicrofacetBrdfBK::new(0.3, 0.3);
        let i = Vec3::new(0.0, 0.0, 1.0);
        let o = Vec3::new(0.1, 0.0, 0.99).normalize();
        let eval = brdf.eval(&i, &o);
        assert!((brdf.evalp(&i, &o) - eval * i.z as f64).abs() < 1.0e-6);
    }

    #[test]
    fn eval_hd_matches_eval() {
        let brdf = MicrofacetBrdfBK::new(0.4, 0.4);
        let i = Vec3::new(0.3, 0.2, 0.93).normalize();
        let o = Vec3::new(-0.1, 0.4, 0.91).normalize();
        let (h, d) = io2hd(&i, &o);
        assert!((brdf.eval_hd(&h, &d) - brdf.eval(&i, &o)).abs() < 1.0e-6);
        assert!((brdf.evalp_hd(&h, &d) - brdf.evalp(&i, &o)).abs() < 1.0e-6);
    }

    #[test]
    fn sampling_returns_valid_pdf() {
        let brdf = MicrofacetBrdfBK::new(0.35, 0.35);
        let o = Vec3::new(0.2, 0.1, 0.97).normalize();
        let mut wi = Vec3::ZERO;
        let mut pdf = 0.0;
        let value = brdf.evalp_is(0.37, 0.58, &o, &mut wi, &mut pdf);
        assert!(wi.z > 0.0);
        assert!(pdf > 0.0);
        assert!((wi.length_squared() - 1.0).abs() < 1.0e-4);
        assert!((value - brdf.evalp(&wi, &o)).abs() < 1.0e-6);
        let pdf_direct = brdf.pdf(&wi, &o);
        assert!((pdf_direct - pdf as f64).abs() < 1.0e-6);
    }

    #[test]
    fn eval_matches_manual_without_fresnel() {
        let brdf = MicrofacetBrdfBK::new(0.28, 0.5);
        let i = Vec3::new(0.15, -0.25, 0.955).normalize();
        let o = Vec3::new(-0.18, 0.35, 0.915).normalize();
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
        let brdf = MicrofacetBrdfBK::new(0.33, 0.44);
        let i = Vec3::new(0.22, 0.14, 0.965).normalize();
        let o = Vec3::new(-0.27, -0.06, 0.96).normalize();
        let f_io = brdf.eval(&i, &o);
        let f_oi = brdf.eval(&o, &i);
        assert!((f_io - f_oi).abs() < 1.0e-10);
    }

    #[test]
    fn anisotropic_sampling_pdf_consistent() {
        let brdf = MicrofacetBrdfBK::new(0.22, 0.61);
        let o = Vec3::new(0.11, -0.2, 0.972).normalize();
        let mut wi = Vec3::ZERO;
        let mut pdf = 0.0;
        let value = brdf.evalp_is(0.42, 0.63, &o, &mut wi, &mut pdf);
        assert!(wi.z > 0.0);
        assert!(pdf > 0.0);
        assert!((wi.length_squared() - 1.0).abs() < 1.0e-4);
        assert!((value - brdf.evalp(&wi, &o)).abs() < 1.0e-6);
        let pdf_direct = brdf.pdf(&wi, &o);
        assert!((pdf_direct - pdf as f64).abs() < 1.0e-6);
    }

    #[test]
    fn returns_zero_when_below_horizon() {
        let brdf = MicrofacetBrdfBK::new(0.25, 0.4);
        let o = Vec3::new(0.2, 0.1, 0.97).normalize();
        let i = Vec3::new(0.1, -0.2, -0.95).normalize();
        assert_eq!(0.0, brdf.pdf(&i, &o));
        assert_eq!(0.0, brdf.eval(&i, &o));
    }
}

use crate::{
    brdf::{analytical::microfacet::MicrofacetBrdf, Bxdf, BxdfFamily},
    distro::{BeckmannDistribution, MicrofacetDistribution, MicrofacetDistroKind},
};
use base::math::{cart_to_sph, cos_theta, Vec3};
#[cfg(feature = "fitting")]
use base::{
    math::{cbr, rcp_f64, sqr},
    optics::{fresnel, ior::Ior},
};
#[cfg(feature = "fitting")]
use libm::erf;

/// Microfacet BRDF model based on Beckmann distribution.
pub type MicrofacetBrdfBK = MicrofacetBrdf<BeckmannDistribution>;

impl MicrofacetBrdfBK {
    /// Creates a new Beckmann microfacet BRDF model.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        MicrofacetBrdf::from(BeckmannDistribution::new(alpha_x, alpha_y))
    }
}

impl Bxdf for MicrofacetBrdfBK {
    type Params = <BeckmannDistribution as MicrofacetDistribution>::Params;

    fn family(&self) -> BxdfFamily { BxdfFamily::Microfacet }

    fn distro(&self) -> Option<MicrofacetDistroKind> { Some(MicrofacetDistroKind::Beckmann) }

    fn isotropic(&self) -> bool { self.distro.is_isotropic() }

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

    fn eval_hd(&self, h: &Vec3, d: &Vec3) -> f64 { todo!() }

    fn evalp(&self, i: &Vec3, o: &Vec3) -> f64 { todo!() }

    fn evalp_hd(&self, h: &Vec3, d: &Vec3) -> f64 { todo!() }

    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> f64 { todo!() }

    fn sample(&self, u: f32, v: f32, o: &Vec3) -> f64 { todo!() }

    fn pdf(&self, i: &Vec3, o: &Vec3) -> f64 { todo!() }

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
        let cos_phi_h = h.y.atan2(h.x);
        let cos_phi_h2 = (cos_phi_h * cos_phi_h) as f64;
        let sin_phi_h2 = 1.0 - cos_phi_h2;
        let c =
            (-tan_theta_h2 * cos_phi_h2 * rcp_f64(alpha_x2) + sin_phi_h2 * rcp_f64(alpha_y2)).exp();

        let phi_h = h.y.atan2(h.x) as f64;
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

        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;

        let nominator_x = f
            * c
            * sec_theta_h4
            * sec_theta_i
            * sec_theta_o
            * (-(-bo2).exp() * cos_phi_ho2 * gi * tan_theta_ho * rcp_ao
                - (-bi2).exp() * cos_phi_hi2 * go * tan_theta_hi * rcp_ai
                + (sqrt_pi * 2.0 * sin_theta_h2 * rcp_alpha_x4 - sqrt_pi * rcp_alpha_x2) * gi * go);
        // let denominator_x = sqrt_pi * sqrt_pi * sqrt_pi * alpha_y * gi2 * go2;

        let nominator_y = f
            * c
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

        // let mut dfr_dalpha_x = nominator_x * rcp_f64(denominator_x);
        // let mut dfr_dalpha_y = nominator_y * rcp_f64(denominator_y);
        // if dfr_dalpha_x.is_nan() || dfr_dalpha_x.is_infinite() {
        //     dfr_dalpha_x = 0.0;
        // }
        // if dfr_dalpha_y.is_nan() || dfr_dalpha_y.is_infinite() {
        //     dfr_dalpha_y = 0.0;
        // }
        if dfr_dalpha_x.is_infinite() || dfr_dalpha_x.is_nan() {
            println!(
                "nominator_x: {}, gi: {}, go: {}, alpha_x: {}",
                nominator_x, gi, go, alpha_x
            );
        }
        if dfr_dalpha_y.is_infinite() || dfr_dalpha_y.is_nan() {
            println!(
                "nominator_y: {}, gi: {}, go: {}, alpha_y: {}",
                nominator_y, gi, go, alpha_y
            );
        }
        assert!(!nominator_x.is_nan(), "nominator_x is NaN");
        assert!(nominator_x.is_finite(), "nominator_x is infinite");
        assert!(!nominator_y.is_nan(), "nominator_y is NaN");
        assert!(nominator_y.is_finite(), "nominator_y is infinite");
        assert!(!alpha_x.is_nan(), "alpha_x is NaN");
        assert!(!alpha_x.is_infinite(), "alpha_x is infinite");
        assert!(!alpha_y.is_nan(), "alpha_y is NaN");
        assert!(!alpha_y.is_infinite(), "alpha_y is infinite");
        assert!(!gi.is_nan(), "gi is NaN");
        assert!(!gi.is_infinite(), "gi is infinite");
        assert!(!go.is_nan(), "go is NaN");
        assert!(!go.is_infinite(), "go is infinite");
        assert!(!gi2.is_nan(), "gi2 is NaN");
        assert!(!gi2.is_infinite(), "gi2 is infinite");
        assert!(!go2.is_nan(), "go2 is NaN");
        assert!(!go2.is_infinite(), "go2 is infinite");
        assert!(!dfr_dalpha_x.is_nan(), "dfr_dalpha_x is NaN");
        assert!(!dfr_dalpha_x.is_infinite(), "dfr_dalpha_x is infinite");
        assert!(!dfr_dalpha_y.is_nan(), "dfr_dalpha_y is NaN");
        assert!(!dfr_dalpha_y.is_infinite(), "dfr_dalpha_y is infinite");

        [dfr_dalpha_x, dfr_dalpha_y]
    }

    #[cfg(feature = "fitting")]
    fn pds_iso(&self, i: &[Vec3], o: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]> {
        debug_assert!(self.distro.is_isotropic());
        let mut result = Box::new_uninit_slice(i.len() * o.len());
        for j in 0..i.len() {
            for k in 0..o.len() {
                result[j * o.len() + k].write(self.pd_iso(&i[j], &o[k], ior_i, ior_t));
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
        let f = fresnel::reflectance(cos_theta(&-*i), ior_i, ior_t) as f64;
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
        let nominator = f * (nominator_part1 - nominator_part2) * (-tan_theta_h2 / alpha2).exp();
        let denominator = cbr(sqrt_pi) * alpha5 * sqr(bhi) * sqr(bho) * cos_theta_h4_i_o;
        nominator * rcp_f64(denominator)
    }
}

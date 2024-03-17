use crate::distro::{MicrofacetDistribution, MicrofacetDistroKind};
use base::{
    math::{cbr, rcp_f64, sqr, Vec3},
    Isotropy,
};
use std::fmt::Debug;

/// Trowbridge-Reitz(GGX) microfacet distribution function.
///
/// $$ D(\mathbf{m}) = \frac{\alpha^2}{\pi \cos^4 \theta_m (\alpha^2 + \tan^2
/// \theta_m)^2} $$
///
/// where $\alpha$ is the width parameter of the NDF, $\theta_m$ is the angle
/// between the microfacet normal and the normal of the surface.
///
/// In case of anisotropic distribution, the NDF is defined as
///
/// $$ D(\mathbf{m}) = \frac{\alpha_x \alpha_y}{\pi \cos^4 \theta_m (\alpha_x^2
/// \cos^2 \phi_m + \alpha_y^2 \sin^2 \phi_m) (\alpha^2 + \tan^2 \theta_m)^2} $$
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzDistribution {
    /// Parameter of the microfacet area distribution function along the
    /// horizontal axis.
    pub alpha_x: f64,
    /// Parameter of the microfacet area distribution function along the
    /// vertical axis.
    pub alpha_y: f64,
}

impl TrowbridgeReitzDistribution {
    /// Creates a new Trowbridge-Reitz distribution with the given roughness.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        TrowbridgeReitzDistribution {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }

    /// Creates a new isotropic Trowbridge-Reitz distribution with the given
    /// roughness.
    pub fn new_isotropic(alpha: f64) -> Self {
        TrowbridgeReitzDistribution {
            alpha_x: alpha.max(1.0e-6),
            alpha_y: alpha.max(1.0e-6),
        }
    }

    /// Returns whether the distribution is isotropic or anisotropic.
    pub fn is_isotropic(&self) -> bool { (self.alpha_x - self.alpha_y).abs() < 1.0e-6 }
}

impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn params(&self) -> Self::Params { [self.alpha_x, self.alpha_y] }

    fn set_params(&mut self, params: &Self::Params) {
        self.alpha_x = params[0];
        self.alpha_y = params[1];
    }

    fn kind(&self) -> MicrofacetDistroKind { MicrofacetDistroKind::TrowbridgeReitz }

    fn isotropy(&self) -> Isotropy {
        if self.is_isotropic() {
            Isotropy::Isotropic
        } else {
            Isotropy::Anisotropic
        }
    }

    fn eval_ndf(&self, cos_theta: f64, cos_phi: f64) -> f64 {
        let cos_theta2 = sqr(cos_theta);
        let tan_theta2 = (1.0 - cos_theta2) / cos_theta2;
        if tan_theta2.is_infinite() {
            return 0.0;
        }
        let cos_theta4 = sqr(cos_theta2);
        let alpha_xy = self.alpha_x * self.alpha_y;
        let cos_phi2 = sqr(cos_phi);
        let sin_phi2 = 1.0 - cos_phi2;
        let e = tan_theta2 * (cos_phi2 / sqr(self.alpha_x) + sin_phi2 / sqr(self.alpha_y));
        rcp_f64(alpha_xy * std::f64::consts::PI * cos_theta4 * sqr(1.0 + e))
    }

    fn eval_msf1(&self, m: &Vec3, v: &Vec3) -> f64 {
        // TODO: recheck the implementation, especially the anisotropic case
        if m.dot(*v) <= 0.0 {
            return 0.0;
        }
        let cos_theta_v2 = sqr(v.z as f64);
        if cos_theta_v2 < 1.0e-16 {
            return 0.0;
        }
        let tan_theta_v2 = (1.0 - cos_theta_v2) * rcp_f64(cos_theta_v2);
        // // aniostropic case
        // let (_, _, phi_m) = cartesian_to_spherical(m, 1.0);
        // let cos_phi_m2 = sqr(phi_m.cos()) as f64;
        // let sin_phi_m2 = 1.0 - cos_phi_m2;
        // let alpha2 = sqr(self.alpha_x) * cos_phi_m2 + sqr(self.alpha_y) * sin_phi_m2;

        // isotropic case
        let alpha2 = sqr(self.alpha_x);

        2.0 * rcp_f64(1.0 + (1.0 + tan_theta_v2 * alpha2).sqrt())
    }

    fn clone_box(&self) -> Box<dyn MicrofacetDistribution<Params = Self::Params>> {
        Box::new(*self)
    }

    #[cfg(feature = "fitting")]
    fn pd_ndf(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Box<[f64]> {
        debug_assert!(
            cos_thetas.len() == cos_phis.len(),
            "The number of cos_thetas and cos_phis must be the same."
        );
        let alpha_x2 = sqr(self.alpha_x);
        let alpha_y2 = sqr(self.alpha_y);
        let alpha_x2_alpha_y2 = alpha_x2 * alpha_y2;
        let alpha_x2_alpha_y3 = alpha_x2 * cbr(self.alpha_y);
        let alpha_x3_alpha_y2 = cbr(self.alpha_x) * alpha_y2;
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
                let rcp_alpha_denom = rcp_f64(
                    std::f64::consts::PI
                        * cbr(alpha_x2_alpha_y2
                            + tan_theta2 * (alpha_x2 * sin_phi2 + alpha_y2 * cos_phi2)),
                );
                let d_alpha_x = {
                    let numerator = -alpha_x2_alpha_y3
                        * sec_theta4
                        * (alpha_x2_alpha_y2
                            + tan_theta2 * (alpha_x2 * sin_phi2 - 3.0 * alpha_y2 * cos_phi2));
                    numerator * rcp_alpha_denom
                };
                let d_alpha_y = {
                    let numerator = -alpha_x3_alpha_y2
                        * sec_theta4
                        * (alpha_x2_alpha_y2
                            + tan_theta2 * (alpha_y2 * cos_phi2 - 3.0 * alpha_x2 * sin_phi2));
                    numerator * rcp_alpha_denom
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    #[cfg(feature = "fitting")]
    fn pd_ndf_iso(&self, cos_thetas: &[f64]) -> Box<[f64]> {
        let mut results = Box::new_uninit_slice(cos_thetas.len());
        let (alpha, alpha2, alpha3) = (self.alpha_x, sqr(self.alpha_x), cbr(self.alpha_x));
        for (pd, cos_theta) in results.iter_mut().zip(cos_thetas.iter()) {
            if cos_theta.abs() < 1.0e-6 {
                pd.write(0.0);
                continue;
            }
            let cos_theta2 = sqr(*cos_theta);
            let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
            let sec_theta4 = rcp_f64(sqr(cos_theta2));
            let numerator = -2.0 * (alpha3 - alpha * tan_theta2) * sec_theta4;
            let denominator = std::f64::consts::PI * cbr(alpha2 + tan_theta2);
            pd.write(numerator * rcp_f64(denominator));
        }
        unsafe { results.assume_init() }
    }

    #[cfg(feature = "fitting")]
    fn pd_msf(&self, _m: Vec3, i: Vec3, o: Vec3) -> f64 {
        let cos_theta_i2 = sqr(i.y as f64);
        let tan_theta_i2 = (1.0 - cos_theta_i2) * rcp_f64(cos_theta_i2);
        let cos_theta_o2 = sqr(o.y as f64);
        let tan_theta_o2 = (1.0 - cos_theta_o2) * rcp_f64(cos_theta_o2);
        let denominator = sqr(1.0 + (1.0 + tan_theta_i2 * sqr(self.alpha_x)).sqrt())
            * sqr(1.0 + (1.0 + tan_theta_o2 * sqr(self.alpha_x)).sqrt());
        let numerator = -4.0
            * self.alpha_x
            * ((1.0 + (1.0 + tan_theta_i2 * sqr(self.alpha_x)).sqrt()) * tan_theta_o2
                / sqr(1.0 + (1.0 + sqr(self.alpha_x) * tan_theta_o2).sqrt())
                + (1.0 + (1.0 + tan_theta_o2 * sqr(self.alpha_x)).sqrt()) * tan_theta_i2
                    / sqr(1.0 + (1.0 + sqr(self.alpha_x) * tan_theta_i2).sqrt()));
        numerator * rcp_f64(denominator)
    }
}

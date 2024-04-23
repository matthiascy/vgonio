use crate::distro::{MicrofacetDistribution, MicrofacetDistroKind};
use base::{
    math::{cbr, cos_phi, cos_theta2, rcp_f64, sin_phi, sqr, tan_theta2, Vec3},
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

    #[rustfmt::skip]
    /// Under the uncorrelated height assumption, the lambda function for the
    /// Trowbridge-Reitz distribution has the analytical form:
    ///
    /// $$\Lambda(\mathbf{\omega})=\frac{\sqrt{1+\alpha^2\tan^2\theta}-1}{2}$$
    ///
    /// where $\alpha$ is the roughness parameter of the distribution in case of
    /// isotropic distribution. In case of anisotropic distribution, the $\alpha$
    /// parameter is replaced by $\sqrt{\alpha_x^2\cos^2\phi +
    /// \alpha_y^2\sin^2\phi}$.
    fn eval_lambda(&self, w: Vec3) -> f64 {
        let cos_theta2 = cos_theta2(&w) as f64;
        let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
        if tan_theta2.is_infinite() {
            return f64::INFINITY;
        }
        let alpha2 = if self.is_isotropic() {
            sqr(self.alpha_x)
        } else {
            let cos_phi2 = sqr(cos_phi(&w)) as f64;
            let sin_phi2 = 1.0 - cos_phi2;
            sqr(self.alpha_x) * cos_phi2 + sqr(self.alpha_y) * sin_phi2
        };
        0.5 * ((1.0 + tan_theta2 * alpha2).sqrt() - 1.0)
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
    fn pd_msf1(&self, wms: &[Vec3], ws: &[Vec3]) -> Box<[f64]> {
        let (count, idx_mul) = if self.is_isotropic() {
            (wms.len() * ws.len(), 1)
        } else {
            (wms.len() * ws.len() * 2, 2)
        };
        let mut results = Box::new_uninit_slice(count);
        if self.is_isotropic() {
            for (i, _) in wms.iter().enumerate() {
                for (j, w) in ws.iter().enumerate() {
                    let idx = (i * ws.len() + j) * idx_mul;
                    let tan_theta2 = tan_theta2(&w) as f64;
                    let a = (1.0 + sqr(self.alpha_x) * tan_theta2).sqrt();
                    results[idx].write(-2.0 * self.alpha_x / sqr(1.0 + a));
                }
            }
        } else {
            for (i, wm) in wms.iter().enumerate() {
                for (j, w) in ws.iter().enumerate() {
                    let idx = (i * ws.len() + j) * idx_mul;
                    let tan_theta2 = tan_theta2(&w) as f64;
                    let cos_phi2 = sqr(cos_phi(&wm)) as f64;
                    let sin_phi2 = sqr(sin_phi(&wm)) as f64;
                    let a = 1.0
                        + (sqr(self.alpha_x) * cos_phi2 + sqr(self.alpha_y) * sin_phi2)
                            * tan_theta2;
                    results[idx]
                        .write(-2.0 * cos_phi2 * self.alpha_x * tan_theta2 / (sqr(1.0 + a) * a));
                    results[idx + 1]
                        .write(-2.0 * sin_phi2 * self.alpha_y * tan_theta2 / (sqr(1.0 + a) * a));
                }
            }
        }
        unsafe { results.assume_init() }
    }
}

#[cfg(test)]
mod test {
    use base::{
        math::sph_to_cart,
        units::{Degs, Rads},
    };

    #[test]
    fn test_msf1_trowbridge_reitz() {
        use super::*;
        use base::math::Vec3;

        let distro = TrowbridgeReitzDistribution::new(0.75, 0.75);
        let wm = Vec3::new(0.0, 0.0, 1.0);
        let evaluated = (0..90)
            .into_iter()
            .map(|i| {
                let w = sph_to_cart(Degs::new(i as f32).to_radians(), Rads::new(0.0));
                distro.eval_msf1(wm, w)
                // distro.eval_g1(&wm, &w)
            })
            .collect::<Vec<_>>();
        println!("{:?}", evaluated);
    }
}

use crate::distro::{MicrofacetDistribution, MicrofacetDistroKind};
#[cfg(feature = "fitting")]
use base::math::{cbr, sin_phi, tan_theta2};
use base::{
    math::{cos_phi, rcp_f64, sqr, tan_theta, Vec3},
    Symmetry,
};
#[cfg(feature = "fitting")]
use libm::{erf, sqrt};
use std::fmt::Debug;

/// Beckmann microfacet distribution function.
///
/// Beckman-Spizzichino distribution is based on the Gaussian distribution of
/// microfacet slopes. If σ is the RMS slope of the microfacets, then the alpha
/// parameter of the Beckmann distribution is given by: $\alpha = \sqrt{2}
/// \sigma$.
#[derive(Copy, Clone)]
pub struct BeckmannDistribution {
    /// Roughness parameter of the MDF (αx = √2σx).
    pub alpha_x: f64,
    /// Roughness parameter of the MDF (αy = √2σy).
    pub alpha_y: f64,
}

impl Debug for BeckmannDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BeckmannDistrib. {{ α_x: {}, α_y: {} }}",
            self.alpha_x, self.alpha_y
        )
    }
}

impl BeckmannDistribution {
    /// Creates a new Beckmann distribution with the given roughness parameters.
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannDistribution {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }

    /// Creates a new isotropic Beckmann distribution with the given roughness.
    pub fn new_isotropic(alpha: f64) -> Self {
        BeckmannDistribution {
            alpha_x: alpha.max(1.0e-6),
            alpha_y: alpha.max(1.0e-6),
        }
    }

    /// Returns whether the distribution is isotropic or anisotropic.
    #[inline]
    pub fn is_isotropic(&self) -> bool { (self.alpha_x - self.alpha_y).abs() < 1.0e-8 }
}

impl MicrofacetDistribution for BeckmannDistribution {
    fn params(&self) -> Self::Params { [self.alpha_x, self.alpha_y] }

    fn set_params(&mut self, params: &Self::Params) {
        self.alpha_x = params[0];
        self.alpha_y = params[1];
    }

    fn kind(&self) -> MicrofacetDistroKind { MicrofacetDistroKind::Beckmann }

    fn symmetry(&self) -> Symmetry {
        if self.is_isotropic() {
            Symmetry::Isotropic
        } else {
            Symmetry::Anisotropic
        }
    }

    #[rustfmt::skip]
    /// Under the assumption that there is no correlation of heights of the
    /// nearby points on the surface, the lambda function for the
    /// Beckmann-Spizzichino distribution has the analytical form:
    ///
    /// $$\Lambda(\mathbf{\omega})=\frac{erf(a)-1+\frac{e^{-a^2}}{a\sqrt{\pi}}}{2}$$
    fn eval_lambda(&self, w: Vec3) -> f64 {
        let alpha = if self.is_isotropic() {
            self.alpha_x
        } else {
            let cos_phi2 = sqr(cos_phi(&w) as f64);
            let sin_phi2 = 1.0 - cos_phi2;
            let alpha2 = sqr(self.alpha_x) * cos_phi2 + sqr(self.alpha_y) * sin_phi2;
            alpha2.sqrt()
        };
        let tan_theta = tan_theta(&w) as f64;
        if tan_theta.is_infinite() {
            return f64::INFINITY;
        }
        if tan_theta.abs() < 1.0e-8 {
            return 0.0;
        }
        let a = 1.0 / (tan_theta * alpha);
        if a.is_infinite() {
            return f64::INFINITY;
        }
        let erf_a = libm::erf(a);
        let exp_a2 = (-sqr(a)).exp();
        (erf_a - 1.0 + exp_a2 * rcp_f64(a * std::f64::consts::PI.sqrt())) * 0.5
    }

    fn eval_ndf(&self, cos_theta: f64, cos_phi: f64) -> f64 {
        let cos_theta2 = sqr(cos_theta);
        let e = if cos_theta2 < 1.0e-16 {
            1.0
        } else {
            let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
            let cos_phi2 = sqr(cos_phi);
            let sin_phi2 = 1.0 - cos_phi2;
            (-tan_theta2
                * (cos_phi2 * rcp_f64(sqr(self.alpha_x)) + sin_phi2 * rcp_f64(sqr(self.alpha_y))))
            .exp()
        };
        let cos_theta4 = sqr(cos_theta2);
        if cos_theta4 < 1.0e-16 {
            return 0.0;
        }
        e * rcp_f64(std::f64::consts::PI * cos_theta4 * self.alpha_x * self.alpha_y)
    }

    fn clone_box(&self) -> Box<dyn MicrofacetDistribution<Params = Self::Params>> {
        Box::new(*self)
    }

    #[cfg(feature = "fitting")]
    fn pd_ndf(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Box<[f64]> {
        debug_assert!(
            cos_thetas.len() == cos_phis.len(),
            "The number of cosines of the zenith angle and the number of cosines of the azimuth \
             angle must be the same."
        );
        let alpha_x2 = sqr(self.alpha_x);
        let alpha_y2 = sqr(self.alpha_y);
        let rcp_alpha_x2 = rcp_f64(alpha_x2);
        let rcp_alpha_y2 = rcp_f64(alpha_y2);
        let d_alpha_x_rcp_denom = rcp_f64(std::f64::consts::PI * sqr(alpha_x2) * self.alpha_y);
        let d_alpha_y_rcp_denom = rcp_f64(std::f64::consts::PI * self.alpha_x * sqr(alpha_y2));
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
                let exp = (-tan_theta2 * (cos_phi2 * rcp_alpha_x2 + sin_phi2 * rcp_alpha_y2)).exp();
                let d_alpha_x = {
                    let numerator = exp * sec_theta4 * (2.0 * cos_phi2 * tan_theta2 - alpha_x2);
                    numerator * d_alpha_x_rcp_denom
                };
                let d_alpha_y = {
                    let numerator = exp * sec_theta4 * (2.0 * sin_phi2 * tan_theta2 - alpha_y2);
                    numerator * d_alpha_y_rcp_denom
                };
                [d_alpha_x, d_alpha_y]
            })
            .collect::<Box<_>>()
    }

    #[cfg(feature = "fitting")]
    fn pd_ndf_iso(&self, cos_thetas: &[f64]) -> Box<[f64]> {
        let mut results = Box::new_uninit_slice(cos_thetas.len());
        let alpha2 = sqr(self.alpha_x);
        let alpha5 = cbr(self.alpha_x) * alpha2;
        for (pd, cos_theta) in results.iter_mut().zip(cos_thetas.iter()) {
            if cos_theta.abs() < 1.0e-6 {
                pd.write(0.0);
                continue;
            }
            let cos_theta2 = sqr(*cos_theta);
            let tan_theta2 = (1.0 - cos_theta2) * rcp_f64(cos_theta2);
            if tan_theta2.is_infinite() {
                pd.write(0.0);
                continue;
            }
            let sec_theta4 = rcp_f64(sqr(cos_theta2));
            let exp = (-tan_theta2 * rcp_f64(alpha2)).exp();
            let d_alpha = {
                let numerator = 2.0 * exp * sec_theta4 * (tan_theta2 - alpha2);
                numerator * rcp_f64(std::f64::consts::PI * alpha5)
            };
            pd.write(d_alpha);
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
                    if tan_theta2.is_infinite() {
                        results[idx].write(0.0);
                    } else if tan_theta2 < 1.0e-8 {
                        results[idx].write(0.0);
                    } else {
                        let cot_theta2 = rcp_f64(tan_theta2);
                        let cot_theta = cot_theta2.sqrt();
                        let e = (cot_theta2 * rcp_f64(self.alpha_x)).exp();
                        let p = e * sqrt(std::f64::consts::PI) * cot_theta;
                        results[idx].write(
                            -2.0 * p
                                * rcp_f64(sqr(
                                    self.alpha_x + p * (1.0 + erf(cot_theta / self.alpha_x))
                                )),
                        );
                    }
                }
            }
        } else {
            for (i, wm) in wms.iter().enumerate() {
                for (j, w) in ws.iter().enumerate() {
                    let idx = (i * ws.len() + j) * idx_mul;
                    let tan_theta2 = tan_theta2(&w) as f64;
                    if tan_theta2.is_infinite() {
                        results[idx].write(0.0);
                    } else if tan_theta2 < 1.0e-8 {
                        results[idx].write(0.0);
                    } else {
                        let cot_theta2 = rcp_f64(tan_theta2);
                        let cot_theta = cot_theta2.sqrt();
                        let cos_phi2 = sqr(cos_phi(wm)) as f64;
                        let sin_phi2 = sqr(sin_phi(wm)) as f64;
                        let alpha2 = sqr(self.alpha_x) * cos_phi2 + sqr(self.alpha_y) * sin_phi2;
                        let alpha = alpha2.sqrt();
                        let e = (cot_theta2 * rcp_f64(alpha2)).exp();
                        let p = e * sqrt(std::f64::consts::PI) * cot_theta;
                        let denom =
                            rcp_f64(alpha * sqr(p * (1.0 + erf(cot_theta / alpha)) + alpha));
                        results[idx].write(-2.0 * p * cos_phi2 * self.alpha_x * denom);
                        results[idx + 1].write(-2.0 * p * sin_phi2 * self.alpha_y * denom);
                    }
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
    fn test_msf1_beckmann() {
        use super::*;
        use base::math::Vec3;

        let distro = BeckmannDistribution::new(0.75, 0.75);
        let wm = Vec3::new(0.0, 0.0, 1.0);
        let evaluated = (0..90)
            .into_iter()
            .map(|i| {
                let w = sph_to_cart(Degs::new(i as f32).to_radians(), Rads::new(0.0));
                distro.eval_msf1(wm, w)
            })
            .collect::<Vec<_>>();
        println!("{:?}", evaluated);
    }
}

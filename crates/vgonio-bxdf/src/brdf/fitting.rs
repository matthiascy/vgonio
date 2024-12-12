use std::arch::x86_64::{_mm512_fmadd_pd, _mm512_set1_pd};
use std::borrow::Cow;
use rayon::prelude::ParallelBridge;
use base::{math, ErrorMetric, Weighting};
use base::optics::ior::IorRegistry;
use base::units::Nanometres;
use jabr::array::DyArr;
use crate::brdf::Bxdf;

mod brute;
mod nllsq;

/// A proxy for a BRDF that can be fitted analytically.
///
/// This is useful when the stored raw BRDF data is in compressed form and
/// cannot be directly used for fitting. The proxy can be used to store
/// the resampled BRDF data to be used for fitting.
///
/// The proxy cannot be constructed directly, but must be created by calling
/// the `proxy` method on the measured BRDF that implements the `AnalyticalFit`
/// trait.
pub struct BrdfFittingProxy<'a, Brdf>
where
    Brdf: AnalyticalFit2,
{
    /// The raw BRDF data that the proxy is associated with.
    brdf: &'a Brdf,
    /// Incident angles (polar angle) of the resampled BRDF data.
    i_thetas: Cow<'a, DyArr<f32>>,
    /// Incident angles (azimuthal angle) of the resampled BRDF data.
    i_phis: Cow<'a, DyArr<f32>>,
    /// Outgoing angles (polar angle) of the resampled BRDF data.
    o_thetas: Cow<'a, DyArr<f32>>,
    /// Outgoing angles (azimuthal angle) of the resampled BRDF data.
    o_phis: Cow<'a, DyArr<f32>>,
    /// The resampled BRDF data that can be used for fitting. The shape of the
    /// array is [Nθ_i, Nφ_i, Nθ_o, Nφ_o] in row-major order.
    resampled: Cow<'a, DyArr<f32, 4>>,
}

/// A trait for BRDFs that can be fitted with an analytical model.
pub trait AnalyticalFit2 {
    /// Create a proxy for this BRDF.
    fn proxy(&self) -> BrdfFittingProxy<Self>;

    /// Returns the wavelengths at which the BRDF is measured.
    fn spectrum(&self) -> &[Nanometres];
}

impl<Brdf> BrdfFittingProxy<'_, Brdf> {
    /// Creates a new proxy filled with the BRDF data computed from the given
    /// analytical BRDF.
    fn new_analytical(model: &dyn Bxdf<Params = [f64; 2]>, iors: &IorRegistry) {
        todo!("Implement this")
    }

    /// Checks if the two proxies have the same parameters.
    fn same_params_p(&self, other: &BrdfFittingProxy<Brdf>) -> bool {
        self.i_thetas == other.i_thetas
            && self.i_phis == other.i_phis
            && self.o_thetas == other.o_thetas
            && self.o_phis == other.o_phis
        && self.brdf.spectrum() == other.brdf.spectrum()
    }

    /// Computes the distance between two BRDF poxies derived from the same
    /// BRDF.
    fn distance(&self, other: &BrdfFittingProxy<Brdf>, metric: ErrorMetric, weighting: Weighting) -> f64 {
        assert!(self.same_params_p(other), "The two BRDFs must have the same parameters");
        assert_eq!(self.resampled.shape(), other.resampled.shape(), "The two BRDFs must have the same shape");

        let factor = match metric {
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
            ErrorMetric::Mse | ErrorMetric::Rmse => math::rcp_f64(self.resampled.len() as f64),
        };

        match weighting {
            Weighting::None => {
                self.resampled.as_slice()
                    .chunks(512).zip(other.resampled.as_slice().chunks(512))
                    .par_bridge()
                    .map(|(xs, ys)| {
                        #[cfg(target_arch = "x86_64")]
                        if is_x86_feature_detected!("avx512f") {
                            use std::arch::x86_64::_mm512_load_pd;
                            use std::arch::x86_64::_mm512_loadu_pd;
                            use std::arch::x86_64::_mm512_sub_pd;
                            use std::arch::x86_64::_mm512_mul_pd;
                            use std::arch::x86_64::_mm512_reduce_add_pd;
                            use std::arch::x86_64::_mm512_set1_pd;

                            unsafe {
                                let factor = _mm512_set1_pd(factor);
                                let sum = xs.chunks(8).zip(ys.chunks(8))
                                    .fold(_mm512_set1_pd(0.0), |acc, (x, y)| {
                                        let (xx, yy) = if x.len() < 8 {
                                            let mut x_aligned = [0.0; 8];
                                            let mut y_aligned = [0.0; 8];
                                            x_aligned[..x.len()].copy_from_slice(x);
                                            y_aligned[..y.len()].copy_from_slice(y);
                                            (_mm512_load_pd(x_aligned.as_ptr()), _mm512_load_pd(y_aligned.as_ptr()))
                                        } else {
                                            (_mm512_loadu_pd(x.as_ptr()), _mm512_loadu_pd(y.as_ptr()))
                                        };

                                        let diff = _mm512_sub_pd(xx, yy);
                                        let sqr = _mm512_mul_pd(diff, diff);
                                       _mm512_fmadd_pd(sqr, factor, acc)
                                    });
                                _mm512_reduce_add_pd(sum)
                            }
                        } else {
                            // TODO: Use AVX2
                            xs.iter().zip(ys.iter())
                                .fold(0.0f64, |acc, (x, y)| {
                                    let diff = *x as f64 - *y as f64;
                                    acc + math::sqr(diff) * factor
                                })
                        }

                        #[cfg(not(target_arch = "x86_64"))]
                        xs.iter().zip(ys.iter())
                            .fold(0.0f64, |acc, (x, y)| {
                                let diff = *x as f64 - *y as f64;
                                acc + math::sqr(diff) * factor
                            })

                    })
                    .reduce(|| 0.0, |a, b| a + b)
            }
            Weighting::LnCos => {
                let theta_i_stride = self.resampled.strides()[0];
                self.resampled.as_slice()
                    .chunks(theta_i_stride)
                    .zip(other.resampled.as_slice().chunks(theta_i_stride))
                    .zip(self.i_thetas.iter())
                    .par_bridge()
                    .map(|((xs, ys), theta_i)| {
                        let cos_theta_i = theta_i.cos();
                        xs.iter().zip(ys.iter())
                            .fold(0.0, |acc, (x, y)| {
                                let diff = *x as f64 - *y as f64;
                                let weight = math::ln_cos(*theta_i as f64);
                                acc + math::sqr(diff) * factor * weight
                            })
                    })
            }
        }
    }

    /// Fit the BRDF with the Cook-Torrance model.
    pub fn fit_cook_torrance(&self) -> CookTorrance {
        let mut fit = CookTorrance::default();
        fit.fit(self);
        fit
    }
}

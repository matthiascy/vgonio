use crate::brdf::Bxdf;
use base::{math, optics::ior::IorRegistry, units::Nanometres, ErrorMetric, Weighting};
use jabr::array::DyArr;
use rayon::prelude::ParallelBridge;
use std::{
    arch::x86_64::{
        _mm512_fmadd_pd, _mm512_load_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd,
        _mm512_sub_pd,
    },
    borrow::Cow,
};

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
    /// Incident angles (polar angle) in radians of the resampled BRDF data.
    i_thetas: Cow<'a, DyArr<f32>>,
    /// Incident angles (azimuthal angle) in radians of the resampled BRDF data.
    i_phis: Cow<'a, DyArr<f32>>,
    /// Outgoing angles (polar angle) in radians of the resampled BRDF data.
    o_thetas: Cow<'a, DyArr<f32>>,
    /// Outgoing angles (azimuthal angle) in radians of the resampled BRDF data.
    o_phis: Cow<'a, DyArr<f32>>,
    /// The resampled BRDF data that can be used for fitting. The shape of the
    /// array is [Nθ_i, Nφ_i, Nθ_o, Nφ_o] in row-major order.
    resampled: Cow<'a, DyArr<f32, 4>>,
}

/// A trait for BRDFs that can be fitted with an analytical model.
pub trait AnalyticalFit2 {
    /// Create a proxy for this BRDF.
    fn proxy() -> BrdfFittingProxy<Self>;

    /// Returns the wavelengths at which the BRDF is measured.
    fn spectrum(&self) -> &[Nanometres];
}

impl<Brdf> BrdfFittingProxy<'_, Brdf>
where
    Brdf: AnalyticalFit2,
{
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
    fn distance(
        &self,
        other: &BrdfFittingProxy<Brdf>,
        metric: ErrorMetric,
        weighting: Weighting,
    ) -> f64 {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );

        let factor = match metric {
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
            ErrorMetric::Mse | ErrorMetric::Rmse => math::rcp_f64(self.resampled.len() as f64),
        };

        match weighting {
            Weighting::None => self
                .resampled
                .as_slice()
                .chunks(512)
                .zip(other.resampled.as_slice().chunks(512))
                .par_bridge()
                .map(|(xs, ys)| {
                    xs.iter().zip(ys.iter()).fold(0.0f64, |acc, (x, y)| {
                        let diff = *x as f64 - *y as f64;
                        acc + math::sqr(diff) * factor
                    })
                })
                .reduce(|| 0.0, |a, b| a + b),
            Weighting::LnCos => {
                let theta_i_stride = self.resampled.strides()[0];
                self.resampled
                    .as_slice()
                    .chunks(theta_i_stride)
                    .zip(other.resampled.as_slice().chunks(theta_i_stride))
                    .zip(self.i_thetas.iter())
                    .par_bridge()
                    .map(|((xs, ys), theta_i)| {
                        let cos_theta_i = theta_i.cos();
                        xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                            let ln_x_cos = (x * cos_theta_i + 1.0).ln();
                            let ln_y_cos = (y * cos_theta_i + 1.0).ln();
                            let diff = ln_x_cos - ln_y_cos;
                            acc + math::sqr(diff) * factor
                        })
                    })
            },
        }
    }

    /// Computes the distance between two BRDF poxies derived from the same
    /// BRDF with a filtered range of incident and outgoing angles.
    fn distance_filtered(
        &self,
        other: &BrdfFittingProxy<Brdf>,
        metric: ErrorMetric,
        weighting: Weighting,
        max_theta_i: f32,
        max_theta_o: f32,
    ) -> f64 {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );

        // Find the cutoff indices for theta angles only
        let theta_i_limit = self.i_thetas.partition_point(|&x| x <= max_theta_i);
        let theta_o_limit = self.o_thetas.partition_point(|&x| x <= max_theta_o);

        let shape = self.resampled.shape();
        let strides = self.resampled.strides();

        let factor = match metric {
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
            ErrorMetric::Mse | ErrorMetric::Rmse => {
                math::rcp_f64((theta_i_limit * shape[1] * theta_o_limit * shape[3]) as f64)
            },
        };

        match weighting {
            Weighting::None => {
                // Process each theta_i slice in parallel
                (0..theta_i_limit)
                    .into_par_iter()
                    .map(|i_theta_idx| {
                        let base_offset = i_theta_idx * strides[0];

                        // Process all phi_i values
                        (0..shape[1])
                            .map(|i_phi_idx| {
                                let phi_offset = base_offset + i_phi_idx * strides[1];

                                // Process theta_o up to limit
                                let mut sum = 0.0;
                                for o_theta_idx in 0..theta_o_limit {
                                    let theta_o_offset = phi_offset + o_theta_idx * strides[2];

                                    // Process all phi_o values
                                    for o_phi_idx in 0..shape[3] {
                                        let idx = theta_o_offset + o_phi_idx;
                                        let x = self.resampled.as_slice()[idx];
                                        let y = other.resampled.as_slice()[idx];
                                        let diff = x as f64 - y as f64;
                                        sum += math::sqr(diff) * factor;
                                    }
                                }
                                sum
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            },
            Weighting::LnCos => (0..theta_i_limit)
                .into_par_iter()
                .map(|i_theta_idx| {
                    let cos_theta_i = self.i_thetas[i_theta_idx].cos();
                    let base_offset = i_theta_idx * strides[0];

                    (0..shape[1])
                        .map(|i_phi_idx| {
                            let phi_offset = base_offset + i_phi_idx * strides[1];

                            let mut sum = 0.0;
                            for o_theta_idx in 0..theta_o_limit {
                                let theta_o_offset = phi_offset + o_theta_idx * strides[2];

                                for o_phi_idx in 0..shape[3] {
                                    let idx = theta_o_offset + o_phi_idx;
                                    let x = self.resampled.as_slice()[idx];
                                    let y = other.resampled.as_slice()[idx];
                                    let ln_x_cos = (x * cos_theta_i + 1.0).ln();
                                    let ln_y_cos = (y * cos_theta_i + 1.0).ln();
                                    let diff = ln_x_cos - ln_y_cos;
                                    sum += math::sqr(diff) * factor;
                                }
                            }
                            sum
                        })
                        .sum::<f64>()
                })
                .sum::<f64>(),
        }
    }

    /// Fit the BRDF with the Cook-Torrance model.
    pub fn fit_cook_torrance(&self) -> CookTorrance {
        let mut fit = CookTorrance::default();
        fit.fit(self);
        fit
    }
}

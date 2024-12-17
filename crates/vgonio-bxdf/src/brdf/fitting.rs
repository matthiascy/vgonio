use crate::{brdf::Bxdf, Scattering};
use base::{
    math::{self, Sph2},
    medium::Medium,
    optics::ior::IorRegistry,
    units::{rad, Nanometres},
    ErrorMetric, Weighting,
};
use jabr::array::{DyArr, DynArr};
use rayon::{iter::ParallelIterator, prelude::ParallelBridge};
use std::borrow::Cow;

pub mod brute;
mod nllsq;

/// The source of the proxy.
///
/// This is needed to mark the proxy as being derived from a measured BRDF or
/// an analytical BRDF as when generating the proxy data points from an
/// analytical BRDF to match the measured BRDF data points, the
/// [`BrdfFittingProxy::brdf`] will be the same as the measured BRDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxySource {
    /// The proxy is derived from a measured BRDF.
    Measured,
    /// The proxy is derived from an analytical BRDF.
    Analytical,
}

/// The outgoing directions of the resampled BRDF data.
///
/// The outgoing directions can be represented in two ways:
/// 1. As a cartesian product of theta and phi angles, where directions are
///    computed by iterating over all combinations of theta_o and phi_o
/// 2. As an explicit list of directions, where each direction is specified
///    directly
#[derive(Debug, Clone)]
pub enum OutgoingDirs<'a> {
    /// Outgoing directions computed from combinations of theta and phi angles
    Grid {
        /// Outgoing polar angles in radians.
        theta_o: Cow<'a, DyArr<f32>>,
        /// Outgoing azimuthal angles in radians.
        phi_o: Cow<'a, DyArr<f32>>,
    },
    /// List of outgoing directions organized by theta bands, suitable for
    /// the case where the outgoing directions are arranged on top of the
    /// hemisphere preserving the theta band structure.
    List {
        /// Outgoing polar angles in radians
        theta_o: Cow<'a, DyArr<f32>>,
        /// All phi_o angles in radians stored in a flat array
        phi_o: Cow<'a, DyArr<f32>>,
        /// Offsets into phi_o array for each theta_o band.
        ///
        /// The length of this array is 1 + theta_o.len(), where the last
        /// element is the length of the phi_o array.
        /// For a given theta_o band of index `i`, `phi_o[i]` is the first
        /// element of the phi_o array for the theta_o band, `phi_o[i + 1]` is
        /// the last element of the phi_o array for the theta_o band.
        offsets: DyArr<usize>,
    },
}

impl<'a> OutgoingDirs<'a> {
    pub fn new_grid(theta_o: Cow<'a, DyArr<f32>>, phi_o: Cow<'a, DyArr<f32>>) -> Self {
        Self::Grid { theta_o, phi_o }
    }

    pub fn new_list(
        theta_o: Cow<'a, DyArr<f32>>,
        phi_o: Cow<'a, DyArr<f32>>,
        offsets: DyArr<usize>,
    ) -> Self {
        assert_eq!(
            theta_o.len(),
            offsets.len() - 1,
            "The length of theta_o must be one less than the length of offsets"
        );
        Self::List {
            theta_o,
            phi_o,
            offsets,
        }
    }
}

/// Partial equality for [`OutgoingDirectionSet`].
///
/// This is needed to check if two [`BrdfFittingProxy`] have the same
/// parameters.
impl<'a> PartialEq for OutgoingDirs<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                OutgoingDirs::Grid {
                    theta_o: a,
                    phi_o: b,
                },
                OutgoingDirs::Grid {
                    theta_o: c,
                    phi_o: d,
                },
            ) => a == c && b == d,
            (
                OutgoingDirs::List {
                    theta_o: a,
                    phi_o: b,
                    offsets: c,
                },
                OutgoingDirs::List {
                    theta_o: d,
                    phi_o: e,
                    offsets: f,
                },
            ) => a == d && b == e && c == f,
            _ => false,
        }
    }
}

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
    /// The source of the proxy.
    pub(crate) source: ProxySource,
    /// The raw BRDF data that the proxy is associated with.
    pub(crate) brdf: &'a Brdf,
    /// Incident angles (polar angle) in radians of the resampled BRDF data.
    pub(crate) i_thetas: Cow<'a, DyArr<f32>>,
    /// Incident angles (azimuthal angle) in radians of the resampled BRDF data.
    pub(crate) i_phis: Cow<'a, DyArr<f32>>,
    /// Outgoing directions of the resampled BRDF data.
    pub(crate) o_dirs: OutgoingDirs<'a>,
    /// The resampled BRDF data that can be used for fitting. Depends on the
    /// outgoing directions. The shape of the array could be
    /// - [Nθ_i, Nφ_i, Nθ_o, Nφ_o, Nλ] in row-major order
    /// - [Nθ_i, Nφ_i, Nω_o, Nλ] in row-major order
    pub(crate) resampled: Cow<'a, DynArr<f32>>,
}

/// A trait for BRDFs that can be fitted with an analytical model.
pub trait AnalyticalFit2 {
    /// Create a proxy for this BRDF.
    fn proxy(&self) -> BrdfFittingProxy<Self>
    where
        Self: Sized;

    /// Returns the wavelengths at which the BRDF is measured.
    fn spectrum(&self) -> &[Nanometres];

    /// Returns the incident medium.
    fn medium_i(&self) -> Medium;

    /// Returns the transmitted medium.
    fn medium_t(&self) -> Medium;
}

impl<Brdf> BrdfFittingProxy<'_, Brdf>
where
    Brdf: AnalyticalFit2,
{
    /// Returns the source of the proxy.
    pub fn source(&self) -> ProxySource { self.source }

    /// Returns the resampled BRDF data.
    pub fn samples(&self) -> &[f32] { self.resampled.as_slice() }

    /// Checks if the two proxies have the same parameters.
    fn same_params_p<O: AnalyticalFit2>(&self, other: &BrdfFittingProxy<O>) -> bool {
        self.i_thetas == other.i_thetas
            && self.i_phis == other.i_phis
            && self.o_dirs == other.o_dirs
            && self.brdf.spectrum() == other.brdf.spectrum()
            && self.resampled.shape() == other.resampled.shape()
    }

    /// Computes the distance between two BRDF poxies derived from the same
    /// BRDF.
    fn distance<O: AnalyticalFit2>(
        &self,
        other: &BrdfFittingProxy<O>,
        metric: ErrorMetric,
        weighting: Weighting,
    ) -> f64 {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );
        let is_rmse = metric == ErrorMetric::Rmse;
        let factor = match metric {
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
            ErrorMetric::Mse | ErrorMetric::Rmse => math::rcp_f64(self.resampled.len() as f64),
        };
        let stride_theta_i = self.resampled.strides()[0];

        let sum = self
            .resampled
            .as_slice()
            .chunks(stride_theta_i)
            .zip(other.resampled.as_slice().chunks(stride_theta_i))
            .zip(self.i_thetas.iter())
            .par_bridge()
            .map(|((xs, ys), theta_i)| {
                let cos_theta_i = theta_i.cos();
                match weighting {
                    Weighting::None => {
                        xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                            let diff = *x as f64 - *y as f64;
                            acc + math::sqr(diff)
                        }) * factor
                    },
                    Weighting::LnCos => {
                        xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                            let diff = ((x * cos_theta_i + 1.0) as f64).ln()
                                - ((y * cos_theta_i + 1.0) as f64).ln();
                            acc + math::sqr(diff)
                        }) * factor
                    },
                }
            })
            .reduce(|| 0.0, |a, b| a + b);

        if is_rmse {
            sum.sqrt()
        } else {
            sum
        }
    }

    /// Computes the distance between two BRDF poxies derived from the same
    /// BRDF with a filtered range of incident and outgoing angles.
    fn distance_filtered<O: AnalyticalFit2>(
        &self,
        other: &BrdfFittingProxy<O>,
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
        let n_theta_i = self
            .i_thetas
            .as_slice()
            .partition_point(|&x| x < max_theta_i);
        let is_rmse = metric == ErrorMetric::Rmse;

        let sum = match &self.o_dirs {
            OutgoingDirs::Grid { theta_o, phi_o: _ } => {
                let n_theta_o = theta_o.as_slice().partition_point(|&x| x < max_theta_o);
                // [Nθ_i, Nφ_i, Nθ_o, Nφ_o, Nλ]
                let shape = self.resampled.shape();

                let factor = match metric {
                    ErrorMetric::Nllsq => 0.5,
                    ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
                    ErrorMetric::Mse | ErrorMetric::Rmse => math::rcp_f64(
                        (n_theta_i * shape[1] * n_theta_o * shape[3] * shape[4]) as f64,
                    ),
                };

                let xs = &self.resampled;
                let ys = &other.resampled;

                (0..n_theta_i)
                    .into_iter()
                    .zip(self.i_thetas.as_slice().iter())
                    .par_bridge()
                    .map(|(i, theta_i)| {
                        let cos_theta_i = theta_i.cos() as f64;
                        (0..shape[1])
                            .map(|j| {
                                let mut sum = 0.0;
                                for k in 0..n_theta_o {
                                    for l in 0..shape[3] {
                                        for m in 0..shape[4] {
                                            let x = xs[[i, j, k, l, m]] as f64;
                                            let y = ys[[i, j, k, l, m]] as f64;
                                            let diff = match weighting {
                                                Weighting::None => x - y,
                                                Weighting::LnCos => {
                                                    let ln_x_cos = (x * cos_theta_i + 1.0).ln();
                                                    let ln_y_cos = (y * cos_theta_i + 1.0).ln();
                                                    ln_x_cos - ln_y_cos
                                                },
                                            };
                                            sum += math::sqr(diff);
                                        }
                                    }
                                }
                                sum * factor
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            },
            OutgoingDirs::List {
                theta_o,
                phi_o: _,
                offsets,
            } => {
                let n_theta_o = theta_o.as_slice().partition_point(|&x| x < max_theta_o);
                // The total number of outgoing directions after filtering
                let n_wo = offsets[n_theta_o];
                // [Nθ_i, Nφ_i, Nω_o, Nλ]
                let shape = self.resampled.shape();

                let factor = match metric {
                    ErrorMetric::Nllsq => 0.5,
                    ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
                    ErrorMetric::Mse | ErrorMetric::Rmse => {
                        math::rcp_f64((n_theta_i * shape[1] * n_wo * shape[3]) as f64)
                    },
                };

                let xs = &self.resampled;
                let ys = &other.resampled;

                (0..n_theta_i)
                    .into_iter()
                    .zip(self.i_thetas.as_slice().iter())
                    .par_bridge()
                    .map(|(i, theta_i)| {
                        let cos_theta_i = theta_i.cos();
                        (0..shape[1])
                            .map(|j| {
                                let mut sum = 0.0;
                                for k in 0..n_theta_o {
                                    for l in offsets[k]..offsets[k + 1] {
                                        for m in 0..shape[3] {
                                            let x = xs[[i, j, l, m]];
                                            let y = ys[[i, j, l, m]];
                                            let diff = match weighting {
                                                Weighting::None => x as f64 - y as f64,
                                                Weighting::LnCos => {
                                                    let ln_x_cos =
                                                        ((x * cos_theta_i + 1.0) as f64).ln();
                                                    let ln_y_cos =
                                                        ((y * cos_theta_i + 1.0) as f64).ln();
                                                    ln_x_cos - ln_y_cos
                                                },
                                            };
                                            sum += math::sqr(diff);
                                        }
                                    }
                                }
                                sum * factor
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            },
        };

        if is_rmse {
            sum.sqrt()
        } else {
            sum
        }
    }

    // TODO: potentially considering generating only the data points for the
    // filtered incident and outgoing angles
    /// Generate the data points following the same incident and outgoing angles
    /// for the given analytical BRDF.
    fn generate_analytical(
        &self,
        medium_i: Medium,
        medium_t: Medium,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &IorRegistry,
    ) -> Self {
        let iors_i = iors
            .ior_of_spectrum(medium_i, self.brdf.spectrum())
            .unwrap();
        let iors_t = iors
            .ior_of_spectrum(medium_t, self.brdf.spectrum())
            .unwrap();
        let n_spectrum = self.brdf.spectrum().len();
        let mut resampled = DynArr::zeros(self.resampled.shape());
        let i_thetas = self.i_thetas.as_slice();
        let i_phis = self.i_phis.as_slice();
        let strides = self.resampled.strides();

        match &self.o_dirs {
            OutgoingDirs::Grid { theta_o, phi_o } => {
                resampled
                    .as_mut_slice()
                    .chunks_mut(strides[0])
                    .zip(i_thetas.iter())
                    .par_bridge()
                    .for_each(|(per_theta_i, theta_i)| {
                        per_theta_i
                            .chunks_mut(strides[1])
                            .zip(i_phis.iter())
                            .for_each(|(per_phi_i, phi_i)| {
                                let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                                per_phi_i
                                    .chunks_mut(strides[2])
                                    .zip(theta_o.as_slice().iter())
                                    .for_each(|(per_theta_o, theta_o)| {
                                        per_theta_o
                                            .chunks_mut(strides[3])
                                            .zip(phi_o.as_slice().iter())
                                            .for_each(|(per_phi_o, phi_o)| {
                                                let vo = Sph2::new(rad!(*theta_o), rad!(*phi_o))
                                                    .to_cartesian();
                                                let spectral_samples =
                                                    Scattering::eval_reflectance_spectrum(
                                                        model, &vi, &vo, &iors_i, &iors_t,
                                                    )
                                                    .iter()
                                                    .map(|&x| x as f32)
                                                    .collect::<Box<[f32]>>();

                                                per_phi_o.copy_from_slice(&spectral_samples);
                                            });
                                    });
                            });
                    });
            },
            OutgoingDirs::List {
                theta_o,
                phi_o,
                offsets,
            } => {
                // Parallelize the filling on the theta_i dimension
                resampled
                    .as_mut_slice()
                    .chunks_mut(strides[0])
                    .zip(i_thetas.iter())
                    .par_bridge()
                    .for_each(|(per_theta_i, theta_i)| {
                        // Iterate over the phi_i dimension
                        per_theta_i
                            .chunks_mut(strides[1])
                            .zip(i_phis.iter())
                            .for_each(|(per_phi_i, phi_i)| {
                                let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                                // Iterate over the wo dimension
                                let mut wo_idx = 0;
                                for (i, theta_o) in theta_o.iter().enumerate() {
                                    for phi_o in phi_o[offsets[i]..offsets[i + 1]].iter() {
                                        let vo =
                                            Sph2::new(rad!(*theta_o), rad!(*phi_o)).to_cartesian();
                                        let spectral_samples =
                                            Scattering::eval_reflectance_spectrum(
                                                model, &vi, &vo, &iors_i, &iors_t,
                                            )
                                            .iter()
                                            .map(|&x| x as f32)
                                            .collect::<Box<[f32]>>();
                                        let offset = wo_idx * strides[2];
                                        per_phi_i[offset..offset + n_spectrum]
                                            .copy_from_slice(&spectral_samples);
                                        wo_idx += 1;
                                    }
                                }
                            });
                    });
            },
        };

        BrdfFittingProxy {
            source: ProxySource::Analytical,
            brdf: self.brdf,
            i_thetas: self.i_thetas.clone(),
            i_phis: self.i_phis.clone(),
            o_dirs: self.o_dirs.clone(),
            resampled: Cow::Owned(resampled),
        }
    }
}

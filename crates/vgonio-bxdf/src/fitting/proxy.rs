use crate::{brdf::AnalyticalBrdf, AnyMeasuredBrdf, Scattering};
#[cfg(feature = "cuda")]
use cust::{
    launch,
    module::Module,
    prelude::{CopyDestination, DeviceBuffer},
    stream::Stream,
};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::borrow::Cow;
#[cfg(feature = "cuda")]
use std::cmp::{max, min};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use vgn_core::{
    math,
    math::Sph2,
    optics::Ior,
    units::{rad, Nanometres, Radians},
    ErrorMetric, Weighting,
};
use vgn_jabr::array::{shape, DyArr, DynArr, MemLayout};
// TODO: fix L1 distance

/// The source of the proxy.
///
/// This is needed to mark the proxy as being derived from a measured BRDF
/// or an analytical BRDF as when generating the proxy data points from
/// an analytical BRDF to match the measured BRDF data points, the
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
/// 1. As a cartesian product of theta and phi angles, where directions are computed by iterating
///    over all combinations of theta_o and phi_o
/// 2. As an explicit list of directions, where each direction is specified directly
#[derive(Debug, Clone)]
pub enum OutgoingDirs<'a> {
    /// Outgoing directions computed from combinations of theta and phi
    /// angles
    Grid {
        /// Outgoing polar angles in radians.
        o_thetas: Cow<'a, DyArr<f32>>,
        /// Outgoing azimuthal angles in radians.
        o_phis: Cow<'a, DyArr<f32>>,
    },
    /// List of outgoing directions organized by theta bands, suitable for
    /// the case where the outgoing directions are arranged on top of the
    /// hemisphere preserving the theta band structure.
    List {
        /// Outgoing polar angles in radians
        o_thetas: Cow<'a, DyArr<f32>>,
        /// All phi_o angles in radians stored in a flat array
        o_phis: Cow<'a, DyArr<f32>>,
        /// Offsets into phi_o array for each theta_o band.
        ///
        /// The length of this array is 1 + theta_o.len(), where the last
        /// element is the length of the phi_o array.
        /// For a given theta_o band of index `i`, `phi_o[i]` is the first
        /// element of the phi_o array for the theta_o band, `phi_o[i + 1]`
        /// is the last element of the phi_o array for the
        /// theta_o band (exclusive).
        offsets: DyArr<usize>,
    },
}

impl<'a> OutgoingDirs<'a> {
    /// Creates a new outgoing directions grid.
    ///
    /// # Arguments
    ///
    /// * `o_thetas` - Outgoing polar angles in radians.
    /// * `o_phis` - Outgoing azimuthal angles in radians.
    pub fn new_grid(o_thetas: Cow<'a, DyArr<f32>>, o_phis: Cow<'a, DyArr<f32>>) -> Self {
        Self::Grid { o_thetas, o_phis }
    }

    /// Creates a new outgoing directions list.
    ///
    /// # Arguments
    ///
    /// * `o_thetas` - Outgoing polar angles in radians.
    /// * `o_phis` - Outgoing azimuthal angles in radians.
    /// * `offsets` - Offsets into phi_o array for each theta_o band.
    pub fn new_list(
        o_thetas: Cow<'a, DyArr<f32>>,
        o_phis: Cow<'a, DyArr<f32>>,
        offsets: DyArr<usize>,
    ) -> Self {
        assert_eq!(
            o_thetas.len(),
            offsets.len() - 1,
            "The length of theta_o must be one less than the length of offsets"
        );
        Self::List {
            o_thetas,
            o_phis,
            offsets,
        }
    }
}

/// Partial equality for [`OutgoingDirectionSet`].
///
/// This is needed to check if two [`BrdfFittingProxy`] have the same
/// parameters.
impl PartialEq for OutgoingDirs<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                OutgoingDirs::Grid {
                    o_thetas: a,
                    o_phis: b,
                },
                OutgoingDirs::Grid {
                    o_thetas: c,
                    o_phis: d,
                },
            ) => a == c && b == d,
            (
                OutgoingDirs::List {
                    o_thetas: a,
                    o_phis: b,
                    offsets: c,
                },
                OutgoingDirs::List {
                    o_thetas: d,
                    o_phis: e,
                    offsets: f,
                },
            ) => a == d && b == e && c == f,
            _ => false,
        }
    }
}

/// A proxy for a measured BRDF which generalises the measured BRDF data
/// to be used for fitting.
///
/// This is useful when the stored raw BRDF data is in compressed form and
/// cannot be directly used for fitting. The proxy can be used to storage
/// the resampled BRDF data to be used for fitting.
///
/// The proxy cannot be constructed directly, but must be created by calling
/// the `proxy` method on the measured BRDF that implements the
/// `AnalyticalFit` trait.
pub struct BrdfProxy<'a> {
    /// Indicates if the proxy has NaN values.
    pub has_nan: bool,
    /// The source of the proxy.
    pub source: ProxySource,
    /// The raw BRDF data that the proxy is associated with.
    pub brdf: &'a dyn AnyMeasuredBrdf,
    /// The wavelength spectrum of the proxy; this is not always the same as
    /// the spectrum of the raw BRDF data as the proxy can be generated for
    /// a single wavelength.
    pub spectrum: Box<[Nanometres]>,
    /// Incident angles (polar angle) in radians of the resampled BRDF data.
    pub i_thetas: Cow<'a, DyArr<f32>>,
    /// Incident angles (azimuthal angle) in radians of the resampled BRDF
    /// data.
    pub i_phis: Cow<'a, DyArr<f32>>,
    /// Outgoing directions of the resampled BRDF data.
    pub o_dirs: OutgoingDirs<'a>,
    /// The resampled BRDF data that can be used for fitting. Depends on the
    /// outgoing directions. The shape of the array could be
    /// - [Nθ_i, Nφ_i, Nθ_o, Nφ_o, Nλ] in row-major order
    /// - [Nθ_i, Nφ_i, Nω_o, Nλ] in row-major order
    pub resampled: Cow<'a, DynArr<f32>>,
    /// Precomputed IORs for the incident medium.
    pub iors_i: Cow<'a, [Ior]>,
    /// Precomputed IORs for the transmitted medium.
    pub iors_t: Cow<'a, [Ior]>,
}

impl<'a> BrdfProxy<'a> {
    /// Creates a new BRDF proxy.
    pub fn new(
        has_nan: bool,
        source: ProxySource,
        brdf: &'a dyn AnyMeasuredBrdf,
        i_thetas: Cow<'a, DyArr<f32>>,
        i_phis: Cow<'a, DyArr<f32>>,
        o_dirs: OutgoingDirs<'a>,
        resampled: Cow<'a, DynArr<f32>>,
        iors_i: Cow<'a, [Ior]>,
        iors_t: Cow<'a, [Ior]>,
    ) -> Self {
        let spectrum = brdf.spectrum().to_vec().into_boxed_slice();
        Self {
            has_nan,
            source,
            brdf,
            spectrum,
            i_thetas,
            i_phis,
            o_dirs,
            resampled,
            iors_i,
            iors_t,
        }
    }

    /// Returns the source of the proxy.
    pub fn source(&self) -> ProxySource { self.source }

    /// Returns the resampled BRDF data.
    pub fn samples(&self) -> &DynArr<f32> { &self.resampled }

    /// Returns the refractive indices for the incident medium.
    pub fn iors_i(&self) -> &[Ior] { &self.iors_i }

    /// Returns the refractive indices for the transmitted medium.
    pub fn iors_t(&self) -> &[Ior] { &self.iors_t }

    /// Returns single wavelength proxy.
    pub fn per_wavelength(&'a self, idx: usize) -> BrdfProxy<'a> {
        assert!(
            idx < self.spectrum.len(),
            "The index must be less than the number of wavelengths"
        );
        // Grid: [Nθ_i, Nφ_i, Nθ_o, Nφ_o, Nλ] -> [Nθ_i, Nφ_i, Nθ_o, Nφ_o, 1]
        // List: [Nθ_i, Nφ_i, Nω_o, Nλ] -> [Nθ_i, Nφ_i, Nω_o, 1]
        let shape = self.resampled.shape();
        let mut new_shape = shape.to_vec();
        let last_idx = new_shape.len() - 1;
        new_shape[last_idx] = 1;

        let mut sampled = DynArr::zeros(&new_shape);
        let iors_i = &self.iors_i[idx..idx + 1];
        let iors_t = &self.iors_t[idx..idx + 1];
        let spectrum = &self.spectrum[idx..idx + 1];

        match self.o_dirs {
            OutgoingDirs::Grid { .. } => {
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            for l in 0..shape[3] {
                                sampled[[i, j, k, l, 0]] = self.resampled[[i, j, k, l, idx]];
                            }
                        }
                    }
                }
            },
            OutgoingDirs::List { .. } => {
                for i in 0..shape[0] {
                    for j in 0..shape[1] {
                        for k in 0..shape[2] {
                            sampled[[i, j, k, 0]] = self.resampled[[i, j, k, idx]];
                        }
                    }
                }
            },
        }

        Self {
            has_nan: self.has_nan,
            source: self.source,
            brdf: self.brdf,
            spectrum: spectrum.to_vec().into_boxed_slice(),
            i_thetas: self.i_thetas.clone(),
            i_phis: self.i_phis.clone(),
            o_dirs: self.o_dirs.clone(),
            resampled: Cow::Owned(sampled),
            iors_i: Cow::Borrowed(iors_i),
            iors_t: Cow::Borrowed(iors_t),
        }
    }

    /// Returns the number of samples used for fitting.
    ///
    /// This excludes the samples that either are NaN values and the samples
    /// that are filtered out.
    pub fn n_filtered_samples(
        &self,
        max_theta_i: Option<Radians>,
        max_theta_o: Option<Radians>,
    ) -> usize {
        let shape = self.filtered_shape(
            max_theta_i.unwrap_or(Radians::HALF_PI).as_f32(),
            max_theta_o.unwrap_or(Radians::HALF_PI).as_f32(),
        );
        let total = shape.iter().product::<usize>();

        if !self.has_nan {
            return total;
        }

        let mut n_nan = 0;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        for m in 0..shape[4] {
                            if self.resampled[[i, j, k, l, m]].is_nan() {
                                n_nan += 1;
                            }
                        }
                    }
                }
            }
        }

        total - n_nan
    }

    /// Returns the shape of the resampled BRDF data after filtering.
    /// Depends on the outgoing directions, the dimensions could be 4 or 5.
    pub fn filtered_shape(&self, max_theta_i: f32, max_theta_o: f32) -> Box<[usize]> {
        if max_theta_i >= std::f32::consts::FRAC_PI_2 && max_theta_o >= std::f32::consts::FRAC_PI_2
        {
            // Clone the shape as no filtering is needed
            return self.resampled.shape().to_vec().into_boxed_slice();
        }

        let old_shape = self.resampled.shape();
        let n_theta_i = self
            .i_thetas
            .as_slice()
            .partition_point(|&x| x < max_theta_i);
        match &self.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o, ..
            } => {
                let n_theta_o = theta_o.as_slice().partition_point(|&x| x < max_theta_o);
                Box::new([
                    n_theta_i,    // Nθ_i
                    old_shape[1], // Nφ_i
                    n_theta_o,    // Nθ_o
                    old_shape[3], // Nφ_o
                    old_shape[4], // Nλ
                ])
            },
            OutgoingDirs::List {
                o_thetas: theta_o,
                offsets,
                ..
            } => {
                let n_theta_o = theta_o.as_slice().partition_point(|&x| x < max_theta_o);
                let n_wo = offsets[n_theta_o];
                Box::new([
                    n_theta_i,    // Nθ_i
                    old_shape[1], // Nφ_i
                    n_wo,         // Nω_o
                    old_shape[3], // Nλ
                ])
            },
        }
    }

    /// Checks if the two proxies have the same parameters.
    fn same_params_p(&self, other: &BrdfProxy) -> bool {
        self.i_thetas == other.i_thetas
            && self.i_phis == other.i_phis
            && self.o_dirs == other.o_dirs
            && self.brdf.spectrum() == other.brdf.spectrum()
            && self.spectrum == other.spectrum
            && self.resampled.shape() == other.resampled.shape()
    }

    /// Computes the distance between two BRDF proxies derived from the same
    /// BRDF.
    pub fn distance(
        &self,
        other: &BrdfProxy,
        metric: ErrorMetric,
        weighting: Weighting,
        #[cfg(feature = "cuda")] gpu_threads_per_block: Option<u32>,
        #[cfg(feature = "cuda")] gpu_modules: Option<&HashMap<&'static str, Module>>,
    ) -> f64 {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );
        let has_nan = self.has_nan || other.has_nan;
        let is_rmse = metric == ErrorMetric::Rmse;
        let effective_n = if has_nan {
            self.resampled
                .iter()
                .zip(other.resampled.iter())
                .filter(|(x, y)| !x.is_nan() && !y.is_nan())
                .count()
        } else {
            self.resampled.len()
        };
        if effective_n == 0 {
            return 0.0;
        }
        let factor = match metric {
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 | ErrorMetric::L2 => 1.0,
            ErrorMetric::Mse | ErrorMetric::Rmse => math::rcp_f64(effective_n as f64),
        };
        let stride_theta_i = self.resampled.strides()[0];

        #[cfg(not(feature = "cuda"))]
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
                            if has_nan && (x.is_nan() || y.is_nan()) {
                                return acc;
                            }
                            let diff = *x as f64 - *y as f64;
                            acc + math::sqr(diff)
                        }) * factor
                    },
                    Weighting::LnCos => {
                        xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                            if has_nan && (x.is_nan() || y.is_nan()) {
                                return acc;
                            }
                            let diff = (*x as f64 * cos_theta_i as f64 + 1.0).ln()
                                - (*y as f64 * cos_theta_i as f64 + 1.0).ln();
                            acc + math::sqr(diff)
                        }) * factor
                    },
                }
            })
            .reduce(|| 0.0, |a, b| a + b);

        #[cfg(not(feature = "cuda"))]
        if is_rmse {
            sum.sqrt()
        } else {
            sum
        }

        #[cfg(feature = "cuda")]
        if gpu_threads_per_block.is_none() || gpu_modules.is_none() {
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
                                if has_nan && (x.is_nan() || y.is_nan()) {
                                    return acc;
                                }
                                let diff = *x as f64 - *y as f64;
                                acc + math::sqr(diff)
                            }) * factor
                        },
                        Weighting::LnCos => {
                            xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                                if has_nan && (x.is_nan() || y.is_nan()) {
                                    return acc;
                                }
                                let diff = (*x as f64 * cos_theta_i as f64 + 1.0).ln()
                                    - (*y as f64 * cos_theta_i as f64 + 1.0).ln();
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
        } else {
            use cust::prelude::*;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

            let n = self.resampled.len();
            let mut d_xs = DeviceBuffer::from_slice(self.resampled.as_slice()).unwrap();
            let mut d_ys = DeviceBuffer::from_slice(other.resampled.as_slice()).unwrap();
            let mut d_diffs = DeviceBuffer::<f32>::zeroed(n).unwrap();

            let threads_per_block = gpu_threads_per_block.unwrap();
            let blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
            let modules = gpu_modules.unwrap();

            unsafe {
                match weighting {
                    Weighting::None => {
                        let func = modules["diff"].get_function("difference_sqr").unwrap();
                        launch!(
                            func<<<(blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                                d_xs.as_device_ptr(),
                                d_ys.as_device_ptr(),
                                d_diffs.as_device_ptr(),
                                n,
                            )
                        )
                        .unwrap();
                    },
                    Weighting::LnCos => {
                        let cos_thetas_i =
                            self.i_thetas.iter().map(|t| t.cos()).collect::<Box<_>>();
                        let d_weights = DeviceBuffer::from_slice(&cos_thetas_i).unwrap();
                        let func = modules["diff"]
                            .get_function("difference_sqr_lncos")
                            .unwrap();
                        launch!(
                            func<<<(blocks, 1, 1), (threads_per_block, 1, 1), 0, stream>>>(
                                d_xs.as_device_ptr(),
                                d_ys.as_device_ptr(),
                                d_weights.as_device_ptr(),
                                d_diffs.as_device_ptr(),
                                stride_theta_i as u32,
                                n,
                            )
                        )
                        .unwrap();
                    },
                }
            }
            stream.synchronize().unwrap();

            #[cfg(feature = "cuda")]
            let sum = if n > 1 << 22 {
                let mut diffs_sqr = vec![0.0; n].into_boxed_slice();
                d_diffs.copy_to(&mut diffs_sqr).unwrap();
                // TODO: update the CUDA kernel to handle large number of samples
                diffs_sqr.iter().fold(0.0, |acc, x| {
                    if x.is_nan() {
                        return acc;
                    }

                    acc + *x as f64 * factor
                })
            } else {
                Self::reduce(&d_diffs, factor as f32, stream, &modules["reduce"]) as f64
            };

            #[cfg(not(feature = "cuda"))]
            let sum = {
                let mut diffs_sqr = vec![0.0; n].into_boxed_slice();
                d_diffs.copy_to(&mut diffs_sqr).unwrap();
                // TODO: update the CUDA kernel to handle large number of samples
                diffs_sqr.iter().fold(0.0, |acc, x| {
                    if x.is_nan() {
                        return acc;
                    }

                    acc + *x as f64 * factor
                })
            };

            if is_rmse {
                sum.sqrt()
            } else {
                sum
            }
        }
    }

    /// Computes the residuals between two BRDF proxies derived from
    /// the same BRDF(may have generated from an analytical BRDF). Stores the
    /// individual residuals in a row-major array following the
    /// shape of the resampled data.
    pub fn residuals(&self, other: &Self, weighting: Weighting, residuals: &mut [f64]) {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );
        assert_eq!(
            residuals.len(),
            self.resampled.len(),
            "The length of residuals must match the length of the resampled data"
        );
        let has_nan = self.has_nan || other.has_nan;
        let stride_theta_i = self.resampled.strides()[0];
        self.resampled
            .as_slice()
            .chunks(stride_theta_i)
            .zip(other.resampled.as_slice().chunks(stride_theta_i))
            .zip(self.i_thetas.iter())
            .zip(residuals.chunks_mut(stride_theta_i))
            .par_bridge()
            .for_each(|(((xs, ys), theta_i), rs)| {
                let cos_theta_i = theta_i.cos();
                xs.iter()
                    .zip(ys.iter())
                    .zip(rs.iter_mut())
                    .for_each(|((x, y), r)| {
                        if has_nan && (x.is_nan() || y.is_nan()) {
                            return;
                        }
                        match weighting {
                            Weighting::None => {
                                *r = *x as f64 - *y as f64;
                            },
                            Weighting::LnCos => {
                                *r = (*x as f64 * cos_theta_i as f64 + 1.0).ln()
                                    - (*y as f64 * cos_theta_i as f64 + 1.0).ln();
                            },
                        }
                    });
            });
    }

    /// Computes the distance between two BRDF proxies derived from the same
    /// BRDF with a filtered range of incident and outgoing angles.
    pub fn distance_filtered(
        &self,
        other: &BrdfProxy,
        metric: ErrorMetric,
        weighting: Weighting,
        max_theta_i: f32,
        max_theta_o: f32,
    ) -> f64 {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );
        let has_nan = self.has_nan || other.has_nan;
        // Find the cutoff indices for theta angles only
        let n_theta_i = self
            .i_thetas
            .as_slice()
            .partition_point(|&x| x < max_theta_i);
        let is_rmse = metric == ErrorMetric::Rmse;

        let sum = match &self.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o, ..
            } => {
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
                                            if has_nan && (x.is_nan() || y.is_nan()) {
                                                continue;
                                            }
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
                o_thetas: theta_o,
                offsets,
                ..
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
                                            if has_nan && (x.is_nan() || y.is_nan()) {
                                                continue;
                                            }
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

    /// Computes the residuals between two BRDF proxies derived from
    /// the same BRDF with a filtered range of incident and outgoing angles.
    /// Stores the individual residuals in a row-major array following the
    /// shape of the resampled data.
    pub fn residuals_filtered(
        &self,
        other: &Self,
        weighting: Weighting,
        residuals: &mut [f64],
        max_theta_i: f32,
        max_theta_o: f32,
    ) {
        assert!(
            self.same_params_p(other),
            "The two BRDFs must have the same parameters"
        );
        let has_nan = self.has_nan || other.has_nan;
        // Find the cutoff indices for theta angles only
        let n_theta_i = self
            .i_thetas
            .as_slice()
            .partition_point(|&x| x < max_theta_i);

        match &self.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o, ..
            } => {
                let n_theta_o = theta_o.as_slice().partition_point(|&x| x < max_theta_o);
                // [Nθ_i, Nφ_i, Nθ_o, Nφ_o, Nλ]
                let shape = self.resampled.shape();
                let filtered_shape = [n_theta_i, shape[1], n_theta_o, shape[3], shape[4]];
                let mut filtered_strides = [0; 5];
                shape::compute_strides(&filtered_shape, &mut filtered_strides, MemLayout::RowMajor);

                let xs = &self.resampled;
                let ys = &other.resampled;

                (0..filtered_shape[0])
                    .into_iter()
                    .zip(self.i_thetas.as_slice().iter())
                    .zip(residuals.chunks_mut(filtered_strides[0]))
                    .par_bridge()
                    .for_each(|((i, theta_i), rs)| {
                        let cos_theta_i = theta_i.cos() as f64;
                        (0..filtered_shape[1]).for_each(|j| {
                            for k in 0..filtered_shape[2] {
                                for l in 0..filtered_shape[3] {
                                    for m in 0..filtered_shape[4] {
                                        let x = xs[[i, j, k, l, m]] as f64;
                                        let y = ys[[i, j, k, l, m]] as f64;
                                        if has_nan && (x.is_nan() || y.is_nan()) {
                                            continue;
                                        }
                                        let idx = shape::compute_index_from_strides(
                                            &[j, k, l, m],
                                            &filtered_strides[1..],
                                        );
                                        rs[idx] = match weighting {
                                            Weighting::None => x - y,
                                            Weighting::LnCos => {
                                                let ln_x_cos = (x * cos_theta_i + 1.0).ln();
                                                let ln_y_cos = (y * cos_theta_i + 1.0).ln();
                                                ln_x_cos - ln_y_cos
                                            },
                                        };
                                    }
                                }
                            }
                        })
                    })
            },
            OutgoingDirs::List {
                o_thetas: theta_o,
                offsets,
                ..
            } => {
                let n_theta_o = theta_o.as_slice().partition_point(|&x| x < max_theta_o);
                // The total number of outgoing directions after filtering
                let n_wo = offsets[n_theta_o];
                // [Nθ_i, Nφ_i, Nω_o, Nλ]
                let shape = self.resampled.shape();
                let filtered_shape = [n_theta_i, shape[1], n_wo, shape[3]];
                let mut filtered_strides = [0; 4];
                shape::compute_strides(&filtered_shape, &mut filtered_strides, MemLayout::RowMajor);

                let xs = &self.resampled;
                let ys = &other.resampled;

                (0..filtered_shape[0])
                    .into_iter()
                    .zip(self.i_thetas.as_slice().iter())
                    .zip(residuals.chunks_mut(filtered_strides[0]))
                    .par_bridge()
                    .for_each(|((i, theta_i), rs)| {
                        let cos_theta_i = theta_i.cos();
                        (0..filtered_shape[1]).for_each(|j| {
                            for k in 0..n_theta_o {
                                for l in offsets[k]..offsets[k + 1] {
                                    for m in 0..filtered_shape[3] {
                                        let x = xs[[i, j, l, m]];
                                        let y = ys[[i, j, l, m]];
                                        if has_nan && (x.is_nan() || y.is_nan()) {
                                            continue;
                                        }
                                        let idx = shape::compute_index_from_strides(
                                            &[j, l, m],
                                            &filtered_strides[1..],
                                        );
                                        rs[idx] = match weighting {
                                            Weighting::None => x as f64 - y as f64,
                                            Weighting::LnCos => {
                                                let ln_x_cos =
                                                    ((x * cos_theta_i + 1.0) as f64).ln();
                                                let ln_y_cos =
                                                    ((y * cos_theta_i + 1.0) as f64).ln();
                                                ln_x_cos - ln_y_cos
                                            },
                                        };
                                    }
                                }
                            }
                        })
                    })
            },
        };
    }

    // TODO: potentially considering generating only the data points for the
    // filtered incident and outgoing angles
    /// Generate the data points following the same incident and outgoing
    /// angles for the given analytical BRDF.
    pub fn generate_analytical(&self, model: &dyn AnalyticalBrdf<[f64; 2]>) -> Self {
        let n_spectrum = self.spectrum.len();
        let mut resampled = if self.has_nan {
            DynArr::splat(f32::NAN, self.resampled.shape())
        } else {
            DynArr::zeros(self.resampled.shape())
        };
        let i_thetas = self.i_thetas.as_slice();
        let i_phis = self.i_phis.as_slice();
        let strides = self.resampled.strides();

        match &self.o_dirs {
            OutgoingDirs::Grid { o_thetas, o_phis } => {
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
                                    .zip(o_thetas.as_slice().iter())
                                    .for_each(|(per_theta_o, theta_o)| {
                                        per_theta_o
                                            .chunks_mut(strides[3])
                                            .zip(o_phis.as_slice().iter())
                                            .for_each(|(per_phi_o, phi_o)| {
                                                let vo = Sph2::new(rad!(*theta_o), rad!(*phi_o))
                                                    .to_cartesian();
                                                let spectral_samples =
                                                    Scattering::eval_reflectance_spectrum(
                                                        model,
                                                        &vi,
                                                        &vo,
                                                        &self.iors_i,
                                                        &self.iors_t,
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
                o_thetas,
                o_phis,
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
                                for (i, theta_o) in o_thetas.iter().enumerate() {
                                    for phi_o in o_phis[offsets[i]..offsets[i + 1]].iter() {
                                        let vo =
                                            Sph2::new(rad!(*theta_o), rad!(*phi_o)).to_cartesian();
                                        let spectral_samples =
                                            Scattering::eval_reflectance_spectrum(
                                                model,
                                                &vi,
                                                &vo,
                                                &self.iors_i,
                                                &self.iors_t,
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

        BrdfProxy {
            has_nan: false,
            source: ProxySource::Analytical,
            brdf: self.brdf,
            spectrum: self.spectrum.clone(),
            i_thetas: self.i_thetas.clone(),
            i_phis: self.i_phis.clone(),
            o_dirs: self.o_dirs.clone(),
            resampled: Cow::Owned(resampled),
            iors_i: self.iors_i.clone(),
            iors_t: self.iors_t.clone(),
        }
    }

    #[cfg(feature = "cuda")]
    fn reduce(arr: &DeviceBuffer<f32>, factor: f32, stream: Stream, module: &Module) -> f32 {
        let block_size = 256;
        let n = arr.len();
        let blocks = {
            let numer = n + (block_size as usize * 2) - 1;
            let denom = block_size as usize * 2;
            max(1, min((numer / denom) as u32, 65_535))
        };

        let shared_mem = block_size * size_of::<f32>() as u32;
        let mut block_sums: DeviceBuffer<f64> =
            unsafe { DeviceBuffer::uninitialized(blocks as usize).unwrap() };

        unsafe {
            let kernel = module.get_function("reduce_blocks").unwrap();
            launch!(
                kernel<<<(blocks, 1, 1), (block_size, 1, 1), shared_mem, stream>>>(
                    arr.as_device_ptr(),
                    block_sums.as_device_ptr(),
                    n
                )
            )
            .unwrap();
        }

        let shared_mem2 = block_size * size_of::<f64>() as u32;
        let mut d_sum: DeviceBuffer<f64> = unsafe { DeviceBuffer::uninitialized(1).unwrap() };

        unsafe {
            let k2 = module.get_function("reduce").unwrap();
            launch!(
                k2<<<(1, 1, 1), (block_size, 1, 1), shared_mem2, stream>>>(
                    block_sums.as_device_ptr(),  // const double* block_sums
                    d_sum.as_device_ptr(),       // double* result
                    blocks as usize,             // size_t num_blocks
                    factor as f64                // double factor
                )
            )
            .unwrap();
        }

        stream.synchronize().unwrap();

        let mut sum = [0.0f64; 1];
        d_sum.copy_to(&mut sum).unwrap();
        sum[0] as f32
    }
}

fn sample_difference_sum(
    xs: &[f32],
    ys: &[f32],
    stride_theta_i: usize,
    factor: f64,
    weighting: Weighting,
    has_nan: bool,
    i_thetas: Option<&[f32]>,
) -> f64 {
    assert_eq!(xs.len(), ys.len());

    match weighting {
        Weighting::None => xs
            .chunks(stride_theta_i)
            .zip(ys.chunks(stride_theta_i))
            .par_bridge()
            .map(|(xs, ys)| {
                xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                    if has_nan && (x.is_nan() || y.is_nan()) {
                        return acc;
                    }
                    let diff = *x as f64 - *y as f64;
                    acc + math::sqr(diff)
                }) * factor
            })
            .reduce(|| 0.0, |a, b| a + b),
        Weighting::LnCos => {
            let i_thetas = i_thetas.expect("i_thetas must be provided for LnCos weighting");
            assert_eq!(xs.len(), i_thetas.len());

            xs.chunks(stride_theta_i)
                .zip(ys.chunks(stride_theta_i))
                .zip(i_thetas.iter())
                .par_bridge()
                .map(|((xs, ys), theta_i)| {
                    let cos_theta_i = theta_i.cos();
                    xs.iter().zip(ys.iter()).fold(0.0, |acc, (x, y)| {
                        if has_nan && (x.is_nan() || y.is_nan()) {
                            return acc;
                        }
                        let diff = (*x as f64 * cos_theta_i as f64 + 1.0).ln()
                            - (*y as f64 * cos_theta_i as f64 + 1.0).ln();
                        acc + math::sqr(diff)
                    }) * factor
                })
                .reduce(|| 0.0, |a, b| a + b)
        },
    }
}

#[cfg(test)]
mod distance_tests {
    use super::{BrdfProxy, OutgoingDirs, ProxySource};
    use crate::{brdf::measured::MeasuredBrdfKind, AnyMeasuredBrdf};
    use std::borrow::Cow;
    use vgn_core::{
        optics::{Ior, IorReg},
        units::{nm, Nanometres},
        utils::medium::MediumId,
        ErrorMetric, Weighting,
    };
    use vgn_jabr::array::{DyArr, DynArr};

    struct TestMeasuredBrdf {
        spectrum: Box<[Nanometres]>,
    }

    impl AnyMeasuredBrdf for TestMeasuredBrdf {
        fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Unknown }

        fn spectrum(&self) -> &[Nanometres] { &self.spectrum }

        fn transmitted_medium(&self) -> MediumId { MediumId::AIR }

        fn incident_medium(&self) -> MediumId { MediumId::AIR }

        fn proxy(&self, _: &IorReg) -> BrdfProxy<'_> { unreachable!("not used in test helper") }

        fn as_any(&self) -> &dyn std::any::Any { self }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    }

    fn make_proxy(samples: &[f32], has_nan: bool) -> BrdfProxy<'static> {
        let measured = Box::leak(Box::new(TestMeasuredBrdf {
            spectrum: vec![nm!(550.0)].into_boxed_slice(),
        }));
        let i_thetas = DyArr::from_vec_1d(vec![0.2f32]);
        let i_phis = DyArr::from_vec_1d(vec![0.0f32]);
        let o_thetas = DyArr::from_vec_1d(vec![0.1f32, 0.4f32]);
        let o_phis = DyArr::from_vec_1d(vec![0.0f32, 1.0f32]);
        let resampled = DynArr::from_vec(&[1, 1, 2, 2, 1], samples.to_vec());
        let iors_i = vec![Ior::new_dielectric(1.0)];
        let iors_t = vec![Ior::new_dielectric(1.5)];

        BrdfProxy::new(
            has_nan,
            ProxySource::Measured,
            measured,
            Cow::Owned(i_thetas),
            Cow::Owned(i_phis),
            OutgoingDirs::new_grid(Cow::Owned(o_thetas), Cow::Owned(o_phis)),
            Cow::Owned(resampled),
            Cow::Owned(iors_i),
            Cow::Owned(iors_t),
        )
    }

    fn distance(a: &BrdfProxy, b: &BrdfProxy, metric: ErrorMetric, weighting: Weighting) -> f64 {
        #[cfg(feature = "cuda")]
        {
            a.distance(b, metric, weighting, None, None)
        }
        #[cfg(not(feature = "cuda"))]
        {
            a.distance(b, metric, weighting)
        }
    }

    #[test]
    fn distance_identical_proxies_is_zero() {
        let proxy = make_proxy(&[0.1, 0.2, 0.3, 0.4], false);
        let d = distance(&proxy, &proxy, ErrorMetric::Mse, Weighting::None);
        assert_eq!(d, 0.0, "distance should be zero for identical proxies");
    }

    #[test]
    fn distance_all_nan_samples_is_zero() {
        let a = make_proxy(&[f32::NAN, f32::NAN, f32::NAN, f32::NAN], true);
        let b = make_proxy(&[f32::NAN, f32::NAN, f32::NAN, f32::NAN], true);
        let d_mse = distance(&a, &b, ErrorMetric::Mse, Weighting::None);
        let d_rmse = distance(&a, &b, ErrorMetric::Rmse, Weighting::None);
        assert_eq!(
            d_mse, 0.0,
            "MSE distance should be zero when no valid samples exist"
        );
        assert_eq!(
            d_rmse, 0.0,
            "RMSE distance should be zero when no valid samples exist"
        );
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod tests {
    use approx;
    use cust::{
        context::ContextFlags, device::DeviceAttribute, launch, memory::DeviceBuffer, prelude::*,
        stream::Stream,
    };
    use vgn_core::cuda::{init_cuda_context, load_ptx_modules};

    #[test]
    fn test_array_difference() {
        let (_context, device) = init_cuda_context(ContextFlags::SCHED_AUTO).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let n = 1 << 24;
        let xs = (0..n).map(|x| x as f32).collect::<Box<_>>();
        let ys = (0..n).map(|x| x as f32 + 1.0).collect::<Box<_>>();

        let d_xs = DeviceBuffer::from_slice(&xs).unwrap();
        let d_ys = DeviceBuffer::from_slice(&ys).unwrap();
        let d_diffs = DeviceBuffer::<f32>::zeroed(n).unwrap();

        let block_size = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap() as u32;
        let blocks = (n as u32 + block_size - 1) / block_size;

        let modules = load_ptx_modules().unwrap();

        // Test the squared difference kernel
        unsafe {
            let kernel = modules["diff"].get_function("difference_sqr").unwrap();
            launch!(
                kernel<<<(blocks, 1, 1), (block_size, 1, 1), 0, stream>>>(
                    d_xs.as_device_ptr(),
                    d_ys.as_device_ptr(),
                    d_diffs.as_device_ptr(),
                    n,
                )
            )
            .unwrap();
        }

        stream.synchronize().unwrap();

        let mut diffs: Vec<f32> = vec![0.0; n];
        d_diffs.copy_to(diffs.as_mut_slice()).unwrap();

        for i in 0..n {
            assert_eq!(diffs[i], 1.0, "at index {}", i);
        }

        // Test the squared difference kernel with vec4
        unsafe {
            let kernel = modules["diff"].get_function("difference_sqr_vec4").unwrap();
            launch!(
                kernel<<<(blocks, 1, 1), (block_size, 1, 1), 0, stream>>>(
                    d_xs.as_device_ptr(),
                    d_ys.as_device_ptr(),
                    d_diffs.as_device_ptr(),
                    n / 4,
                )
            )
            .unwrap();
        }

        stream.synchronize().unwrap();

        let mut diffs: Vec<f32> = vec![0.0; n];
        d_diffs.copy_to(diffs.as_mut_slice()).unwrap();

        for i in 0..n {
            assert_eq!(diffs[i], 1.0, "at index {}", i);
        }

        // Test the squared difference kernel with weighting
        let weights = (0..n / 256).map(|x| (x as f32).cos()).collect::<Box<_>>();
        let d_weights = DeviceBuffer::from_slice(&weights).unwrap();

        unsafe {
            let kernel = modules["diff"]
                .get_function("difference_sqr_lncos")
                .unwrap();
            launch!(
                kernel<<<(blocks, 1, 1), (block_size, 1, 1), 0, stream>>>(
                    d_xs.as_device_ptr(),
                    d_ys.as_device_ptr(),
                    d_weights.as_device_ptr(),
                    d_diffs.as_device_ptr(),
                    256u32,
                    n,
                )
            )
            .unwrap();
        }

        stream.synchronize().unwrap();

        let mut diffs: Vec<f32> = vec![0.0; n];
        d_diffs.copy_to(diffs.as_mut_slice()).unwrap();

        for i in 0..n {
            let x = xs[i] as f64;
            let y = ys[i] as f64;
            let w = weights[i / 256] as f64;
            let diff = ((x * w + 1.0).ln() - (y * w + 1.0).ln()).powi(2);
            if diff.is_nan() && diffs[i].is_nan() {
                continue;
            }

            approx::assert_ulps_eq!(diffs[i], diff as f32);
        }
    }

    #[test]
    fn test_reduction() {
        let (_context, _device) = init_cuda_context(ContextFlags::SCHED_AUTO).unwrap();
        let modules = load_ptx_modules().unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        let n = 1 << 26;
        let xs = (0..n)
            .map(|x| {
                if x == 0 || x == 2 || x == 4 {
                    f32::NAN
                } else {
                    1.0f32
                }
            })
            .collect::<Box<_>>();

        let d_xs = DeviceBuffer::from_slice(&xs).unwrap();

        let sum = super::BrdfProxy::reduce(&d_xs, 0.5, stream, &modules["reduce"]);

        // Expected: NaNs treated as zero => (n-3) ones, scaled by 0.5
        let expected = (n as f32 - 3.0) * 0.5;
        assert_eq!(sum, expected, "sum={} expected={}", sum, expected);
    }
}

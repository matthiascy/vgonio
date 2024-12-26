use crate::{
    brdf::{Bxdf, BxdfFamily},
    distro::{MicrofacetDistribution, MicrofacetDistroKind},
};
use base::{range::StepRangeIncl, units::Radians, ErrorMetric, Isotropy, Weighting};
use levenberg_marquardt::MinimizationReport;
use std::fmt::Debug;

/// Types of the fitting problem.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FittingProblemKind {
    /// Fitting the microfacet distribution related model.
    Mfd {
        /// The target microfacet distribution model.
        model: MicrofacetDistroKind,
        /// The isotropy of the model.
        isotropy: Isotropy,
    },
    /// Fitting the bidirectional scattering distribution function.
    Bxdf {
        /// The target BxDF model family.
        family: BxdfFamily,
        /// The target microfacet distribution model in case of microfacet-based
        /// BxDFs.
        distro: Option<MicrofacetDistroKind>,
        /// The isotropy of the model.
        isotropy: Isotropy,
    },
}

/// A model after fitting.
#[derive(Debug, Clone)]
pub enum FittedModel {
    /// Bidirectional scattering distribution function.
    Bsdf(Box<dyn Bxdf<Params = [f64; 2]>>),
    /// Microfacet area distribution function with the scaling factor applied to
    /// the measured data.
    Ndf(Box<dyn MicrofacetDistribution<Params = [f64; 2]>>, f32),
    /// Microfacet Masking-shadowing function.
    Msf(Box<dyn MicrofacetDistribution<Params = [f64; 2]>>),
}

impl FittedModel {
    /// Returns the isotropy of the model.
    pub fn isotropy(&self) -> Isotropy {
        match self {
            FittedModel::Bsdf(model) => model.isotropy(),
            FittedModel::Msf(model) | FittedModel::Ndf(model, _) => model.isotropy(),
        }
    }

    /// Returns the kind of fitting problem.
    pub fn kind(&self) -> FittingProblemKind {
        match self {
            FittedModel::Bsdf(model) => FittingProblemKind::Bxdf {
                family: model.family(),
                distro: model.distro(),
                isotropy: model.isotropy(),
            },
            FittedModel::Msf(model) => FittingProblemKind::Mfd {
                model: model.kind(),
                isotropy: model.isotropy(),
            },
            FittedModel::Ndf(model, _) => FittingProblemKind::Mfd {
                model: model.kind(),
                isotropy: model.isotropy(),
            },
        }
    }

    /// Returns the scaling factor applied to the measured NDF.
    pub fn scale(&self) -> Option<f32> {
        match self {
            FittedModel::Ndf(_, scale) => Some(*scale),
            _ => None,
        }
    }
}

/// Report of a fitting process.
pub struct FittingReport<M> {
    /// Index of the best model found.
    best: Option<usize>,
    /// The reports of the fitting process. Includes the model and the
    /// minimization report with different initial values.
    pub reports: Box<[(M, MinimizationReport<f64>)]>,
}

impl<M> FittingReport<M> {
    /// Creates a new fitting report from the results of the fitting process.
    pub fn new(results: Box<[(M, MinimizationReport<f64>)]>) -> Self {
        let mut reports = results;
        reports.sort_by(|(_, r1), (_, r2)| {
            r1.objective_function
                .partial_cmp(&r2.objective_function)
                .unwrap()
        });
        if reports.is_empty() {
            return FittingReport {
                best: None,
                reports,
            };
        }
        FittingReport {
            best: Some(0),
            reports,
        }
    }

    /// Creates an empty fitting report.
    pub fn empty() -> Self {
        FittingReport {
            best: None,
            reports: Box::new([]),
        }
    }

    /// Returns the best model found.
    pub fn best_model(&self) -> Option<&M> { self.best.map(|i| &self.reports[i].0) }

    /// Returns the report of the best model found.
    pub fn best_model_report(&self) -> Option<&(M, MinimizationReport<f64>)> {
        self.best.map(|i| &self.reports[i])
    }

    /// Log the fitting report.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of best models to log.
    pub fn print_fitting_report(&self, n: usize)
    where
        M: Debug,
    {
        println!("Fitting reports (first {}):", n);
        for (m, r) in self.reports.iter().take(n) {
            println!(
                "    - Model: {:?}, objective_function: {}",
                m, r.objective_function
            );
        }
        println!(
            "  Best model: {:?}, Err: {}",
            self.best_model(),
            self.best_model_report().unwrap().1.objective_function
        );
    }
}

/// A fitting problem.
pub trait FittingProblem {
    /// The model to fit.
    type Model;

    /// Non-linear least squares fitting using Levenberg-Marquardt algorithm.
    fn nllsq_fit(
        &self,
        target: MicrofacetDistroKind,
        isotropy: Isotropy,
        weighting: Weighting,
        initial: StepRangeIncl<f64>,
        max_theta_i: Option<Radians>,
        max_theta_o: Option<Radians>,
    ) -> FittingReport<Self::Model>;

    /// Brute force fitting.
    ///
    /// Note: Currently only *isotropic* models are supported.
    ///
    /// # Arguments
    ///
    /// * `metric` - The error metric to use.
    /// * `weighting` - The weighting to use.
    /// * `max_theta_i` - The maximum incident angle to consider.
    /// * `max_theta_o` - The maximum outgoing angle to consider.
    /// * `precision` - The number of digits after the decimal point to
    ///   consider.
    fn brute_fit(
        &self,
        target: MicrofacetDistroKind,
        metric: ErrorMetric,
        weighting: Weighting,
        max_theta_i: Option<Radians>,
        max_theta_o: Option<Radians>,
        precision: u32,
    ) -> FittingReport<Self::Model>;
}

/// Fitting for BRDFs.
pub mod brdf {
    /// Fitting for BRDFs using brute force.
    pub mod brute;
    /// Fitting for BRDFs using non-linear least squares.
    pub mod nllsq;

    use std::borrow::Cow;

    use base::{
        math::{self, Sph2},
        optics::ior::{Ior, IorRegistry},
        range::StepRangeIncl,
        units::{rad, Nanometres, Radians},
        ErrorMetric, Isotropy, MeasuredBrdfKind, Weighting,
    };
    use brute::compute_distance_between_measured_and_modelled;
    use jabr::array::{
        shape::{compute_index_from_strides, compute_strides},
        DyArr, DynArr, MemLayout,
    };
    use levenberg_marquardt::{LevenbergMarquardt, MinimizationReport, TerminationReason};
    use nllsq::{init_microfacet_brdf_models, NllsqBrdfFittingProxy};
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };

    use crate::{
        brdf::{
            analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR},
            Bxdf,
        },
        distro::MicrofacetDistroKind,
        Scattering,
    };

    use super::{FittingProblem, FittingReport};

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
    /// 1. As a cartesian product of theta and phi angles, where directions are
    ///    computed by iterating over all combinations of theta_o and phi_o
    /// 2. As an explicit list of directions, where each direction is specified
    ///    directly
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
        pub fn new_grid(o_thetas: Cow<'a, DyArr<f32>>, o_phis: Cow<'a, DyArr<f32>>) -> Self {
            Self::Grid { o_thetas, o_phis }
        }

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
    impl<'a> PartialEq for OutgoingDirs<'a> {
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

    /// A proxy for a BRDF that can be fitted analytically.
    ///
    /// This is useful when the stored raw BRDF data is in compressed form and
    /// cannot be directly used for fitting. The proxy can be used to store
    /// the resampled BRDF data to be used for fitting.
    ///
    /// The proxy cannot be constructed directly, but must be created by calling
    /// the `proxy` method on the measured BRDF that implements the
    /// `AnalyticalFit` trait.
    pub struct BrdfFittingProxy<'a, Brdf>
    where
        Brdf: AnalyticalFit,
    {
        /// Indicates if the proxy has NaN values.
        pub(crate) has_nan: bool,
        /// The source of the proxy.
        pub(crate) source: ProxySource,
        /// The raw BRDF data that the proxy is associated with.
        pub(crate) brdf: &'a Brdf,
        /// Incident angles (polar angle) in radians of the resampled BRDF data.
        pub(crate) i_thetas: Cow<'a, DyArr<f32>>,
        /// Incident angles (azimuthal angle) in radians of the resampled BRDF
        /// data.
        pub(crate) i_phis: Cow<'a, DyArr<f32>>,
        /// Outgoing directions of the resampled BRDF data.
        pub(crate) o_dirs: OutgoingDirs<'a>,
        /// The resampled BRDF data that can be used for fitting. Depends on the
        /// outgoing directions. The shape of the array could be
        /// - [Nθ_i, Nφ_i, Nθ_o, Nφ_o, Nλ] in row-major order
        /// - [Nθ_i, Nφ_i, Nω_o, Nλ] in row-major order
        pub(crate) resampled: Cow<'a, DynArr<f32>>,
        /// Precomputed IORs for the incident medium.
        pub(crate) iors_i: Cow<'a, [Ior]>,
        /// Precomputed IORs for the transmitted medium.
        pub(crate) iors_t: Cow<'a, [Ior]>,
    }

    /// A trait for BRDFs that can be fitted with an analytical model.
    pub trait AnalyticalFit: Sync {
        /// Create a proxy for this BRDF.
        fn proxy(&self, iors: &IorRegistry) -> BrdfFittingProxy<Self>
        where
            Self: Sized;

        /// Returns the wavelengths at which the BRDF is measured.
        fn spectrum(&self) -> &[Nanometres];

        /// Returns the kind of the measured BRDF.
        fn kind(&self) -> MeasuredBrdfKind;
    }

    impl<Brdf> BrdfFittingProxy<'_, Brdf>
    where
        Brdf: AnalyticalFit,
    {
        /// Returns the source of the proxy.
        pub fn source(&self) -> ProxySource { self.source }

        /// Returns the resampled BRDF data.
        pub fn samples(&self) -> &[f32] { self.resampled.as_slice() }

        /// Returns the shape of the resampled BRDF data after filtering.
        /// Depends on the outgoing directns, the dimensions could be 4 or 5.
        pub fn filtered_shape(&self, max_theta_i: f32, max_theta_o: f32) -> Box<[usize]> {
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
        fn same_params_p<O: AnalyticalFit>(&self, other: &BrdfFittingProxy<O>) -> bool {
            self.i_thetas == other.i_thetas
                && self.i_phis == other.i_phis
                && self.o_dirs == other.o_dirs
                && self.brdf.spectrum() == other.brdf.spectrum()
                && self.resampled.shape() == other.resampled.shape()
        }

        /// Computes the distance between two BRDF poxies derived from the same
        /// BRDF.
        fn distance<O: AnalyticalFit>(
            &self,
            other: &BrdfFittingProxy<O>,
            metric: ErrorMetric,
            weighting: Weighting,
        ) -> f64 {
            assert!(
                self.same_params_p(other),
                "The two BRDFs must have the same parameters"
            );
            let has_nan = self.has_nan || other.has_nan;
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

        /// Computes the residuals between two BRDF proxies derived from the
        /// the same BRDF(may generated from an analytical BRDF). Stores the
        /// individual residuals in a row-major array following the
        /// shape of the resampled data.
        fn residuals(&self, other: &Self, weighting: Weighting, residuals: &mut [f64]) {
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
                                    let diff = *x as f64 - *y as f64;
                                    *r = diff;
                                },
                                Weighting::LnCos => {
                                    let diff = ((x * cos_theta_i + 1.0) as f64).ln()
                                        - ((y * cos_theta_i + 1.0) as f64).ln();
                                    *r = diff;
                                },
                            }
                        });
                });
        }

        /// Computes the distance between two BRDF poxies derived from the same
        /// BRDF with a filtered range of incident and outgoing angles.
        fn distance_filtered<O: AnalyticalFit + Sync>(
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

        /// Computes the residuals between two BRDF proxies derived from the
        /// the same BRDF with a filtered range of incident and outgoing angles.
        /// Stores the individual residuals in a row-major array following the
        /// shape of the resampled data.
        fn residuals_filtered(
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
                    compute_strides(&filtered_shape, &mut filtered_strides, MemLayout::RowMajor);

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
                                            let idx = compute_index_from_strides(
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
                    compute_strides(&filtered_shape, &mut filtered_strides, MemLayout::RowMajor);

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
                                            let idx = compute_index_from_strides(
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
        fn generate_analytical(&self, model: &dyn Bxdf<Params = [f64; 2]>) -> Self {
            let n_spectrum = self.brdf.spectrum().len();
            let mut resampled = DynArr::zeros(self.resampled.shape());
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
                                                    let vo =
                                                        Sph2::new(rad!(*theta_o), rad!(*phi_o))
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
                has_nan: false,
                source: ProxySource::Analytical,
                brdf: self.brdf,
                i_thetas: self.i_thetas.clone(),
                i_phis: self.i_phis.clone(),
                o_dirs: self.o_dirs.clone(),
                resampled: Cow::Owned(resampled),
                iors_i: self.iors_i.clone(),
                iors_t: self.iors_t.clone(),
            }
        }
    }

    impl<'a, Brdf: AnalyticalFit> FittingProblem for BrdfFittingProxy<'a, Brdf> {
        type Model = Box<dyn Bxdf<Params = [f64; 2]>>;

        fn nllsq_fit(
            &self,
            target: MicrofacetDistroKind,
            isotropy: Isotropy,
            weighting: Weighting,
            initial: StepRangeIncl<f64>,
            max_theta_i: Option<Radians>,
            max_theta_o: Option<Radians>,
        ) -> FittingReport<Self::Model> {
            let solver = LevenbergMarquardt::new();
            let cpu_count = (std::thread::available_parallelism().unwrap().get() / 2).max(1);
            let tasks = init_microfacet_brdf_models(initial, target, isotropy);
            let tasks_per_cpu = tasks.len().div_ceil(cpu_count);
            log::debug!(
                "Solve {} models, {} per CPU, {} CPUs",
                tasks.len(),
                tasks_per_cpu,
                cpu_count
            );
            let results = tasks
                .par_chunks(tasks_per_cpu)
                .flat_map(|models| {
                    models
                        .iter()
                        .filter_map(|model| {
                            let (fitted_model, report) = match isotropy {
                                Isotropy::Isotropic => {
                                    let nllsq_proxy = NllsqBrdfFittingProxy::<
                                        '_,
                                        _,
                                        { Isotropy::Isotropic },
                                    >::new(
                                        &self,
                                        model.clone(),
                                        weighting,
                                        max_theta_i,
                                        max_theta_o,
                                    );
                                    let (result, report) = solver.minimize(nllsq_proxy);
                                    (result.model, report)
                                },
                                Isotropy::Anisotropic => {
                                    let nllsq_proxy = NllsqBrdfFittingProxy::<
                                        '_,
                                        _,
                                        { Isotropy::Anisotropic },
                                    >::new(
                                        &self,
                                        model.clone(),
                                        weighting,
                                        max_theta_i,
                                        max_theta_o,
                                    );
                                    let (result, report) = solver.minimize(nllsq_proxy);
                                    (result.model, report)
                                },
                            };

                            match report.termination {
                                TerminationReason::Converged { .. }
                                | TerminationReason::LostPatience => Some((fitted_model, report)),
                                _ => None,
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Box<[_]>>();
            FittingReport::new(results)
        }

        /// Fit coarsely within the initial range then refine the fit within
        /// the range of the best fit. Repeat until the precision is reached.
        fn brute_fit(
            &self,
            target: MicrofacetDistroKind,
            metric: ErrorMetric,
            weighting: Weighting,
            max_theta_i: Option<Radians>,
            max_theta_o: Option<Radians>,
            precision: u32,
        ) -> FittingReport<Self::Model> {
            let cpu_count = ((std::thread::available_parallelism().unwrap().get()) / 2).max(1);
            let mut step_size = 0.01;
            let mut alphas = StepRangeIncl::new(0.0, 1.0, step_size)
                .values()
                .collect::<Box<[f64]>>();
            let mut errs = vec![f64::NAN; 256].into_boxed_slice();
            // The number of iterations as each iteration has two precision
            let n_times = (precision + 1) / 2;
            let mut records = Vec::with_capacity(n_times as usize * 256);

            for _ in 0..n_times {
                let chunk_size = alphas.len().div_ceil(cpu_count);
                alphas
                    .chunks(chunk_size)
                    .zip(errs.chunks_mut(chunk_size))
                    .par_bridge()
                    .for_each(|(alpha_chunks, err_chunks)| {
                        err_chunks
                            .iter_mut()
                            .zip(alpha_chunks.iter())
                            .for_each(|(err, alpha)| {
                                *err = compute_distance_between_measured_and_modelled(
                                    &self,
                                    target,
                                    metric,
                                    weighting,
                                    *alpha,
                                    *alpha,
                                    max_theta_i.unwrap_or(Radians::HALF_PI),
                                    max_theta_o.unwrap_or(Radians::HALF_PI),
                                );
                            });
                    });
                // Record the error and alpha
                errs.iter().zip(alphas.iter()).for_each(|(err, alpha)| {
                    if err.is_nan() {
                        return;
                    }
                    let m = match target {
                        MicrofacetDistroKind::Beckmann => {
                            Box::new(MicrofacetBrdfBK::new(*alpha, *alpha))
                                as Box<dyn Bxdf<Params = [f64; 2]>>
                        },
                        MicrofacetDistroKind::TrowbridgeReitz => {
                            Box::new(MicrofacetBrdfTR::new(*alpha, *alpha))
                                as Box<dyn Bxdf<Params = [f64; 2]>>
                        },
                    };
                    records.push((
                        m,
                        MinimizationReport {
                            termination: TerminationReason::User("brute force"),
                            number_of_evaluations: 1,
                            objective_function: *err,
                        },
                    ));
                });
                // Find the range of the best fit
                let min_err = errs.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                let min_err_idx = errs.iter().position(|&x| x == min_err).unwrap();
                let min_err_alpha = alphas[min_err_idx];
                // Refine the range of the best fit
                alphas = StepRangeIncl::new(
                    (min_err_alpha - step_size).max(0.0),
                    (min_err_alpha + step_size).min(1.0),
                    step_size * 0.01,
                )
                .values()
                .collect::<Box<[f64]>>();
                errs.fill(f64::NAN);
                step_size = step_size * 0.01;
            }

            // Convert the records to FittingReport
            FittingReport::new(records.into_boxed_slice())
        }
    }
}

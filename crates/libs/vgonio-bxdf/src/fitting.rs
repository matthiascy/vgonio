//! Types and structures for BRDF fitting.
use levenberg_marquardt::{MinimizationReport, TerminationReason};
use std::fmt::Debug;
use vgonio_core::{
    bxdf::{AnalyticalBrdf, BrdfFamily, MicrofacetDistribution, MicrofacetDistroKind},
    math::rcp_f64,
    units::Radians,
    utils::range::StepRangeIncl,
    ErrorMetric, Symmetry, Weighting,
};

/// Types of the fitting problem.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FittingProblemKind {
    /// Fitting the microfacet distribution related model.
    Mfd {
        /// The target microfacet distribution model.
        model: MicrofacetDistroKind,
        /// The symmetry of the model.
        symmetry: Symmetry,
    },
    /// Fitting the bidirectional scattering distribution function.
    Bxdf {
        /// The target BxDF model family.
        family: BrdfFamily,
        /// The target microfacet distribution model in case of microfacet-based
        /// BxDFs.
        distro: Option<MicrofacetDistroKind>,
        /// The symmetry of the model.
        symmetry: Symmetry,
    },
}

/// A model after fitting.
#[derive(Debug, Clone)]
pub enum FittedModel {
    /// Bidirectional scattering distribution function.
    Bsdf(Box<dyn AnalyticalBrdf<Params = [f64; 2]>>),
    /// Microfacet area distribution function with the scaling factor applied to
    /// the measured data.
    Ndf(Box<dyn MicrofacetDistribution<Params = [f64; 2]>>, f32),
    /// Microfacet Masking-shadowing function.
    Msf(Box<dyn MicrofacetDistribution<Params = [f64; 2]>>),
}

impl FittedModel {
    /// Returns the symmetry of the model.
    pub fn symmetry(&self) -> Symmetry {
        match self {
            FittedModel::Bsdf(model) => model.symmetry(),
            FittedModel::Msf(model) | FittedModel::Ndf(model, _) => model.symmetry(),
        }
    }

    /// Returns the kind of fitting problem.
    pub fn kind(&self) -> FittingProblemKind {
        match self {
            FittedModel::Bsdf(model) => FittingProblemKind::Bxdf {
                family: model.family(),
                distro: model.distro(),
                symmetry: model.symmetry(),
            },
            FittedModel::Msf(model) => FittingProblemKind::Mfd {
                model: model.kind(),
                symmetry: model.symmetry(),
            },
            FittedModel::Ndf(model, _) => FittingProblemKind::Mfd {
                model: model.kind(),
                symmetry: model.symmetry(),
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

/// Report of a minimisation process.
#[derive(Debug)]
pub struct MinimisationReport {
    /// The number of data points used in the fitting process (excluding NaN
    /// values) later used for computing the error metric.
    pub n_data_points: usize,
    /// Error metric used for the fitting.
    pub error_metric: ErrorMetric,
    /// The objective function value which is the same as the error metric
    /// value.
    pub objective_fn: f64,
    /// The number of iterations performed.
    pub n_iteration: usize,
    /// The reason for termination.
    pub termination: TerminationReason,
}

impl MinimisationReport {
    /// Creates a new minimisation report from the results of the levenberg
    /// marquardt minimisation process.
    pub fn from_lm_nllsq(report: MinimizationReport<f64>, n_data_points: usize) -> Self {
        MinimisationReport {
            n_data_points,
            error_metric: ErrorMetric::Nllsq,
            objective_fn: report.objective_function,
            n_iteration: report.number_of_evaluations,
            termination: report.termination,
        }
    }

    /// Creates a new minimisation report from the results of the brute force
    /// fitting process.
    pub fn from_brute_force(
        error_metric: ErrorMetric,
        objective_fn: f64,
        n_data_points: usize,
        n_iteration: usize,
    ) -> Self {
        MinimisationReport {
            n_data_points,
            error_metric,
            objective_fn,
            n_iteration,
            termination: TerminationReason::User("Brute force fitting"),
        }
    }
}

/// Report of a fitting process.
pub struct FittingReport<M> {
    /// Index of the best model found.
    best: Option<usize>,
    /// The reports of the fitting process. Includes the model and the
    /// minimisation report with different initial values.
    pub reports: Box<[(M, MinimisationReport)]>,
}

impl<M> FittingReport<M> {
    /// Creates a new fitting report from the results of the fitting process.
    pub fn new(results: Box<[(M, MinimisationReport)]>) -> Self {
        let mut reports = results;
        reports.sort_by(|(_, r1), (_, r2)| r1.objective_fn.partial_cmp(&r2.objective_fn).unwrap());
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
    pub fn best_model_report(&self) -> Option<&(M, MinimisationReport)> {
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
        if n == 0 {
            return;
        }

        if self.reports.is_empty() {
            println!("No fitting reports");
            return;
        }

        println!("Fitting reports (first {}):", n);
        for (m, r) in self.reports.iter().take(n) {
            println!(
                "    - Model: {:?}, Err: {:?}, ObjFn: {}",
                m, r.error_metric, r.objective_fn
            );
        }
        let best = self.best_model();
        if best.is_none() {
            println!("  No best model found");
            return;
        }
        let best_report = self.best_model_report().unwrap();
        // Check if the best model is user terminated
        // Currently, the user termination is only used in the brute force fitting
        if let TerminationReason::User(_) = best_report.1.termination {
            println!(
                "  Best model: {:?}, metric: {:?}, err: {}",
                best.unwrap(),
                best_report.1.error_metric,
                best_report.1.objective_fn
            );
        } else {
            // Compute the mse error for nllsq fitting
            let rcp = rcp_f64(best_report.1.n_data_points as f64);
            let mse = best_report.1.objective_fn * 2.0 * rcp;
            println!(
                "  Best model: {:?}, metric: {:?}, err: {}, mse: {}",
                best.unwrap(),
                best_report.1.error_metric,
                best_report.1.objective_fn,
                mse
            );
        }
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
        symmetry: Symmetry,
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

    use super::{FittingProblem, FittingReport, MinimisationReport};
    use crate::brdf::analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR};
    use brute::compute_distance_between_measured_and_modelled;
    use levenberg_marquardt::{LevenbergMarquardt, TerminationReason};
    use nllsq::{init_microfacet_brdf_models, NllsqBrdfFittingProxy};
    use rayon::{
        iter::{ParallelBridge, ParallelIterator},
        slice::ParallelSlice,
    };
    use vgonio_core::{
        bxdf::{AnalyticalBrdf, BrdfProxy, MicrofacetDistroKind},
        units::Radians,
        utils::range::StepRangeIncl,
        ErrorMetric, Symmetry, Weighting,
    };

    impl<'a> FittingProblem for BrdfProxy<'a> {
        type Model = Box<dyn AnalyticalBrdf<Params = [f64; 2]>>;

        fn nllsq_fit(
            &self,
            target: MicrofacetDistroKind,
            symmetry: Symmetry,
            weighting: Weighting,
            initial: StepRangeIncl<f64>,
            max_theta_i: Option<Radians>,
            max_theta_o: Option<Radians>,
        ) -> FittingReport<Self::Model> {
            let solver = LevenbergMarquardt::new();
            let cpu_count = (std::thread::available_parallelism().unwrap().get() / 2).max(1);
            let tasks = init_microfacet_brdf_models(initial, target, symmetry);
            let tasks_per_cpu = tasks.len().div_ceil(cpu_count);
            log::debug!(
                "Solve {} models, {} per CPU, {} CPUs",
                tasks.len(),
                tasks_per_cpu,
                cpu_count
            );
            let n_filtered_samples = self.n_filtered_samples(max_theta_i, max_theta_o);
            let results = tasks
                .par_chunks(tasks_per_cpu)
                .flat_map(|models| {
                    models
                        .iter()
                        .filter_map(|model| {
                            let (fitted_model, report) = match symmetry {
                                Symmetry::Isotropic => {
                                    let nllsq_proxy =
                                        NllsqBrdfFittingProxy::<'_, { Symmetry::Isotropic }>::new(
                                            &self,
                                            model.clone(),
                                            weighting,
                                            max_theta_i,
                                            max_theta_o,
                                        );
                                    let (result, report) = solver.minimize(nllsq_proxy);
                                    (
                                        result.model,
                                        MinimisationReport::from_lm_nllsq(
                                            report,
                                            n_filtered_samples,
                                        ),
                                    )
                                },
                                Symmetry::Anisotropic => {
                                    let nllsq_proxy = NllsqBrdfFittingProxy::<
                                        '_,
                                        { Symmetry::Anisotropic },
                                    >::new(
                                        &self,
                                        model.clone(),
                                        weighting,
                                        max_theta_i,
                                        max_theta_o,
                                    );
                                    let (result, report) = solver.minimize(nllsq_proxy);
                                    (
                                        result.model,
                                        MinimisationReport::from_lm_nllsq(
                                            report,
                                            n_filtered_samples,
                                        ),
                                    )
                                },
                            };

                            match report.termination {
                                TerminationReason::Converged { .. }
                                | TerminationReason::LostPatience => Some((fitted_model, report)),
                                _ => {
                                    log::warn!(
                                        "Fitting failed for model: {:?} with reason: {:?}",
                                        model,
                                        report.termination
                                    );
                                    None
                                },
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
            log::debug!("start brute force fitting");
            let cpu_count = ((std::thread::available_parallelism().unwrap().get()) / 2).max(1);
            let mut step_size = 0.01;
            let mut alphas = StepRangeIncl::new(0.0, 1.0, step_size)
                .values()
                .collect::<Box<[f64]>>();
            let mut errs = vec![f64::NAN; 256].into_boxed_slice();
            // The number of iterations as each iteration has two digits of precision
            let n_times = (precision + 1) / 2;
            log::debug!("Brute force fitting for {} times", n_times);
            let n_filtered_samples = self.n_filtered_samples(max_theta_i, max_theta_o);
            let mut records = Vec::with_capacity(n_times as usize * 256);

            for i in 0..n_times {
                let chunk_size = alphas.len().div_ceil(cpu_count);
                log::debug!(
                    "--- iteration: {}, chunk_size: {}, \n - {:?}",
                    i,
                    chunk_size,
                    alphas
                );

                let multi_pb = indicatif::MultiProgress::new();
                let pbs = (0..alphas.len() / chunk_size)
                    .map(|_| {
                        let pb = multi_pb.add(indicatif::ProgressBar::new(chunk_size as u64));
                        pb.set_style(
                            indicatif::ProgressStyle::default_bar()
                                .template(
                                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                                     {pos}/{len} ({eta})",
                                )
                                .unwrap()
                                .progress_chars("#>-"),
                        );
                        pb.set_position(0);
                        pb
                    })
                    .collect::<Vec<_>>();
                alphas
                    .chunks(chunk_size)
                    .zip(errs.chunks_mut(chunk_size))
                    .zip(pbs.iter())
                    .par_bridge()
                    .for_each(|((alpha_chunks, err_chunks), pb)| {
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
                                pb.inc(1);
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
                                as Box<dyn AnalyticalBrdf<Params = [f64; 2]>>
                        },
                        MicrofacetDistroKind::TrowbridgeReitz => {
                            Box::new(MicrofacetBrdfTR::new(*alpha, *alpha))
                                as Box<dyn AnalyticalBrdf<Params = [f64; 2]>>
                        },
                    };
                    records.push((
                        m,
                        MinimisationReport::from_brute_force(
                            metric,
                            *err,
                            n_filtered_samples,
                            i as usize,
                        ),
                    ));
                });
                log::debug!("- errs: {:?}", errs);
                // Find the range of the best fit
                let min_err = errs.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                let min_err_idx = errs.iter().position(|&x| x == min_err).unwrap();
                let min_err_alpha = alphas[min_err_idx];
                log::debug!(
                    "min err: {}, alpha: {}, min_index: {}",
                    min_err,
                    min_err_alpha,
                    min_err_idx
                );
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

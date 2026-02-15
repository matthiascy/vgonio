//! Types and structures for BRDF fitting.
use crate::{
    brdf::AnalyticalBrdf,
    distro::{MicrofacetDistribution, MicrofacetDistroKind},
    BrdfFamily,
};
use levenberg_marquardt::{MinimizationReport, TerminationReason};
use std::fmt::Debug;
use vgn_core::{
    cli, math::rcp_f64, units::Radians, utils::range::StepRangeIncl, ErrorMetric, Symmetry,
    Weighting,
};

pub mod proxy;

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

/// Roughness values for the isotropic and anisotropic cases.
#[derive(Debug, Clone, Copy)]
pub enum Roughness {
    Isotropic {
        a: StepRangeIncl<f64>,
    },
    Anisotropic {
        ax: StepRangeIncl<f64>,
        ay: StepRangeIncl<f64>,
    },
}

/// A model after fitting.
#[derive(Debug, Clone)]
pub enum FittedModel {
    /// Bidirectional scattering distribution function.
    Bsdf(Box<dyn AnalyticalBrdf<[f64; 2]>>),
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

    /// Computes the mean squared error from the objective function value.
    pub fn mse(&self) -> f64 {
        match self.error_metric {
            ErrorMetric::L1 => {
                panic!("Mean squared error cannot be directly computed from L1 error metric")
            },
            ErrorMetric::L2 => {
                let rcp = rcp_f64(self.n_data_points as f64);
                self.objective_fn * self.objective_fn * rcp
            },
            ErrorMetric::Mse => self.objective_fn,
            ErrorMetric::Rmse => self.objective_fn * self.objective_fn,
            ErrorMetric::Nllsq => {
                let rcp = rcp_f64(self.n_data_points as f64);
                self.objective_fn * 2.0 * rcp
            },
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
        let mut reports = results
            .into_vec()
            .into_iter()
            .filter(|x| !x.1.objective_fn.is_nan())
            .collect::<Box<_>>();
        reports.sort_by(|(_, r1), (_, r2)| {
            r1.objective_fn
                .partial_cmp(&r2.objective_fn)
                .expect("NaN values in objective function")
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
    pub fn best_model_report(&self) -> Option<&(M, MinimisationReport)> {
        self.best.map(|i| &self.reports[i])
    }

    /// Log the fitting report.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of best models to log.
    /// * `indent` - The number of spaces to indent the log.
    pub fn print_fitting_report(&self, n: usize, indent: u32)
    where
        M: Debug,
    {
        // Only log the best model
        if n > 0 {
            if self.reports.is_empty() {
                println!("No fitting reports");
                return;
            }

            println!("Fitting reports (first {}):", n);
            for (m, r) in self.reports.iter().take(n) {
                println!(
                    "    - {:?}, metric: {}, obj_fn: {}",
                    m, r.error_metric, r.objective_fn
                );
            }
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
            cli::success(
                indent,
                format_args!(
                    "{:?}, metric: {}, obj_fn: {}",
                    best.unwrap(),
                    best_report.1.error_metric,
                    best_report.1.objective_fn
                ),
            );
        } else {
            // Compute the mse error for nllsq fitting
            let rcp = rcp_f64(best_report.1.n_data_points as f64);
            let mse = best_report.1.objective_fn * 2.0 * rcp;
            cli::success(
                indent,
                format_args!(
                    "{:?}, metric: {}, obj_fn: {}, mse: {}",
                    best.unwrap(),
                    best_report.1.error_metric,
                    best_report.1.objective_fn,
                    mse,
                ),
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
    /// * `precision` - The number of digits after the decimal point to consider.
    fn brute_fit(
        &self,
        target: MicrofacetDistroKind,
        symmetry: Symmetry,
        metric: ErrorMetric,
        weighting: Weighting,
        max_theta_i: Option<Radians>,
        max_theta_o: Option<Radians>,
        precision: u32,
        #[cfg(feature = "cuda")] on_gpu: bool,
        alpha: Option<Roughness>,
    ) -> FittingReport<Self::Model>;
}

/// Fitting for BRDFs.
pub mod brdf {
    /// Fitting for BRDFs using brute force.
    pub mod brute;
    /// Fitting for BRDFs using non-linear least squares.
    pub mod nllsq;

    use super::{FittingProblem, FittingReport, MinimisationReport, Roughness};
    use crate::{
        brdf::{
            analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR},
            AnalyticalBrdf,
        },
        distro::MicrofacetDistroKind,
        fitting::proxy::BrdfProxy,
    };
    use brute::compute_distance_between_measured_and_modelled;
    #[cfg(feature = "cuda")]
    use cust::{
        context::{Context, ContextFlags, CurrentContext},
        device::{Device, DeviceAttribute},
        module::Module,
        CudaFlags,
    };
    use indicatif::{MultiProgress, ProgressBar};
    use levenberg_marquardt::TerminationReason;
    use nllsq::{init_microfacet_brdf_models, NllsqBrdfFittingProxy};
    use rayon::iter::{ParallelBridge, ParallelIterator};
    use vgn_core::{units::Radians, utils::range::StepRangeIncl, ErrorMetric, Symmetry, Weighting};

    #[cfg(feature = "cli")]
    use vgn_core::cli;

    impl FittingProblem for BrdfProxy<'_> {
        type Model = Box<dyn AnalyticalBrdf<[f64; 2]>>;

        fn nllsq_fit(
            &self,
            target: MicrofacetDistroKind,
            symmetry: Symmetry,
            weighting: Weighting,
            initial: StepRangeIncl<f64>,
            max_theta_i: Option<Radians>,
            max_theta_o: Option<Radians>,
        ) -> FittingReport<Self::Model> {
            let cpu_count = (std::thread::available_parallelism().unwrap().get() / 2).max(1);
            let tasks = init_microfacet_brdf_models(initial, target, symmetry);
            let tasks_per_cpu = tasks.len().div_ceil(cpu_count);
            #[cfg(feature = "cli")]
            cli::step(
                6,
                format_args!(
                    "Solve {} models on {} CPUs, {} per CPU",
                    tasks.len(),
                    cpu_count,
                    tasks_per_cpu,
                ),
            );

            let multi_pb = MultiProgress::new();
            let pbs = (0..cpu_count)
                .map(|_| {
                    let pb = multi_pb.add(ProgressBar::new(tasks_per_cpu as u64));
                    pb.set_style(
                        indicatif::ProgressStyle::default_bar()
                            .template(
                                "      {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                                 {pos}/{len} ({eta})",
                            )
                            .unwrap()
                            .progress_chars("#>-"),
                    );
                    pb.set_position(0);
                    pb
                })
                .collect::<Vec<_>>();
            let results = tasks
                .chunks(tasks_per_cpu)
                .zip(pbs.iter())
                .par_bridge()
                .flat_map(|(models, pb)| {
                    models
                        .iter()
                        .filter_map(|model| {
                            let (fitted_model, report) = match symmetry {
                                Symmetry::Isotropic => {
                                    let nllsq =
                                        NllsqBrdfFittingProxy::<'_, { Symmetry::Isotropic }>::new(
                                            self,
                                            model.clone(),
                                            weighting,
                                            max_theta_i,
                                            max_theta_o,
                                        );
                                    nllsq.minimise()
                                },
                                Symmetry::Anisotropic => {
                                    let nllsq = NllsqBrdfFittingProxy::<
                                        '_,
                                        { Symmetry::Anisotropic },
                                    >::new(
                                        self,
                                        model.clone(),
                                        weighting,
                                        max_theta_i,
                                        max_theta_o,
                                    );
                                    nllsq.minimise()
                                },
                            };

                            pb.inc(1);

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
            symmetry: Symmetry,
            metric: ErrorMetric,
            weighting: Weighting,
            max_theta_i: Option<Radians>,
            max_theta_o: Option<Radians>,
            precision: u32,
            #[cfg(feature = "cuda")] on_gpu: bool,
            alpha: Option<Roughness>, // TODO: use in isotropic case
        ) -> FittingReport<Self::Model> {
            log::debug!("start brute force fitting");
            #[cfg(not(feature = "cuda"))]
            let cpu_count = ((std::thread::available_parallelism().unwrap().get()) / 2).max(1);
            #[cfg(feature = "cuda")]
            let cpu_count = ((std::thread::available_parallelism().unwrap().get()) / 5 * 3).max(1);

            let n_filtered_samples = self.n_filtered_samples(max_theta_i, max_theta_o);

            #[cfg(feature = "cuda")]
            let (context, device, modules, threads_per_block) = if on_gpu {
                let (context, device) =
                    vgn_core::cuda::init_cuda_context(ContextFlags::SCHED_AUTO).unwrap();
                let modules = vgn_core::cuda::load_ptx_modules().unwrap();
                let threads_per_block = device
                    .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
                    .unwrap() as u32;
                (
                    Some(context),
                    Some(device),
                    Some(modules),
                    Some(threads_per_block),
                )
            } else {
                (None, None, None, None)
            };

            match symmetry {
                Symmetry::Isotropic => {
                    let mut step_size = 0.01;
                    let mut alphas = StepRangeIncl::new(0.0, 1.0, step_size)
                        .values()
                        .collect::<Box<[f64]>>();
                    let mut errs = vec![f64::NAN; 256].into_boxed_slice();
                    // The number of iterations as each iteration has two digits of precision
                    let n_times = (precision + 1) / 2;
                    log::debug!("Brute force fitting for {} times", n_times);
                    let mut records = Vec::with_capacity(n_times as usize * 256);

                    log::debug!(
                        "Source: {:?}, kind: {:?}, spectrum: {:?}",
                        self.source,
                        self.brdf.kind(),
                        self.spectrum
                    );

                    for i in 0..n_times {
                        let chunk_size = alphas.len().div_ceil(cpu_count);
                        log::debug!(
                            "--- iteration: {}, chunk_size: {}, \n - {:?}",
                            i,
                            chunk_size,
                            alphas
                        );

                        let (_, pbs) =
                            create_multi_progress_bar(alphas.len() as u64, cpu_count as u64);
                        alphas
                            .chunks(chunk_size)
                            .zip(errs.chunks_mut(chunk_size))
                            .zip(pbs.iter())
                            .par_bridge()
                            .for_each(|((alpha_chunks, err_chunks), pb)| {
                                #[cfg(feature = "cuda")]
                                if on_gpu {
                                    CurrentContext::set_current(context.as_ref().unwrap()).unwrap();
                                }
                                err_chunks.iter_mut().zip(alpha_chunks.iter()).for_each(
                                    |(err, alpha)| {
                                        *err = compute_distance_between_measured_and_modelled(
                                            self,
                                            target,
                                            metric,
                                            weighting,
                                            *alpha,
                                            *alpha,
                                            max_theta_i.unwrap_or(Radians::HALF_PI),
                                            max_theta_o.unwrap_or(Radians::HALF_PI),
                                            #[cfg(feature = "cuda")]
                                            threads_per_block,
                                            #[cfg(feature = "cuda")]
                                            modules.as_ref(),
                                        );
                                        pb.inc(1);
                                    },
                                );
                            });
                        // Record the error and alpha
                        errs.iter().zip(alphas.iter()).for_each(|(err, alpha)| {
                            if err.is_nan() {
                                log::debug!("Skipping NaN error");
                                return;
                            }
                            let m = match target {
                                MicrofacetDistroKind::Beckmann => {
                                    Box::new(MicrofacetBrdfBK::new(*alpha, *alpha))
                                        as Box<dyn AnalyticalBrdf<[f64; 2]>>
                                },
                                MicrofacetDistroKind::TrowbridgeReitz => {
                                    Box::new(MicrofacetBrdfTR::new(*alpha, *alpha))
                                        as Box<dyn AnalyticalBrdf<[f64; 2]>>
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
                        log::debug!("- min_err: {:?}", min_err);
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
                        step_size *= 0.01;
                    }

                    // Convert the records to FittingReport
                    FittingReport::new(records.into_boxed_slice())
                },
                Symmetry::Anisotropic => {
                    let Roughness::Anisotropic { ax, ay } =
                        alpha.unwrap_or(Roughness::Anisotropic {
                            ax: StepRangeIncl::new(0.0, 1.0, 0.01),
                            ay: StepRangeIncl::new(0.0, 1.0, 0.01),
                        })
                    else {
                        unreachable!("Anisotropic roughness must be provided");
                    };
                    let count = ax.step_count() * ay.step_count();
                    let mut errs = Box::new_uninit_slice(count);
                    let chunk_size = count.div_ceil(cpu_count);

                    let alphas = (0..count)
                        .map(|i| {
                            let alpha_x_idx = i / ay.step_count();
                            let alpha_y_idx = i % ay.step_count();
                            let alpha_x = ax.start + alpha_x_idx as f64 * ax.step_size;
                            let alpha_y = ay.start + alpha_y_idx as f64 * ay.step_size;
                            (alpha_x, alpha_y)
                        })
                        .collect::<Box<[(f64, f64)]>>();

                    let (_, pbs) = create_multi_progress_bar(count as u64, cpu_count as u64);

                    errs.chunks_mut(chunk_size)
                        .zip(alphas.chunks(chunk_size))
                        .zip(pbs.iter())
                        .enumerate()
                        .par_bridge()
                        .for_each(|(i, ((err_chunks, alpha_chunks), pb))| {
                            #[cfg(feature = "cuda")]
                            if on_gpu {
                                CurrentContext::set_current(context.as_ref().unwrap()).unwrap();
                            }
                            for j in 0..err_chunks.len() {
                                let (alpha_x, alpha_y) = alpha_chunks[j];
                                err_chunks[j].write(
                                    compute_distance_between_measured_and_modelled(
                                        self,
                                        target,
                                        metric,
                                        weighting,
                                        alpha_x,
                                        alpha_y,
                                        max_theta_i.unwrap_or(Radians::HALF_PI),
                                        max_theta_o.unwrap_or(Radians::HALF_PI),
                                        #[cfg(feature = "cuda")]
                                        threads_per_block,
                                        #[cfg(feature = "cuda")]
                                        modules.as_ref(),
                                    ),
                                );
                                pb.inc(1);
                            }
                        });

                    let errs = unsafe { errs.assume_init() };
                    let records = errs
                        .iter()
                        .zip(alphas.iter())
                        .map(|(err, alpha)| {
                            let m = match target {
                                MicrofacetDistroKind::Beckmann => {
                                    Box::new(MicrofacetBrdfBK::new(alpha.0, alpha.1))
                                        as Box<dyn AnalyticalBrdf<[f64; 2]>>
                                },
                                MicrofacetDistroKind::TrowbridgeReitz => {
                                    Box::new(MicrofacetBrdfTR::new(alpha.0, alpha.1))
                                        as Box<dyn AnalyticalBrdf<[f64; 2]>>
                                },
                            };
                            (
                                m,
                                MinimisationReport::from_brute_force(
                                    metric,
                                    *err,
                                    n_filtered_samples,
                                    0,
                                ),
                            )
                        })
                        .collect::<Box<_>>();
                    FittingReport::new(records)
                },
            }
        }
    }

    /// Creates a multi-progress bar.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of total tasks to perform.
    /// * `n_cpu` - The number of CPUs to use, each CPU will have a progress bar.
    fn create_multi_progress_bar(n: u64, n_cpu: u64) -> (MultiProgress, Box<[ProgressBar]>) {
        let n_tasks_per_cpu = n.div_ceil(n_cpu);
        let multi_pb = MultiProgress::new();
        let pbs = (0..n_cpu)
            .map(|i| {
                let remaining = n.saturating_sub(i * n_tasks_per_cpu);
                let n_tasks = n_tasks_per_cpu.min(remaining);
                let pb = multi_pb.add(ProgressBar::new(n_tasks));
                pb.set_style(
                    indicatif::ProgressStyle::default_bar()
                        .template(
                            "      {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                             {pos}/{len} ({eta})",
                        )
                        .unwrap()
                        .progress_chars("#>-"),
                );
                pb.set_position(0);
                pb
            })
            .collect::<Box<_>>();
        (multi_pb, pbs)
    }
}

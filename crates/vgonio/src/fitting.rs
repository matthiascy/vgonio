//! Fitting of microfacet distribution functions and bidirectional
//! scattering distribution functions.

/// Common method when implementing [`LeastSquaresProblem`] for a fitting
macro_rules! impl_least_squares_problem_common_methods {
    ($self:ident, $params_ty:ty) => {
        fn set_params(&mut $self, params: &$params_ty) {
            $self.model.set_alpha_x(params[0]);
            $self.model.set_alpha_y(params[1]);
        }

        fn params(&self) -> $params_ty {
            <$params_ty>::new(self.model.alpha_x(), self.model.alpha_y())
        }
    }
}

mod brdf;
mod ceres;
mod mdf;
pub mod mse;

pub use brdf::*;
pub use mdf::*;

use base::Isotropy;
use bxdf::{
    MicrofacetBasedBrdfModel, MicrofacetBasedBrdfModelKind, MicrofacetDistribution,
    MicrofacetDistributionKind,
};
use levenberg_marquardt::{MinimizationReport, TerminationReason};
use std::{fmt::Debug, ops::Bound};

/// Types of the fitting problem.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FittingProblemKind {
    /// Fitting the microfacet area distribution function.
    Mdf {
        /// The target microfacet distribution model.
        model: MicrofacetDistributionKind,
        /// The fitting variant.
        variant: MicrofacetDistributionFittingVariant,
    },
    /// Fitting the bidirectional scattering distribution function.
    Bsdf {
        /// The target BSDF model.
        model: MicrofacetBasedBrdfModelKind,
    },
}

/// A model after fitting.
#[derive(Debug, Clone)]
pub enum FittedModel {
    /// Bidirectional scattering distribution function.
    Bsdf(Box<dyn MicrofacetBasedBrdfModel>),
    /// Microfacet area distribution function with the scaling factor applied to
    /// the measured data.
    Adf(Box<dyn MicrofacetDistribution>, f32),
    /// Microfacet Masking-shadowing function.
    Msf(Box<dyn MicrofacetDistribution>),
}

impl FittedModel {
    /// Returns the isotropy of the model.
    pub fn isotropy(&self) -> Isotropy {
        match self {
            FittedModel::Bsdf(model) => model.isotropy(),
            FittedModel::Msf(model) | FittedModel::Adf(model, _) => model.isotropy(),
        }
    }

    /// Returns the kind of fitting problem.
    pub fn kind(&self) -> FittingProblemKind {
        match self {
            FittedModel::Bsdf(model) => FittingProblemKind::Bsdf {
                model: model.kind(),
            },
            FittedModel::Msf(model) => FittingProblemKind::Mdf {
                model: model.kind(),
                variant: MicrofacetDistributionFittingVariant::Msf,
            },
            FittedModel::Adf(model, _) => FittingProblemKind::Mdf {
                model: model.kind(),
                variant: MicrofacetDistributionFittingVariant::Adf,
            },
        }
    }

    pub fn scale(&self) -> Option<f32> {
        match self {
            FittedModel::Adf(_, scale) => Some(*scale),
            _ => None,
        }
    }
}

/// A collection of fitted models without repetition.
#[derive(Debug, Clone, Default)]
pub struct FittedModels(Vec<FittedModel>);

impl FittedModels {
    /// Checks if the collection already contains a model with the same kind and
    /// isotropy.
    pub fn contains(&self, kind: &FittingProblemKind, scale: Option<f32>) -> bool {
        self.0
            .iter()
            .any(|f| f.kind() == *kind && f.scale() == scale)
    }

    /// Push a new model to the collection.
    pub fn push(&mut self, model: FittedModel) { self.0.push(model); }
}

impl AsRef<[FittedModel]> for FittedModels {
    fn as_ref(&self) -> &[FittedModel] { self.0.as_ref() }
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
    pub fn new(results: Vec<(M, MinimizationReport<f64>)>, cond: impl Fn(&M) -> bool) -> Self {
        let reports = results
            .into_iter()
            .filter(|(_, r)| matches!(r.termination, TerminationReason::Converged { .. }))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut lowest_obj_func = f64::INFINITY;
        let mut best = None;
        for (i, (m, r)) in reports.iter().enumerate() {
            if r.objective_function < lowest_obj_func && cond(m) {
                lowest_obj_func = r.objective_function;
                best = Some(i);
            }
        }
        FittingReport { best, reports }
    }

    /// Returns the best model found.
    pub fn best_model(&self) -> Option<&M> { self.best.map(|i| &self.reports[i].0) }

    /// Returns the report of the best model found.
    pub fn best_model_report(&self) -> Option<&(M, MinimizationReport<f64>)> {
        self.best.map(|i| &self.reports[i])
    }

    /// Log the fitting report.
    pub fn log_fitting_reports(&self)
    where
        M: Debug,
    {
        log::info!("Fitting report:");
        log::info!("  Best model: {:?}", self.best_model());
        log::info!("  Reports:");
        for (m, r) in self.reports.iter() {
            log::info!(
                "    - Model: {:?}, objective_function: {}",
                m,
                r.objective_function
            );
        }
    }
}

/// A fitting problem.
pub trait FittingProblem {
    /// The model to fit.
    type Model;

    /// Non-linear least squares fitting using Levenberg-Marquardt algorithm.
    fn lsq_lm_fit(self) -> FittingReport<Self::Model>;
}

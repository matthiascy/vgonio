//! Fitting of microfacet distribution functions and bidirectional
//! scattering distribution functions.

mod mdf;

pub use mdf::*;

use base::Isotropy;
use bxdf::{MicrofacetDistributionModel, MicrofacetDistributionModelKind};
use levenberg_marquardt::MinimizationReport;
use std::fmt::Debug;

/// Types of the fitting problem.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FittingProblemKind {
    /// Fitting the microfacet area distribution function.
    Mdf {
        /// The target microfacet distribution model.
        model: MicrofacetDistributionModelKind,
        /// The fitting method.
        method: MicrofacetDistributionFittingMethod,
    },
    /// Fitting the bidirectional scattering distribution function.
    Bsdf(), // TODO: implement
}

/// A model after fitting.
#[derive(Debug, Clone)]
pub enum FittedModel {
    /// Bidirectional scattering distribution function.
    Bsdf(),
    /// Microfacet area distribution function with the scaling factor applied to
    /// the measured data.
    Adf(Box<dyn MicrofacetDistributionModel>, f32),
    /// Microfacet Masking-shadowing function.
    Msf(Box<dyn MicrofacetDistributionModel>),
}

impl FittedModel {
    /// Returns the isotropy of the model.
    pub fn isotropy(&self) -> Isotropy {
        match self {
            FittedModel::Bsdf() => Isotropy::Isotropic, // TODO: implement
            FittedModel::Msf(model) | FittedModel::Adf(model, _) => model.isotropy(),
        }
    }

    /// Returns the kind of fitting problem.
    pub fn kind(&self) -> FittingProblemKind {
        match self {
            FittedModel::Bsdf() => FittingProblemKind::Bsdf(), // TODO: implement
            FittedModel::Msf(model) => FittingProblemKind::Mdf {
                model: model.kind(),
                method: MicrofacetDistributionFittingMethod::Msf,
            },
            FittedModel::Adf(model, _) => FittingProblemKind::Mdf {
                model: model.kind(),
                method: MicrofacetDistributionFittingMethod::Adf,
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
    best: usize,
    /// The reports of the fitting process. Includes the model and the
    /// minimization report with different initial values.
    pub reports: Vec<(M, MinimizationReport<f64>)>,
}

impl<M> FittingReport<M> {
    /// Returns the best model found.
    pub fn best_model(&self) -> &M { &self.reports[self.best].0 }

    /// Returns the report of the best model found.
    pub fn best_model_report(&self) -> &(M, MinimizationReport<f64>) { &self.reports[self.best] }

    /// Log the fitting report.
    pub fn log_fitting_report(&self)
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

    /// Non linear least squares fitting using Levenberg-Marquardt algorithm.
    fn lsq_lm_fit(self) -> FittingReport<Self::Model>;
}

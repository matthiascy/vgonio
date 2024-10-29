//! Fitting of microfacet distribution functions and bidirectional
//! scattering distribution functions.

/// Common method when implementing [`LeastSquaresProblem`] for a fitting
macro_rules! impl_least_squares_problem_common_methods {
    (@aniso => $self:ident, $params_ty:ty) => {
        fn set_params(&mut $self, params: &$params_ty) {
            $self.model.set_params(params.as_ref());
        }

        fn params(&$self) -> $params_ty {
            let [x, y] = $self.model.params();
            <$params_ty>::new(x, y)
        }
    };
    (@iso2 => $self:ident, $params_ty:ty) => {
        fn set_params(&mut $self, params: &$params_ty) {
            $self.model.set_params(&[params[0], params[0]]);
        }

        fn params(&$self) -> $params_ty {
            let [x, _] = $self.model.params();
            <$params_ty>::new(x)
        }
    }
}

mod brdf;
pub mod err;
mod mfd;

pub use brdf::*;
pub use mfd::*;

use base::{Isotropy, ResidualErrorMetric};
use bxdf::{
    brdf::{Bxdf, BxdfFamily},
    distro::{MicrofacetDistribution, MicrofacetDistroKind},
};
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

/// A collection of fitted models without repetition.
#[derive(Debug, Clone, Default)]
pub struct FittedModels(Vec<FittedModel>);

impl FittedModels {
    /// Checks if the collection already contains a model with the same kind and
    /// isotropy.
    pub fn contains(
        &self,
        kind: &FittingProblemKind,
        scale: Option<f32>,
        isotropy: Isotropy,
    ) -> bool {
        self.0
            .iter()
            .any(|f| f.kind() == *kind && f.scale() == scale && f.isotropy() == isotropy)
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
    pub fn new(results: Vec<(M, MinimizationReport<f64>)>) -> Self {
        let mut reports = results.into_boxed_slice();
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
    pub fn print_fitting_report(&self)
    where
        M: Debug,
    {
        println!("Fitting reports (first 16):");
        for (m, r) in self.reports.iter().take(16) {
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
    fn lsq_lm_fit(
        self,
        isotropy: Isotropy,
        rmetric: ResidualErrorMetric,
    ) -> FittingReport<Self::Model>;
}

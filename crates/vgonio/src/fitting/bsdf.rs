use crate::{
    fitting::{FittingProblem, FittingReport},
    measure::bsdf::MeasuredBsdfData,
    partition::SphericalPartition,
};
use base::{math::Vec3, Isotropy};
use bxdf::MicrofacetBasedBsdfModelKind;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Dyn, Matrix, Owned, VecStorage, Vector, U1, U2};
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

/// Beckmann microfacet BSDF model.
/// See [Beckmann Distribution](crate::dist::BeckmannDistribution).
#[derive(Debug, Clone, Copy)]
pub struct BeckmannBsdfModel {
    /// Roughness parameter of the originated from microfacet distribution
    /// function.
    alpha_x: f64,
    /// Roughness parameter of the originated from microfacet distribution
    alpha_y: f64,
}

/// Trowbridge-Reitz(GGX) microfacet BSDF model.
/// See [Trowbridge-Reitz
/// Distribution](crate::dist::TrowbridgeReitzDistribution).
#[derive(Debug, Clone, Copy)]
pub struct TrowbridgeReitzBsdfModel {
    /// Roughness parameter of the originated from microfacet distribution
    alpha_x: f64,
    /// Roughness parameter of the originated from microfacet distribution
    alpha_y: f64,
}

impl BeckmannBsdfModel {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        BeckmannBsdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

impl TrowbridgeReitzBsdfModel {
    pub fn new(alpha_x: f64, alpha_y: f64) -> Self {
        TrowbridgeReitzBsdfModel {
            alpha_x: alpha_x.max(1.0e-6),
            alpha_y: alpha_y.max(1.0e-6),
        }
    }
}

impl MicrofacetBasedBsdfModel for BeckmannBsdfModel {
    fn kind(&self) -> MicrofacetBasedBsdfModelKind { todo!() }

    fn isotropy(&self) -> Isotropy { todo!() }

    fn alpha_x(&self) -> f64 { todo!() }

    fn set_alpha_x(&mut self, alpha_x: f64) { todo!() }

    fn alpha_y(&self) -> f64 { todo!() }

    fn set_alpha_y(&mut self, alpha_y: f64) { todo!() }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 { todo!() }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBsdfModel> { todo!() }
}

impl MicrofacetBasedBsdfModel for TrowbridgeReitzBsdfModel {
    fn kind(&self) -> MicrofacetBasedBsdfModelKind { todo!() }

    fn isotropy(&self) -> Isotropy { todo!() }

    fn alpha_x(&self) -> f64 { todo!() }

    fn set_alpha_x(&mut self, alpha_x: f64) { todo!() }

    fn alpha_y(&self) -> f64 { todo!() }

    fn set_alpha_y(&mut self, alpha_y: f64) { todo!() }

    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 { todo!() }

    fn clone_box(&self) -> Box<dyn MicrofacetBasedBsdfModel> { todo!() }
}

pub trait MicrofacetBasedBsdfModel: Debug + Send {
    /// Returns the kind of the BSDF model.
    fn kind(&self) -> MicrofacetBasedBsdfModelKind;

    /// Returns the isotropy of the model.
    fn isotropy(&self) -> Isotropy;

    /// Returns the roughness parameter αx of the model.
    fn alpha_x(&self) -> f64;

    /// Sets the roughness parameter αx of the model.
    fn set_alpha_x(&mut self, alpha_x: f64);

    /// Returns the roughness parameter αy of the model.
    fn alpha_y(&self) -> f64;

    /// Sets the roughness parameter αy of the model.
    fn set_alpha_y(&mut self, alpha_y: f64);

    /// Evaluates the BSDF model.
    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3;

    /// Clones the model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn MicrofacetBasedBsdfModel>;
}

impl Clone for Box<dyn MicrofacetBasedBsdfModel> {
    fn clone(&self) -> Box<dyn MicrofacetBasedBsdfModel> { self.clone_box() }
}

pub trait MicrofactBasedBsdfModelFittingModel: MicrofacetBasedBsdfModel {
    fn partial_derivative(&self, wo: Vec3, wi: Vec3) -> Vec3;
    fn partial_derivatives(&self, wo: Vec3, wi: Vec3) -> (Vec3, Vec3);
}

/// The fitting problem for the microfacet based BSDF model.
///
/// The fitting procedure ir based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetBasedBsdfFittingProblem<'a> {
    /// The measured BSDF data.
    pub measured: Cow<'a, MeasuredBsdfData>,
    /// The target BSDF model.
    pub target: MicrofacetBasedBsdfModelKind,
}

impl<'a> Display for MicrofacetBasedBsdfFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSDF Fitting Problem")
            .field("target", &self.target)
            .finish()
    }
}

impl<'a> MicrofacetBasedBsdfFittingProblem<'a> {
    /// Creates a new BSDF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured BSDF data.
    /// * `target` - The target BSDF model.
    pub fn new(measured: &'a MeasuredBsdfData, target: MicrofacetBasedBsdfModelKind) -> Self {
        Self {
            measured: Cow::Borrowed(measured),
            target,
        }
    }
}

/// Initialises the microfacet based BSDF models with the given range of
/// roughness parameters as the initial guess.
fn initialise_microfacet_bsdf_models(
    min: f64,
    max: f64,
    num: u32,
    target: MicrofacetBasedBsdfModelKind,
) -> Vec<Box<dyn MicrofacetBasedBsdfModel>> {
    let step = (max - min) / (num - 1) as f64;
    match target {
        MicrofacetBasedBsdfModelKind::TrowbridgeReitz => (0..num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(TrowbridgeReitzBsdfModel::new(alpha, alpha)) as _
            })
            .collect(),
        MicrofacetBasedBsdfModelKind::Beckmann => (0..num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(BeckmannBsdfModel::new(alpha, alpha)) as _
            })
            .collect(),
    }
}

// Actual implementation of the fitting problem.
impl<'a> FittingProblem for MicrofacetBasedBsdfFittingProblem<'a> {
    type Model = Box<dyn MicrofacetBasedBsdfModel>;

    fn lsq_lm_fit(self) -> FittingReport<Self::Model> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let solver = LevenbergMarquardt::new();
        let mut results = {
            initialise_microfacet_bsdf_models(0.001, 2.0, 32, self.target)
                .into_par_iter()
                .filter_map(|model| {
                    log::debug!(
                        "Fitting model: {:?} with αx = {} αy = {}",
                        model,
                        model.alpha_x(),
                        model.alpha_y()
                    );
                    let problem = MicrofacetBasedFittingProblemProxy::new(&self.measured, model);
                    let (result, report) = solver.minimize(problem);
                    log::debug!(
                        "Fitted αx = {} αy = {}, report: {:?}",
                        result.model.alpha_x(),
                        result.model.alpha_y(),
                        report
                    );
                    match report.termination {
                        TerminationReason::Converged { .. } => Some((result.model as _, report)),
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
        };
        results.shrink_to_fit();
        let reports = results
            .into_iter()
            .filter(|(_, report)| matches!(report.termination, TerminationReason::Converged { .. }))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut lowest = f64::INFINITY;
        let mut best = 0;
        for (i, (_, report)) in reports.iter().enumerate() {
            if report.objective_function < lowest {
                lowest = report.objective_function;
                best = i;
            }
        }
        FittingReport { best, reports }
    }
}

struct MicrofacetBasedFittingProblemProxy<'a> {
    /// The measured BSDF data.
    measured: &'a MeasuredBsdfData,
    /// The target BSDF model.
    model: Box<dyn MicrofacetBasedBsdfModel>,
    /// The actual partition (not the params) of the measured BSDF data.
    partition: SphericalPartition,
}

impl<'a> MicrofacetBasedFittingProblemProxy<'a> {
    pub fn new(measured: &'a MeasuredBsdfData, model: Box<dyn MicrofacetBasedBsdfModel>) -> Self {
        let partition = measured.params.receiver.partitioning();
        Self {
            measured,
            model,
            partition,
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2> for MicrofacetBasedFittingProblemProxy<'a> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_alpha_x(x[0]);
        self.model.set_alpha_y(x[1]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::new(self.model.alpha_x(), self.model.alpha_y())
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> { todo!() }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> { todo!() }
}

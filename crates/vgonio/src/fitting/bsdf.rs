use crate::{
    fitting::{FittingProblem, FittingReport},
    measure::bsdf::MeasuredBsdfData,
    partition::SphericalPartition,
};
use base::{
    math::{Sph2, Vec3},
    Isotropy,
};
use bxdf::MicrofacetBasedBsdfModelKind;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Dyn, Matrix, Owned, VecStorage, Vector, U1, U2};
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

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

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        // The number of residuals is the number of patches times the number of
        // snapshots.
        let residuals_count = self.partition.num_patches() * self.measured.snapshots.len();
        let mut residuals = Box::new_uninit_slice(residuals_count);
        self.measured
            .snapshots
            .iter()
            .enumerate()
            .for_each(|(i, snapshot)| {
                let wi = snapshot.w_i;
                self.partition
                    .patches
                    .iter()
                    .enumerate()
                    .for_each(|(j, patch)| {
                        let wo = patch.center();
                        // Only the first wavelength is used. TODO: Use all
                        let measured = snapshot.samples[j][0];
                        let modelled = self.model.eval(wi, wo);
                    });
            });
        Some(Matrix::from_column_slice_generic(
            U2,
            self.partition.len(),
            residuals.as_slice(),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> { todo!() }
}

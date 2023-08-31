use crate::{
    fitting::{FittingProblem, FittingReport, MicrofacetDistributionModel},
    measure::microfacet::{MeasuredAdfData, MeasuredMsfData},
    RangeByStepSizeInclusive,
};
use levenberg_marquardt::{
    LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use std::{assert_matches::debug_assert_matches, borrow::Cow, fmt::Display};
use vgbxdf::{
    BeckmannSpizzichinoDistribution, MicrofacetDistributionFittingModel,
    MicrofacetDistributionModelKind, TrowbridgeReitzDistribution,
};
use vgcore::units::Radians;

/// The measured data related to the microfacet distribution function (MDF).
#[derive(Debug, Clone)]
pub enum MeasuredMdfData<'a> {
    /// The measured area distribution function (ADF).
    Adf(Cow<'a, MeasuredAdfData>),
    /// The measured masking-shadowing function (MSF).
    Msf(Cow<'a, MeasuredMsfData>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MicrofacetDistributionFittingMethod {
    /// Fitting the MDF through area distribution function (ADF).
    Adf,
    /// Fitting the MDF through masking-shadowing function (MSF).
    Msf,
}

/// Fitting procedure trying to find different models for the microfacet
/// distribution function (MDF) that best fits the measured data.
///
/// The MDF models can be obtained by either fitting the measured area
/// distribution function (ADF) or the measured masking-shadowing function
/// (MSF).
///
/// The fitting procedure is based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetDistributionFittingProblem<'a> {
    /// The measured data to fit to.
    measured: MeasuredMdfData<'a>,
    /// The target model.
    target: MicrofacetDistributionModelKind,
}

impl<'a> Display for MicrofacetDistributionFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MDF-ADF-FittingProblem")
            .field("target", &self.target)
            .finish()
    }
}

impl<'a> MicrofacetDistributionFittingProblem<'a> {
    /// Creates a new ADF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured ADF data.
    /// * `target` - The target model.
    pub fn new_adf_fitting(
        measured: &'a MeasuredAdfData,
        target: MicrofacetDistributionModelKind,
    ) -> Self {
        Self {
            measured: MeasuredMdfData::Adf(Cow::Borrowed(measured)),
            target,
        }
    }

    /// Creates a new MSF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured MSF data.
    /// * `target` - The target model.
    pub fn new_msf_fitting(
        measured: &'a MeasuredMsfData,
        target: MicrofacetDistributionModelKind,
    ) -> Self {
        Self {
            measured: MeasuredMdfData::Msf(Cow::Borrowed(measured)),
            target,
        }
    }

    pub fn new(
        measured: MeasuredMdfData<'a>,
        method: MicrofacetDistributionFittingMethod,
        target: MicrofacetDistributionModelKind,
    ) -> Self {
        debug_assert_matches!(
            (method, &measured),
            (
                MicrofacetDistributionFittingMethod::Adf,
                MeasuredMdfData::Adf(_)
            ) | (
                MicrofacetDistributionFittingMethod::Msf,
                MeasuredMdfData::Msf(_)
            ),
            "The fitting method and the measured data must match."
        );
        Self { measured, target }
    }
}

impl<'a> FittingProblem for MicrofacetDistributionFittingProblem<'a> {
    type Model = Box<dyn MicrofacetDistributionModel>;

    fn lsq_lm_fit(self) -> FittingReport<Self::Model> {
        let solver = LevenbergMarquardt::new();
        let mut result: Vec<(
            Box<dyn MicrofacetDistributionModel>,
            MinimizationReport<f64>,
        )> = {
            match self.measured {
                MeasuredMdfData::Adf(measured) => {
                    initialise_microfacet_adf_models(0.001, 3.0, 16, self.target)
                        .into_iter()
                        .filter_map(|model| {
                            let problem = AreaDistributionFittingProblemProxy {
                                measured: measured.as_ref(),
                                model,
                            };
                            let (result, report) = solver.minimize(problem);
                            match report.termination {
                                TerminationReason::Converged { .. } => {
                                    Some((result.model as _, report))
                                }
                                _ => {
                                    log::debug!("Model fitting failed: {:?}", report);
                                    None
                                }
                            }
                        })
                        .collect()
                }
                MeasuredMdfData::Msf(_) => {
                    todo!()
                }
            }
        };
        let reports = result
            .into_iter()
            .filter(|(m, r)| matches!(r.termination, TerminationReason::Converged { .. }))
            .collect::<Vec<_>>();
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

/// Proxy for the ADF fitting problem.
struct AreaDistributionFittingProblemProxy<'a> {
    measured: &'a MeasuredAdfData,
    model: Box<dyn MicrofacetDistributionFittingModel>,
}

struct MaskingShadowingFittingProblemProxy<'a> {
    measured: &'a MeasuredMsfData,
    model: Box<dyn MicrofacetDistributionFittingModel>,
}

fn initialise_microfacet_adf_models(
    min: f64,
    max: f64,
    num: u32,
    target: MicrofacetDistributionModelKind,
) -> Vec<Box<dyn MicrofacetDistributionFittingModel>> {
    let step = (max - min) / (num as f64);
    match target {
        MicrofacetDistributionModelKind::BeckmannSpizzichino => (0..num)
            .map(|i| {
                Box::new(BeckmannSpizzichinoDistribution::new(
                    (i + 1) as f64 * step,
                    (i + 1) as f64 * step,
                )) as _
            })
            .collect(),
        MicrofacetDistributionModelKind::TrowbridgeReitz => (0..num)
            .map(|i| {
                Box::new(TrowbridgeReitzDistribution::new(
                    (i + 1) as f64 * step,
                    (i + 1) as f64 * step,
                )) as _
            })
            .collect(),
    }
}

/// Extracts the cosine of all the measured zenith and azimuth angles.
fn extract_thetas_phis(
    len: usize,
    zenith: RangeByStepSizeInclusive<Radians>,
    azimuth: RangeByStepSizeInclusive<Radians>,
) -> (Vec<f64>, Vec<f64>) {
    let theta_step_count = zenith.step_count_wrapped();
    (0..len)
        .map(|idx| {
            let theta_idx = idx % theta_step_count;
            let phi_idx = idx / theta_step_count;
            (
                zenith.step(theta_idx).cos() as f64,
                azimuth.step(phi_idx).cos() as f64,
            )
        })
        .unzip()
}

/// Extracts the cosine of all the measured zenith angles.
fn extract_thetas(len: usize, zenith: RangeByStepSizeInclusive<Radians>) -> Vec<f64> {
    let theta_step_count = zenith.step_count_wrapped();
    (0..len)
        .map(|idx| {
            let theta_idx = idx % theta_step_count;
            zenith.step(theta_idx).cos() as f64
        })
        .collect()
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2> for AreaDistributionFittingProblemProxy<'a> {
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
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        let residuals = self.measured.samples.iter().enumerate().map(|(idx, meas)| {
            let theta_idx = idx % theta_step_count;
            let theta = self.measured.params.zenith.step(theta_idx);
            let phi_idx = idx / theta_step_count;
            let phi = self.measured.params.azimuth.step(phi_idx);
            self.model.eval_adf(theta.cos() as f64, phi.cos() as f64) * 0.25 - *meas as f64
        });
        Some(OMatrix::<f64, Dyn, U1>::from_iterator(
            residuals.len(),
            residuals,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        let (cos_thetas, cos_phis) = extract_thetas_phis(
            self.measured.samples.len(),
            self.measured.params.zenith,
            self.measured.params.azimuth,
        );
        Some(OMatrix::<f64, Dyn, U2>::from_row_slice(
            &self.model.adf_partial_derivatives(&cos_thetas, &cos_phis),
        ))
    }
}

// TODO: implement this.
// impl<'a> LeastSquaresProblem<f64, Dyn, U2> for
// MaskingShadowingFittingProblemProxy<'a> {
//
// }
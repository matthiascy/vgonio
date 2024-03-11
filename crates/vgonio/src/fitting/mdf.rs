use crate::{
    fitting::{FittingProblem, FittingReport, MicrofacetDistribution},
    measure::{
        microfacet::{MeasuredAdfData, MeasuredMsfData},
        params::AdfMeasurementMode,
    },
    RangeByStepSizeInclusive,
};
use base::units::Radians;
use bxdf::{
    dist::{BeckmannDistribution, TrowbridgeReitzDistribution},
    MicrofacetDistributionFittingModel, MicrofacetDistributionKind,
};
use levenberg_marquardt::{
    LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use rayon::slice::ParallelSlice;
use std::{assert_matches::debug_assert_matches, borrow::Cow, fmt::Display};

// TODO: Would MDF = ADF + MSF be more appropriate?
/// The measured microfacet distribution data (MDF).
///
/// The measured data can be either the area distribution function (ADF) or the
/// masking-shadowing function (MSF).
#[derive(Debug, Clone)]
pub enum MeasuredMdfData<'a> {
    /// The measured area distribution function (ADF).
    Adf(Cow<'a, MeasuredAdfData>),
    /// The measured masking-shadowing function (MSF).
    Msf(Cow<'a, MeasuredMsfData>),
}

/// Fitting variant for the microfacet distribution function (MDF).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MicrofacetDistributionFittingVariant {
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
    /// Scaling factor for the measured data.
    scale: f32,
    /// The measured data to fit to.
    measured: MeasuredMdfData<'a>,
    /// The target model.
    target: MicrofacetDistributionKind,
}

impl<'a> Display for MicrofacetDistributionFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MDF-ADF-FittingProblem")
            .field("target", &self.target)
            .field("scale", &self.scale)
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
        target: MicrofacetDistributionKind,
        scale: f32,
    ) -> Self {
        Self {
            scale,
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
        target: MicrofacetDistributionKind,
        scale: f32,
    ) -> Self {
        Self {
            scale,
            measured: MeasuredMdfData::Msf(Cow::Borrowed(measured)),
            target,
        }
    }

    /// Creates a new fitting problem.
    pub fn new(
        measured: MeasuredMdfData<'a>,
        method: MicrofacetDistributionFittingVariant,
        target: MicrofacetDistributionKind,
        scale: f32,
    ) -> Self {
        debug_assert_matches!(
            (method, &measured),
            (
                MicrofacetDistributionFittingVariant::Adf,
                MeasuredMdfData::Adf(_)
            ) | (
                MicrofacetDistributionFittingVariant::Msf,
                MeasuredMdfData::Msf(_)
            ),
            "The fitting method and the measured data must match."
        );
        Self {
            scale,
            measured,
            target,
        }
    }
}

impl<'a> FittingProblem for MicrofacetDistributionFittingProblem<'a> {
    type Model = Box<dyn MicrofacetDistribution>;

    fn lsq_lm_fit(self) -> FittingReport<Self::Model> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let solver = LevenbergMarquardt::new();
        let mut result: Vec<(Box<dyn MicrofacetDistribution>, MinimizationReport<f64>)> = {
            match self.measured {
                MeasuredMdfData::Adf(measured) => {
                    initialise_microfacet_mdf_models(0.001, 2.0, 32, self.target)
                        .into_par_iter()
                        .filter_map(|model| {
                            log::debug!(
                                "Fitting with αx = {} αy = {}",
                                model.alpha_x(),
                                model.alpha_y()
                            );
                            let problem = AreaDistributionFittingProblemProxy {
                                scale: self.scale,
                                measured: measured.as_ref(),
                                model,
                            };
                            let (result, report) = solver.minimize(problem);
                            log::debug!(
                                "Fitted αx = {} αy = {}, report: {:?}",
                                result.model.alpha_x(),
                                result.model.alpha_y(),
                                report
                            );
                            match report.termination {
                                TerminationReason::Converged { .. } => {
                                    Some((result.model as _, report))
                                }
                                _ => None,
                            }
                        })
                        .collect()
                }
                MeasuredMdfData::Msf(measured) => {
                    initialise_microfacet_mdf_models(0.001, 2.0, 32, self.target)
                        .into_iter()
                        .filter_map(|model| {
                            let problem = MaskingShadowingFittingProblemProxy {
                                measured: measured.as_ref(),
                                model,
                            };
                            let (result, report) = solver.minimize(problem);
                            match report.termination {
                                TerminationReason::Converged { .. } => {
                                    Some((result.model as _, report))
                                }
                                _ => None,
                            }
                        })
                        .collect()
                }
            }
        };
        result.shrink_to_fit();
        FittingReport::new(result, |m| m.alpha_x() > 0.0 && m.alpha_y() > 0.0)
    }
}

/// Proxy for the ADF fitting problem.
struct AreaDistributionFittingProblemProxy<'a> {
    scale: f32,
    measured: &'a MeasuredAdfData,
    model: Box<dyn MicrofacetDistributionFittingModel>,
}

struct MaskingShadowingFittingProblemProxy<'a> {
    measured: &'a MeasuredMsfData,
    model: Box<dyn MicrofacetDistributionFittingModel>,
}

fn initialise_microfacet_mdf_models(
    min: f64,
    max: f64,
    num: u32,
    target: MicrofacetDistributionKind,
) -> Vec<Box<dyn MicrofacetDistributionFittingModel>> {
    let step = (max - min) / (num as f64);
    match target {
        MicrofacetDistributionKind::Beckmann => (0..num)
            .map(|i| {
                Box::new(BeckmannDistribution::new(
                    (i + 1) as f64 * step,
                    (i + 1) as f64 * step,
                )) as _
            })
            .collect(),
        MicrofacetDistributionKind::TrowbridgeReitz => (0..num)
            .map(|i| {
                Box::new(TrowbridgeReitzDistribution::new(
                    (i + 1) as f64 * step,
                    (i + 1) as f64 * step,
                )) as _
            })
            .collect(),
    }
}

/// Extracts all the measured azimuth and zenith angles and applies the given
/// operation to them.p
fn extract_azimuth_zenith_angles(
    len: usize, // sample count
    azimuth: RangeByStepSizeInclusive<Radians>,
    zenith: RangeByStepSizeInclusive<Radians>,
) -> (Vec<f64>, Vec<f64>) {
    let theta_step_count = zenith.step_count_wrapped();
    (0..len)
        .map(|idx| {
            let theta_idx = idx % theta_step_count;
            let phi_idx = idx / theta_step_count;
            (
                azimuth.step(phi_idx).as_f64(),
                zenith.step(theta_idx).as_f64(),
            )
        })
        .unzip()
}

fn extract_azimuth_zenith_angles_cos(
    len: usize, // sample count
    azimuth: RangeByStepSizeInclusive<Radians>,
    zenith: RangeByStepSizeInclusive<Radians>,
) -> (Vec<f64>, Vec<f64>) {
    let theta_step_count = zenith.step_count_wrapped();
    (0..len)
        .map(|idx| {
            let theta_idx = idx % theta_step_count;
            let phi_idx = idx / theta_step_count;
            (
                azimuth.step(phi_idx).as_f64().cos(),
                zenith.step(theta_idx).as_f64().cos(),
            )
        })
        .unzip()
}

// TODO: maybe split this into two different structs for measured ADF in
// different modes?
impl<'a> LeastSquaresProblem<f64, Dyn, U2> for AreaDistributionFittingProblemProxy<'a> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    impl_least_squares_problem_common_methods!(self, Vector<f64, U2, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        match self.measured.params.mode {
            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
                let theta_step_count = zenith.step_count_wrapped();
                let residuals = self.measured.samples.iter().enumerate().map(|(idx, meas)| {
                    let theta_idx = idx % theta_step_count;
                    let theta = zenith.step(theta_idx);
                    let phi_idx = idx / theta_step_count;
                    let phi = azimuth.step(phi_idx);
                    self.model.eval_adf(theta.cos() as f64, phi.cos() as f64)
                        - *meas as f64 * self.scale as f64
                });
                Some(OMatrix::<f64, Dyn, U1>::from_iterator(
                    residuals.len(),
                    residuals,
                ))
            }
            AdfMeasurementMode::ByPartition { .. } => {
                // TODO: Implement this.
                None
            }
        }
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        match self.measured.params.mode {
            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
                let (cos_phis, cos_thetas) =
                    extract_azimuth_zenith_angles_cos(self.measured.samples.len(), azimuth, zenith);
                Some(OMatrix::<f64, Dyn, U2>::from_row_slice(
                    &self.model.adf_partial_derivatives(&cos_thetas, &cos_phis),
                ))
            }
            AdfMeasurementMode::ByPartition { .. } => None,
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2> for MaskingShadowingFittingProblemProxy<'a> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    impl_least_squares_problem_common_methods!(self, Vector<f64, U2, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        // let (phis, thetas) = extract_azimuth_zenith_angles(
        //     self.measured.samples.len(),
        //     self.measured.params.azimuth,
        //     self.measured.params.zenith,
        //     identity,
        // );
        // phis
        //     .iter()
        //     .map(|phi_m| {
        //         thetas.iter().map(|theta_m| {
        //             phis.iter().map(|phi_v| {
        //                 thetas.iter().map(|theta_v| {
        //                     let m = SphericalCoord::new(1.0, )
        //                     self.model
        //                         .eval_msf(*cos_theta_m, *cos_phi_m, *cos_theta_v,
        // *cos_phi_v)
        //                         * 0.25
        //                 })
        //             })
        //         })
        //     })
        //     .collect()
        todo!()
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> { todo!() }
}

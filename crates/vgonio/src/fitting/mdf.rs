use crate::{
    fitting::{FittingProblem, FittingReport},
    measure::{
        microfacet::{MeasuredAdfData, MeasuredMsfData},
        params::{AdfMeasurementMode, MsfMeasurementParams},
    },
};
use base::{math::sph_to_cart, range::RangeByStepSizeInclusive, units::Radians, Isotropy};
use bxdf::distro::{
    BeckmannDistribution, MicrofacetDistribution, MicrofacetDistroKind, TrowbridgeReitzDistribution,
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
/// The measured data can be either the normal (area) distribution function
/// (NDF) or the masking-shadowing function (MSF).
#[derive(Debug, Clone)]
pub enum MeasuredMdfData<'a> {
    /// The measured area distribution function (ADF).
    Ndf(Cow<'a, MeasuredAdfData>),
    /// The measured masking-shadowing function (MSF).
    Msf(Cow<'a, MeasuredMsfData>),
}

/// Fitting variant for the microfacet distribution function (MDF).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MicrofacetDistributionFittingVariant {
    /// Fitting the MDF through area distribution function (NDF).
    Ndf,
    /// Fitting the MDF through masking-shadowing function (MSF).
    Msf,
}

/// Fitting procedure trying to find different models for the microfacet
/// distribution function (MDF) that best fits the measured data.
///
/// The MDF models can be obtained by either fitting the measured area (normal)
/// distribution function (NDF) or the measured masking-shadowing function
/// (MSF).
///
/// The fitting procedure is based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetDistributionFittingProblem<'a> {
    /// Scaling factor for the measured data.
    scale: f32,
    /// The measured data to fit to.
    measured: MeasuredMdfData<'a>,
    /// The target model distribution kind.
    target: MicrofacetDistroKind,
}

impl<'a> Display for MicrofacetDistributionFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MDF-NDF-FittingProblem")
            .field("target", &self.target)
            .field("scale", &self.scale)
            .finish()
    }
}

impl<'a> MicrofacetDistributionFittingProblem<'a> {
    /// Creates a new NDF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured NDF data.
    /// * `target` - The target model.
    pub fn new_ndf_fitting(
        measured: &'a MeasuredAdfData,
        target: MicrofacetDistroKind,
        scale: f32,
    ) -> Self {
        Self {
            scale,
            measured: MeasuredMdfData::Ndf(Cow::Borrowed(measured)),
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
        target: MicrofacetDistroKind,
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
        target: MicrofacetDistroKind,
        scale: f32,
    ) -> Self {
        debug_assert_matches!(
            (method, &measured),
            (
                MicrofacetDistributionFittingVariant::Ndf,
                MeasuredMdfData::Ndf(_)
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
    type Model = Box<dyn MicrofacetDistribution<Params = [f64; 2]>>;

    fn lsq_lm_fit(self, isotropy: Isotropy) -> FittingReport<Self::Model> {
        println!("Fitting MDF with isotropy: {:?}", isotropy);
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let solver = LevenbergMarquardt::new();
        let mut result: Vec<(
            Box<dyn MicrofacetDistribution<Params = [f64; 2]>>,
            MinimizationReport<f64>,
        )> =
            {
                match self.measured {
                    MeasuredMdfData::Ndf(measured) => {
                        initialise_microfacet_mdf_models(0.001, 2.0, 32, self.target)
                            .into_par_iter()
                            .filter_map(|model| {
                                log::debug!(
                                    "Fitting with αx = {} αy = {}",
                                    model.params()[0],
                                    model.params()[1]
                                );
                                let (model, report) = match isotropy {
                                    Isotropy::Isotropic => {
                                        let problem =
                                            NdfFittingProblemProxy::<{ Isotropy::Isotropic }> {
                                                scale: self.scale,
                                                measured: measured.as_ref(),
                                                model,
                                            };
                                        let (result, report) = solver.minimize(problem);
                                        (result.model, report)
                                    }
                                    Isotropy::Anisotropic => {
                                        let problem =
                                            NdfFittingProblemProxy::<{ Isotropy::Anisotropic }> {
                                                scale: self.scale,
                                                measured: measured.as_ref(),
                                                model,
                                            };
                                        let (result, report) = solver.minimize(problem);
                                        (result.model, report)
                                    }
                                };
                                log::debug!(
                                    "Fitted αx = {} αy = {}, report: {:?}",
                                    model.params()[0],
                                    model.params()[1],
                                    report
                                );
                                match report.termination {
                                    TerminationReason::Converged { .. } => Some((model, report)),
                                    _ => None,
                                }
                            })
                            .collect()
                    }
                    MeasuredMdfData::Msf(measured) => {
                        initialise_microfacet_mdf_models(0.001, 2.0, 32, self.target)
                            .into_iter()
                            .filter_map(|model| {
                                log::debug!(
                                    "Fitting with αx = {} αy = {}",
                                    model.params()[0],
                                    model.params()[1]
                                );
                                let (model, report) = match isotropy {
                                    Isotropy::Isotropic => {
                                        let problem =
                                            MsfFittingProblemProxy::<{ Isotropy::Isotropic }> {
                                                measured: measured.as_ref(),
                                                model,
                                            };
                                        let (result, report) = solver.minimize(problem);
                                        (result.model, report)
                                    }
                                    Isotropy::Anisotropic => {
                                        let problem =
                                            MsfFittingProblemProxy::<{ Isotropy::Anisotropic }> {
                                                measured: measured.as_ref(),
                                                model,
                                            };
                                        let (result, report) = solver.minimize(problem);
                                        (result.model, report)
                                    }
                                };
                                log::debug!(
                                    "Fitted αx = {} αy = {}, report: {:?}",
                                    model.params()[0],
                                    model.params()[1],
                                    report
                                );
                                match report.termination {
                                    TerminationReason::Converged { .. } => Some((model, report)),
                                    _ => None,
                                }
                            })
                            .collect()
                    }
                }
            };
        result.shrink_to_fit();
        FittingReport::new(result, |m| m.params().iter().all(|p| p > &0.0))
    }
}

/// Proxy for the NDF fitting problem.
struct NdfFittingProblemProxy<'a, const I: Isotropy> {
    scale: f32,
    measured: &'a MeasuredAdfData,
    model: Box<dyn MicrofacetDistribution<Params = [f64; 2]>>,
}

/// Proxy for the MSF fitting problem.
struct MsfFittingProblemProxy<'a, const I: Isotropy> {
    measured: &'a MeasuredMsfData,
    model: Box<dyn MicrofacetDistribution<Params = [f64; 2]>>,
}

fn initialise_microfacet_mdf_models(
    min: f64,
    max: f64,
    num: u32,
    target: MicrofacetDistroKind,
) -> Vec<Box<dyn MicrofacetDistribution<Params = [f64; 2]>>> {
    let step = (max - min) / (num as f64);
    match target {
        MicrofacetDistroKind::Beckmann => (0..num)
            .map(|i| {
                Box::new(BeckmannDistribution::new(
                    (i + 1) as f64 * step,
                    (i + 1) as f64 * step,
                )) as _
            })
            .collect(),
        MicrofacetDistroKind::TrowbridgeReitz => (0..num)
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
impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for NdfFittingProblemProxy<'a, { Isotropy::Anisotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    impl_least_squares_problem_common_methods!(@aniso => self, Vector<f64, U2, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        match self.measured.params.mode {
            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
                let theta_step_count = zenith.step_count_wrapped();
                let residuals = self.measured.samples.iter().enumerate().map(|(idx, meas)| {
                    let theta_idx = idx % theta_step_count;
                    let theta = zenith.step(theta_idx);
                    let phi_idx = idx / theta_step_count;
                    let phi = azimuth.step(phi_idx);
                    self.model.eval_ndf(theta.cos() as f64, phi.cos() as f64)
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
                    &self.model.pd_ndf(&cos_thetas, &cos_phis),
                ))
            }
            // TODO: Implement this.
            AdfMeasurementMode::ByPartition { .. } => None,
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1> for NdfFittingProblemProxy<'a, { Isotropy::Isotropic }> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    impl_least_squares_problem_common_methods!(@iso2 => self, Vector<f64, U1, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        match self.measured.params.mode {
            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
                let theta_step_count = zenith.step_count_wrapped();
                let residuals = self.measured.samples.iter().enumerate().map(|(idx, meas)| {
                    let theta_idx = idx % theta_step_count;
                    let theta = zenith.step(theta_idx);
                    let phi_idx = idx / theta_step_count;
                    let phi = azimuth.step(phi_idx);
                    self.model.eval_ndf(theta.cos() as f64, phi.cos() as f64)
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

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        match self.measured.params.mode {
            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
                let (_, cos_thetas) =
                    extract_azimuth_zenith_angles_cos(self.measured.samples.len(), azimuth, zenith);
                Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
                    &self.model.pd_ndf_iso(&cos_thetas),
                ))
            }
            // TODO: Implement this.
            AdfMeasurementMode::ByPartition { .. } => None,
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for MsfFittingProblemProxy<'a, { Isotropy::Anisotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    impl_least_squares_problem_common_methods!(@aniso => self, Vector<f64, U2, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_iterator(
            self.measured.samples.len(),
            calc_msf_residuals(&self.measured, self.model.as_ref()),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(OMatrix::<f64, Dyn, U2>::from_row_slice(&calc_msf_jacobian(
            &self.measured,
            self.model.as_ref(),
        )))
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1> for MsfFittingProblemProxy<'a, { Isotropy::Isotropic }> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    impl_least_squares_problem_common_methods!(@iso2 => self, Vector<f64, U1, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_iterator(
            self.measured.samples.len(),
            calc_msf_residuals(&self.measured, self.model.as_ref()),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&calc_msf_jacobian(
            &self.measured,
            self.model.as_ref(),
        )))
    }
}

fn calc_msf_residuals<'a>(
    measured: &'a MeasuredMsfData,
    model: &'a dyn MicrofacetDistribution<Params = [f64; 2]>,
) -> impl IntoIterator<Item = f64> + 'a {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let phi_step_count = measured.params.azimuth.step_count_wrapped();
    measured.samples.iter().enumerate().map(move |(idx, meas)| {
        let theta_w_idx = idx % theta_step_count;
        let phi_w_idx = idx / theta_step_count;
        let theta_wm_idx = idx % (theta_step_count * phi_step_count);
        let phi_wm_idx = idx / (theta_step_count * phi_step_count);
        let theta_w = measured.params.zenith.step(theta_w_idx);
        let phi_w = measured.params.azimuth.step(phi_w_idx);
        let theta_wm = measured.params.zenith.step(theta_wm_idx);
        let phi_wm = measured.params.azimuth.step(phi_wm_idx);
        model.eval_msf1(sph_to_cart(theta_wm, phi_wm), sph_to_cart(theta_w, phi_w)) - *meas as f64
    })
}

fn calc_msf_jacobian(
    measured: &MeasuredMsfData,
    model: &dyn MicrofacetDistribution<Params = [f64; 2]>,
) -> Box<[f64]> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let phi_step_count = measured.params.azimuth.step_count_wrapped();
    let wms = (0..measured.samples.len())
        .map(|idx| {
            let theta_wm_idx = idx % (theta_step_count * phi_step_count);
            let phi_wm_idx = idx / (theta_step_count * phi_step_count);
            sph_to_cart(
                measured.params.zenith.step(theta_wm_idx),
                measured.params.azimuth.step(phi_wm_idx),
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let ws = (0..measured.samples.len())
        .map(|idx| {
            let theta_w_idx = idx % theta_step_count;
            let phi_w_idx = idx / theta_step_count;
            sph_to_cart(
                measured.params.zenith.step(theta_w_idx),
                measured.params.azimuth.step(phi_w_idx),
            )
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    model.pd_msf1(&wms, &ws)
}

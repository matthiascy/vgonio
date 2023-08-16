use crate::{
    fitting::{MicrofacetAreaDistributionModel, MicrofacetModelFamily, ReflectionModelFamily},
    measure::microfacet::MeasuredMadfData,
};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{Dim, Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, Vector1, U1, U2};
use std::{fmt::Display, marker::PhantomData};
use vgcore::math::{DVec3, Handedness, SphericalCoord, Vec3};

/// Describes how the measured data is used to fit the model.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AreaDistributionFittingMode {
    #[default]
    /// Fit the model to the whole measured data.
    Complete,
    /// Accumulate the measured data of each azimuthal angle for different
    /// zenith angle and fit the model.
    Accumulated,
}

impl AreaDistributionFittingMode {
    pub fn as_suffix_str(&self) -> &'static str {
        match self {
            Self::Complete => "",
            Self::Accumulated => "(Accumulated)",
        }
    }
}

/// Fitting procedure trying to find the width parameter for the microfacet area
/// (normal) distribution function (NDF) of the Trowbridge-Reitz (GGX) model.
///
/// Use Levenberg-Marquardt to fit this parameter to measured data.
/// The problem has 1 parameter and 1 residual.
pub struct AreaDistributionFittingProblem<'a> {
    /// The measured data to fit to.
    measured: &'a MeasuredMadfData,
    /// The normal vector of the microfacet.
    normal: Vec3,
    /// The target model.
    model: Box<dyn MicrofacetAreaDistributionModel>,
    /// The fitting mode.
    mode: AreaDistributionFittingMode,
}

impl<'a> Display for AreaDistributionFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MadfFittingProblem")
            .field("target_model", &self.model.family())
            .finish()
    }
}

impl<'a> AreaDistributionFittingProblem<'a> {
    /// Creates a new fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured data to fit to.
    /// * `model` - The target model.
    /// * `n` - The normal vector of the microfacet.
    pub fn new<M: MicrofacetAreaDistributionModel + 'static>(
        measured: &'a MeasuredMadfData,
        model: M,
        n: Vec3,
        mode: AreaDistributionFittingMode,
    ) -> Self {
        Self {
            measured,
            normal: n,
            model: Box::new(model),
            mode,
        }
    }
}

impl<'a> FittingProblem for AreaDistributionFittingProblem<'a> {
    type Model = Box<dyn MicrofacetAreaDistributionModel>;

    fn lsq_lm_fit(self) -> (Self::Model, MinimizationReport<f64>) {
        let effective_params_count = self.model.effective_params_count();
        if effective_params_count == 1 {
            #[cfg(not(feature = "scaled-adf-fitting"))]
            {
                let problem = AreaDistributionFittingProblemProxy::<U1> {
                    measured: self.measured,
                    normal: self.normal.as_dvec3(),
                    model: self.model.clone_box(),
                    mode: self.mode,
                    marker: Default::default(),
                };
                let solver = LevenbergMarquardt::new();
                let (result, report) = solver.minimize(problem);
                (result.model, report)
            }
            #[cfg(feature = "scaled-adf-fitting")]
            {
                if self.model.scale().is_some() {
                    let problem = AreaDistributionFittingProblemProxy::<U1, true> {
                        measured: self.measured,
                        normal: self.normal.as_dvec3(),
                        model: self.model.clone_box(),
                        mode: self.mode,
                        marker: Default::default(),
                    };
                    let solver = LevenbergMarquardt::new();
                    let (result, report) = solver.minimize(problem);
                    (result.model, report)
                } else {
                    let problem = AreaDistributionFittingProblemProxy::<U1, false> {
                        measured: self.measured,
                        normal: self.normal.as_dvec3(),
                        model: self.model.clone_box(),
                        mode: self.mode,
                        marker: Default::default(),
                    };
                    let solver = LevenbergMarquardt::new();
                    let (result, report) = solver.minimize(problem);
                    (result.model, report)
                }
            }
        } else if effective_params_count == 2 {
            todo!("2 effective params count")
        } else {
            todo!("3 effective params count")
        }
    }
}

const SCALED: bool = true;
const UNSCALED: bool = false;

#[cfg(feature = "scaled-adf-fitting")]
/// The inner fitting problem for the microfacet area distribution function of
/// different effective parameters count.
///
/// If the feature `scaled-adf-fitting` is enabled, and the model is created
/// with a non-none scale factor, the fitting problem is a N + 1 parameter
/// fitting problem, where N is the number of effective parameters of the model,
/// and the additional parameter is the scale factor.
///
/// Otherwise, the fitting problem is a N parameter fitting problem. For models
/// with 1 effective parameter (isotropic models), the fitting problem is
/// simplified to a single parameter fitting problem. For models with 2
/// effective parameters (anisotropic models), the fitting problem is a 2
/// parameter fitting problem.
struct AreaDistributionFittingProblemProxy<'a, N: Dim, const Scaled: bool> {
    measured: &'a MeasuredMadfData,
    normal: DVec3,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
    marker: PhantomData<N>,
}

#[cfg(not(feature = "scaled-adf-fitting"))]
/// The inner fitting problem for the microfacet area distribution function of
/// different effective parameters count. `N` is the number of effective
/// parameters of the model, this is introduced because the levenberg_marquardt
/// crate requires the parameter count to be a type parameter.
struct AreaDistributionFittingProblemProxy<'a, N: Dim> {
    measured: &'a MeasuredMadfData,
    normal: DVec3,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
    marker: PhantomData<N>,
}

/// Calculates the residuals of the evaluated model against the whole measured
/// data.
fn calculate_madf_residuals_complete(
    measured: &MeasuredMadfData,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let residuals = measured
        .samples
        .iter()
        .enumerate()
        .map(|(idx, meas)| {
            let phi_idx = idx / theta_step_count;
            let theta_idx = idx % theta_step_count;
            let theta = measured.params.zenith.step(theta_idx);
            let phi = measured.params.azimuth.step(phi_idx);
            let m = SphericalCoord::new(1.0, theta, phi)
                .to_cartesian(Handedness::RightHandedYUp)
                .as_dvec3();
            model.eval(m, normal) - *meas as f64
        })
        .collect();
    OMatrix::<f64, Dyn, U1>::from_vec(residuals)
}

fn calculate_madf_residuals_accumulated(
    measured: &MeasuredMadfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let theta_step_size = measured.params.zenith.step_size;
    OMatrix::<f64, Dyn, U1>::from_iterator(
        (theta_step_count * 2 - 1),
        measured
            .accumulated_slice()
            .iter()
            .map(|(theta, val)| model.eval_with_theta_m(*theta as f64) - *val as f64),
    )
}

#[cfg(feature = "scaled-adf-fitting")]
/// Calculates the residuals of the scaled evaluated model against the measured
/// data.
fn calculate_scaled_madf_residuals_complete(
    measured: &MeasuredMadfData,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let scale = model.scale().unwrap();
    let residuals = measured
        .samples
        .iter()
        .enumerate()
        .map(|(idx, meas)| {
            let phi_idx = idx / theta_step_count;
            let theta_idx = idx % theta_step_count;
            let theta = measured.params.zenith.step(theta_idx);
            let phi = measured.params.azimuth.step(phi_idx);
            let m = SphericalCoord::new(1.0, theta, phi)
                .to_cartesian(Handedness::RightHandedYUp)
                .as_dvec3();
            model.eval(m, normal) * scale - *meas as f64
        })
        .collect();
    OMatrix::<f64, Dyn, U1>::from_vec(residuals)
}

#[cfg(feature = "scaled-adf-fitting")]
/// Calculates the residuals of the scaled evaluated model against the measured
/// data.
fn calculate_scaled_madf_residuals_accumulated(
    measured: &MeasuredMadfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let scale = model.scale().unwrap_or(1.0);
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    OMatrix::<f64, Dyn, U1>::from_iterator(
        (theta_step_count * 2 - 1),
        measured
            .accumulated_slice()
            .iter()
            .map(|(theta, val)| model.eval_with_theta_m(*theta as f64) * scale - *val as f64),
    )
}

/// Calculates the partial derivative of the fitting parameter.
fn calculate_single_param_madf_jacobian(
    measured: &MeasuredMadfData,
    alpha: f64,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let alpha2 = alpha * alpha;
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let derivatives = match model.family() {
        ReflectionModelFamily::Microfacet(m) => match m {
            MicrofacetModelFamily::TrowbridgeReitz => measured
                .samples
                .iter()
                .enumerate()
                .map(move |(idx, meas)| {
                    let phi_idx = idx / theta_step_count;
                    let theta_idx = idx % theta_step_count;
                    let theta = measured.params.zenith.step(theta_idx);
                    let phi = measured.params.azimuth.step(phi_idx);
                    let m = SphericalCoord::new(1.0, theta, phi)
                        .to_cartesian(Handedness::RightHandedYUp)
                        .as_dvec3();
                    let cos_theta_m = m.dot(normal);
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    let numerator =
                        2.0 * alpha * (tan_theta_m2 - alpha2) * sec_theta_m2 * sec_theta_m2;
                    let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);

                    numerator / denominator
                })
                .collect(),
            MicrofacetModelFamily::BeckmannSpizzichino => measured
                .samples
                .iter()
                .enumerate()
                .map(move |(idx, meas)| {
                    let phi_idx = idx / theta_step_count;
                    let theta_idx = idx % theta_step_count;
                    let theta = measured.params.zenith.step(theta_idx);
                    let phi = measured.params.azimuth.step(phi_idx);
                    let m = SphericalCoord::new(1.0, theta, phi)
                        .to_cartesian(Handedness::RightHandedYUp)
                        .as_dvec3();
                    let cos_theta_m = m.dot(normal);
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    let numerator = 2.0
                        * (-tan_theta_m2 / alpha2).exp()
                        * (tan_theta_m2 - alpha2)
                        * sec_theta_m2
                        * sec_theta_m2;
                    let denominator = std::f64::consts::PI * alpha.powi(5);

                    numerator / denominator
                })
                .collect(),
        },
    };
    OMatrix::<f64, Dyn, U1>::from_vec(derivatives)
}

/// Calculates the partial derivative of the fitting parameter.
fn calculate_single_param_madf_jacobian_accumulated(
    measured: &MeasuredMadfData,
    alpha: f64,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let alpha2 = alpha * alpha;
    let derivatives = match model.family() {
        ReflectionModelFamily::Microfacet(m) => match m {
            MicrofacetModelFamily::TrowbridgeReitz => measured
                .accumulated_slice()
                .iter()
                .map(move |(theta, _)| {
                    let cos_theta_m = theta.cos() as f64;
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    let numerator =
                        2.0 * alpha * (tan_theta_m2 - alpha2) * sec_theta_m2 * sec_theta_m2;
                    let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);

                    numerator / denominator
                })
                .collect(),
            MicrofacetModelFamily::BeckmannSpizzichino => measured
                .accumulated_slice()
                .iter()
                .map(move |(theta, _)| {
                    let cos_theta_m = theta.cos() as f64;
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    let numerator = 2.0
                        * (-tan_theta_m2 / alpha2).exp()
                        * (tan_theta_m2 - alpha2)
                        * sec_theta_m2
                        * sec_theta_m2;
                    let denominator = std::f64::consts::PI * alpha.powi(5);

                    numerator / denominator
                })
                .collect(),
        },
    };
    OMatrix::<f64, Dyn, U1>::from_vec(derivatives)
}

#[cfg(feature = "scaled-adf-fitting")]
fn calculate_scaled_madf_jacobian_complete(
    measured: &MeasuredMadfData,
    alpha: f64,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U2> {
    let alpha2 = alpha * alpha;
    let scale = model.scale().unwrap();
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let derivatives: Vec<f64> = match model.family() {
        ReflectionModelFamily::Microfacet(m) => match m {
            MicrofacetModelFamily::TrowbridgeReitz => measured
                .samples
                .iter()
                .enumerate()
                .flat_map(move |(idx, meas)| {
                    let phi_idx = idx / theta_step_count;
                    let theta_idx = idx % theta_step_count;
                    let theta = measured.params.zenith.step(theta_idx);
                    let phi = measured.params.azimuth.step(phi_idx);
                    let m = SphericalCoord::new(1.0, theta, phi)
                        .to_cartesian(Handedness::RightHandedYUp)
                        .as_dvec3();
                    let cos_theta_m = m.dot(normal);
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let sec_theta_m4 = sec_theta_m2 * sec_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    // Derivative of the Trowbridge-Reitz distribution with respect to alpha
                    let d_alpha = {
                        let numerator =
                            2.0 * alpha * scale * (tan_theta_m2 - alpha2) * sec_theta_m4;
                        let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);
                        numerator / denominator
                    };
                    let d_scale = {
                        let numerator = alpha2 * sec_theta_m4;
                        let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(2);
                        numerator / denominator
                    };
                    [d_alpha, d_scale]
                })
                .collect(),
            MicrofacetModelFamily::BeckmannSpizzichino => measured
                .samples
                .iter()
                .enumerate()
                .flat_map(move |(idx, meas)| {
                    let phi_idx = idx / measured.params.zenith.step_count_wrapped();
                    let theta_idx = idx % measured.params.zenith.step_count_wrapped();
                    let theta = measured.params.zenith.step(theta_idx);
                    let phi = measured.params.azimuth.step(phi_idx);
                    let m = SphericalCoord::new(1.0, theta, phi)
                        .to_cartesian(Handedness::RightHandedYUp)
                        .as_dvec3();
                    let cos_theta_m = m.dot(normal);
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let sec_theta_m4 = sec_theta_m2 * sec_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    let d_alpha = {
                        let numerator = 2.0
                            * (-tan_theta_m2 / alpha2).exp()
                            * (tan_theta_m2 - alpha2)
                            * sec_theta_m4
                            * scale;
                        let denominator = std::f64::consts::PI * alpha.powi(5);

                        numerator / denominator
                    };

                    let d_scale = {
                        let numerator = (-tan_theta_m2 / alpha2).exp() * sec_theta_m4;
                        let denominator = std::f64::consts::PI * alpha2;
                        numerator / denominator
                    };
                    [d_alpha, d_scale]
                })
                .collect(),
        },
    };
    OMatrix::<f64, Dyn, U2>::from_row_slice(&derivatives)
}

#[cfg(feature = "scaled-adf-fitting")]
fn calculate_scaled_madf_jacobian_accumulated(
    measured: &MeasuredMadfData,
    alpha: f64,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U2> {
    let alpha2 = alpha * alpha;
    let scale = model.scale().unwrap();
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let derivatives: Vec<f64> = match model.family() {
        ReflectionModelFamily::Microfacet(m) => match m {
            MicrofacetModelFamily::TrowbridgeReitz => measured
                .accumulated_slice()
                .iter()
                .flat_map(move |(theta, _)| {
                    let cos_theta_m = theta.cos() as f64;
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let sec_theta_m4 = sec_theta_m2 * sec_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    // Derivative of the Trowbridge-Reitz distribution with respect to alpha
                    let d_alpha = {
                        let numerator =
                            2.0 * alpha * scale * (tan_theta_m2 - alpha2) * sec_theta_m4;
                        let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);
                        numerator / denominator
                    };
                    let d_scale = {
                        let numerator = alpha2 * sec_theta_m4;
                        let denominator = std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(2);
                        numerator / denominator
                    };
                    [d_alpha, d_scale]
                })
                .collect(),
            MicrofacetModelFamily::BeckmannSpizzichino => measured
                .accumulated_slice()
                .iter()
                .flat_map(move |(theta, _)| {
                    let cos_theta_m = theta.cos() as f64;
                    let cos_theta_m2 = cos_theta_m * cos_theta_m;
                    let sec_theta_m2 = 1.0 / cos_theta_m2;
                    let sec_theta_m4 = sec_theta_m2 * sec_theta_m2;
                    let tan_theta_m2 = if cos_theta_m2 == 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 / cos_theta_m2 - cos_theta_m2
                    };

                    let d_alpha = {
                        let numerator = 2.0
                            * (-tan_theta_m2 / alpha2).exp()
                            * (tan_theta_m2 - alpha2)
                            * sec_theta_m4
                            * scale;
                        let denominator = std::f64::consts::PI * alpha.powi(5);

                        numerator / denominator
                    };

                    let d_scale = {
                        let numerator = (-tan_theta_m2 / alpha2).exp() * sec_theta_m4;
                        let denominator = std::f64::consts::PI * alpha2;
                        numerator / denominator
                    };
                    [d_alpha, d_scale]
                })
                .collect(),
        },
    };
    OMatrix::<f64, Dyn, U2>::from_row_slice(&derivatives)
}

// When the feature `scaled-adf-fitting` is not enabled, the fitting problem
// is simplified to a single parameter fitting problem.
#[cfg(not(feature = "scaled-adf-fitting"))]
impl<'a> LeastSquaresProblem<f64, Dyn, U1> for AreaDistributionFittingProblemProxy<'a, U1> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) { self.model.set_params([x[0], x[0]]) }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.params()[0]) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        match self.mode {
            AreaDistributionFittingMode::Complete => Some(calculate_madf_residuals_complete(
                self.measured,
                self.normal,
                &self.model,
            )),
            AreaDistributionFittingMode::Accumulated => Some(calculate_madf_residuals_accumulated(
                self.measured,
                &self.model,
            )),
        }
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        match self.mode {
            AreaDistributionFittingMode::Complete => Some(calculate_single_param_madf_jacobian(
                self.measured,
                self.params().x,
                self.normal,
                &self.model,
            )),
            AreaDistributionFittingMode::Accumulated => {
                Some(calculate_single_param_madf_jacobian_accumulated(
                    self.measured,
                    self.params().x,
                    &self.model,
                ))
            }
        }
    }
}

#[cfg(not(feature = "scaled-adf-fitting"))]
impl<'a> LeastSquaresProblem<f64, Dyn, U2> for AreaDistributionFittingProblemProxy<'a, U2> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) { todo!() }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> { todo!() }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(calculate_madf_residuals_complete(
            self.measured,
            self.normal,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> { todo!() }
}

use crate::fitting::FittingProblem;
#[cfg(feature = "scaled-adf-fitting")]
use nalgebra::U3;
use vgcore::units::Radians;

#[cfg(feature = "scaled-adf-fitting")]
/// Isotropic MADF fitting problem without the scaling factor.
impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for AreaDistributionFittingProblemProxy<'a, U1, UNSCALED>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) { self.model.set_params([x[0], x[0]]) }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.params()[0]) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        match self.mode {
            AreaDistributionFittingMode::Complete => Some(calculate_madf_residuals_complete(
                self.measured,
                self.normal,
                &self.model,
            )),
            AreaDistributionFittingMode::Accumulated => Some(calculate_madf_residuals_accumulated(
                self.measured,
                &self.model,
            )),
        }
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        match self.mode {
            AreaDistributionFittingMode::Complete => Some(calculate_single_param_madf_jacobian(
                self.measured,
                self.params().x,
                self.normal,
                &self.model,
            )),
            AreaDistributionFittingMode::Accumulated => {
                Some(calculate_single_param_madf_jacobian_accumulated(
                    self.measured,
                    self.params().x,
                    &self.model,
                ))
            }
        }
    }
}

#[cfg(feature = "scaled-adf-fitting")]
/// Isotropic MADF fitting problem with a scaling factor.
impl<'a> LeastSquaresProblem<f64, Dyn, U2> for AreaDistributionFittingProblemProxy<'a, U1, SCALED> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params([x[0], x[0]]);
        self.model.set_scale(x[1]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::new(
            self.model.params()[0],
            self.model.scale().unwrap(),
        )
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        match self.mode {
            AreaDistributionFittingMode::Complete => Some(
                calculate_scaled_madf_residuals_complete(self.measured, self.normal, &self.model),
            ),
            AreaDistributionFittingMode::Accumulated => Some(
                calculate_scaled_madf_residuals_accumulated(self.measured, &self.model),
            ),
        }
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        match self.mode {
            AreaDistributionFittingMode::Complete => Some(calculate_scaled_madf_jacobian_complete(
                self.measured,
                self.params().x,
                self.normal,
                &self.model,
            )),
            AreaDistributionFittingMode::Accumulated => {
                Some(calculate_scaled_madf_jacobian_accumulated(
                    self.measured,
                    self.params().x,
                    &self.model,
                ))
            }
        }
    }
}

#[cfg(feature = "scaled-adf-fitting")]
/// Anisotropic MADF fitting problem without the scaling factor.
impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for AreaDistributionFittingProblemProxy<'a, U2, UNSCALED>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params([x[0], x[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(self.model.params())
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> { todo!() }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> { todo!() }
}

#[cfg(feature = "scaled-adf-fitting")]
/// Anisotropic MADF fitting problem with a scaling factor.
impl<'a> LeastSquaresProblem<f64, Dyn, U3> for AreaDistributionFittingProblemProxy<'a, U2, SCALED> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U3>;
    type ParameterStorage = Owned<f64, U3, U1>;

    fn set_params(&mut self, x: &Vector<f64, U3, Self::ParameterStorage>) {
        self.model.set_params([x[0], x[1]]);
        self.model.set_scale(x[2]);
    }

    fn params(&self) -> Vector<f64, U3, Self::ParameterStorage> {
        Vector::<f64, U3, Self::ParameterStorage>::new(
            self.model.params()[0],
            self.model.params()[1],
            self.model.scale().unwrap(),
        )
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> { todo!() }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U3, Self::JacobianStorage>> { todo!() }
}

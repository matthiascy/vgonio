use crate::{
    fitting::{MicrofacetAreaDistributionModel, MicrofacetModelFamily, ReflectionModelFamily},
    measure::microfacet::MeasuredMndfData,
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
    measured: &'a MeasuredMndfData,
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
        measured: &'a MeasuredMndfData,
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
        if self.model.is_isotropic() {
            #[cfg(not(feature = "scaled-ndf-fitting"))]
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
            #[cfg(feature = "scaled-ndf-fitting")]
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
        } else {
            #[cfg(not(feature = "scaled-ndf-fitting"))]
            {
                let problem = AreaDistributionFittingProblemProxy::<U2> {
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
            #[cfg(feature = "scaled-ndf-fitting")]
            {
                if self.model.scale().is_some() {
                    let problem = AreaDistributionFittingProblemProxy::<U2, SCALED> {
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
                    let problem = AreaDistributionFittingProblemProxy::<U2, UNSCALED> {
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
        }
    }
}

const SCALED: bool = true;
const UNSCALED: bool = false;

#[cfg(feature = "scaled-ndf-fitting")]
/// The inner fitting problem for the microfacet area distribution function of
/// different effective parameters count.
///
/// If the feature `scaled-ndf-fitting` is enabled, and the model is created
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
    measured: &'a MeasuredMndfData,
    normal: DVec3,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
    marker: PhantomData<N>,
}

#[cfg(not(feature = "scaled-ndf-fitting"))]
/// The inner fitting problem for the microfacet area distribution function of
/// different effective parameters count. `N` is the number of effective
/// parameters of the model, this is introduced because the levenberg_marquardt
/// crate requires the parameter count to be a type parameter.
struct AreaDistributionFittingProblemProxy<'a, N: Dim> {
    measured: &'a MeasuredMndfData,
    normal: DVec3,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
    marker: PhantomData<N>,
}

/// Calculates the residuals of the evaluated model against the whole measured
/// data.
fn calculate_mndf_residuals(
    measured: &MeasuredMndfData,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
) -> OMatrix<f64, Dyn, U1> {
    match mode {
        AreaDistributionFittingMode::Complete => {
            let theta_step_count = measured.params.zenith.step_count_wrapped();
            let residuals = measured.samples.iter().enumerate().map(|(idx, meas)| {
                let phi_idx = idx / theta_step_count;
                let theta_idx = idx % theta_step_count;
                let theta = measured.params.zenith.step(theta_idx);
                let phi = measured.params.azimuth.step(phi_idx);
                let m = SphericalCoord::new(1.0, theta, phi)
                    .to_cartesian(Handedness::RightHandedYUp)
                    .as_dvec3();
                model.eval(m, normal) - *meas as f64
            });

            OMatrix::<f64, Dyn, U1>::from_iterator(residuals.len(), residuals)
        }
        AreaDistributionFittingMode::Accumulated => {
            let theta_step_count = measured.params.zenith.step_count_wrapped();
            OMatrix::<f64, Dyn, U1>::from_iterator(
                (theta_step_count * 2 - 1),
                measured
                    .accumulated_slice()
                    .iter()
                    .map(|(theta, val)| model.eval_with_theta_m(*theta as f64) - *val as f64),
            )
        }
    }
}

#[cfg(feature = "scaled-ndf-fitting")]
/// Calculates the residuals of the scaled evaluated model against the measured
/// data.
fn calculate_scaled_mndf_residuals(
    measured: &MeasuredMndfData,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let scale = model.scale().unwrap();
    match mode {
        AreaDistributionFittingMode::Complete => {
            let residuals = measured.samples.iter().enumerate().map(|(idx, meas)| {
                let phi_idx = idx / theta_step_count;
                let theta_idx = idx % theta_step_count;
                let theta = measured.params.zenith.step(theta_idx);
                let phi = measured.params.azimuth.step(phi_idx);
                let m = SphericalCoord::new(1.0, theta, phi)
                    .to_cartesian(Handedness::RightHandedYUp)
                    .as_dvec3();
                model.eval(m, normal) * scale - *meas as f64
            });
            OMatrix::<f64, Dyn, U1>::from_iterator(residuals.len(), residuals)
        }
        AreaDistributionFittingMode::Accumulated => OMatrix::<f64, Dyn, U1>::from_iterator(
            (theta_step_count * 2 - 1),
            measured
                .accumulated_slice()
                .iter()
                .map(|(theta, val)| model.eval_with_theta_m(*theta as f64) * scale - *val as f64),
        ),
    }
}

/// Calculates the partial derivative of the fitting parameter.
fn calculate_isotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let cos_theta_ms = match mode {
        AreaDistributionFittingMode::Complete => {
            if normal == DVec3::Y {
                measured
                    .samples
                    .iter()
                    .enumerate()
                    .map(move |(idx, meas)| {
                        let theta_idx = idx % theta_step_count;
                        measured.params.zenith.step(theta_idx).cos() as f64
                    })
                    .collect::<Vec<_>>()
            } else {
                measured
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
                        m.dot(normal)
                    })
                    .collect::<Vec<_>>()
            }
        }
        AreaDistributionFittingMode::Accumulated => measured
            .accumulated_slice()
            .iter()
            .map(|(theta, _)| theta.cos() as f64)
            .collect::<Vec<_>>(),
    };
    OMatrix::<f64, Dyn, U1>::from_vec(model.calc_param_pd_isotropic(&cos_theta_ms))
}

fn calculate_anisotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    normal: DVec3,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
) -> OMatrix<f64, Dyn, U2> {
    assert!(
        matches!(mode, AreaDistributionFittingMode::Complete),
        "Anisotropic MADF fitting only supports complete mode"
    );
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let cos_theta_ms = if normal == DVec3::Y {
        measured
            .samples
            .iter()
            .enumerate()
            .map(move |(idx, meas)| {
                let theta_idx = idx % theta_step_count;
                measured.params.zenith.step(theta_idx).cos() as f64
            })
            .collect::<Vec<_>>()
    } else {
        measured
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
                m.dot(normal)
            })
            .collect::<Vec<_>>()
    };
    OMatrix::<f64, Dyn, U2>::from_row_slice(&model.calc_param_pd_anisotropic(&cos_theta_ms))
}

#[cfg(feature = "scaled-ndf-fitting")]
fn calculate_scaled_isotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
) -> OMatrix<f64, Dyn, U2> {
    let cos_theta_ms = match mode {
        AreaDistributionFittingMode::Complete => {
            let theta_step_count = measured.params.zenith.step_count_wrapped();
            measured
                .samples
                .iter()
                .enumerate()
                .map(move |(idx, meas)| {
                    let theta_idx = idx % theta_step_count;
                    measured.params.zenith.step(theta_idx).cos() as f64
                })
                .collect::<Vec<_>>()
        }
        AreaDistributionFittingMode::Accumulated => measured
            .accumulated_slice()
            .iter()
            .map(|(theta, _)| theta.cos() as f64)
            .collect::<Vec<_>>(),
    };
    OMatrix::<f64, Dyn, U2>::from_row_slice(&model.calc_param_pd_isotropic_scaled(&cos_theta_ms))
}

#[cfg(feature = "scaled-ndf-fitting")]
fn calculate_scaled_anisotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
    mode: AreaDistributionFittingMode,
) -> OMatrix<f64, Dyn, U3> {
    assert!(
        matches!(mode, AreaDistributionFittingMode::Complete),
        "Anisotropic MADF fitting only supports complete mode"
    );
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let cos_theta_ms = measured
        .samples
        .iter()
        .enumerate()
        .map(move |(idx, meas)| {
            let theta_idx = idx % theta_step_count;
            measured.params.zenith.step(theta_idx).cos() as f64
        })
        .collect::<Vec<_>>();
    OMatrix::<f64, Dyn, U3>::from_row_slice(&model.calc_param_pd_anisotropic_scaled(&cos_theta_ms))
}

// When the feature `scaled-ndf-fitting` is not enabled, the fitting problem
// is simplified to a single parameter fitting problem.
#[cfg(not(feature = "scaled-ndf-fitting"))]
impl<'a> LeastSquaresProblem<f64, Dyn, U1> for AreaDistributionFittingProblemProxy<'a, U1> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) { self.model.set_params([x[0], x[0]]) }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.params()[0]) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_mndf_residuals(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(calculate_isotropic_mndf_jacobian(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }
}

#[cfg(not(feature = "scaled-ndf-fitting"))]
/// Anisotropic MADF fitting problem.
impl<'a> LeastSquaresProblem<f64, Dyn, U2> for AreaDistributionFittingProblemProxy<'a, U2> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params([x[0], x[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(self.model.params())
    }
}

use crate::fitting::FittingProblem;
#[cfg(feature = "scaled-ndf-fitting")]
use nalgebra::U3;
use vgcore::units::Radians;

#[cfg(feature = "scaled-ndf-fitting")]
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
        Some(calculate_mndf_residuals(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(calculate_isotropic_mndf_jacobian(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }
}

#[cfg(feature = "scaled-ndf-fitting")]
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
        Some(calculate_scaled_mndf_residuals(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(calculate_scaled_isotropic_mndf_jacobian(
            self.measured,
            &self.model,
            self.mode,
        ))
    }
}

#[cfg(feature = "scaled-ndf-fitting")]
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

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(calculate_mndf_residuals(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(calculate_anisotropic_mndf_jacobian(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }
}

#[cfg(feature = "scaled-ndf-fitting")]
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

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(calculate_scaled_mndf_residuals(
            self.measured,
            self.normal,
            &self.model,
            self.mode,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U3, Self::JacobianStorage>> {
        Some(calculate_scaled_anisotropic_mndf_jacobian(
            self.measured,
            &self.model,
            self.mode,
        ))
    }
}

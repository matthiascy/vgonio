use crate::{
    fitting::{
        BeckmannSpizzichinoNDF, MicrofacetAreaDistributionModel, MicrofacetModelFamily,
        ReflectionModelFamily,
    },
    measure::microfacet::MeasuredMndfData,
    RangeByStepSizeInclusive,
};
use levenberg_marquardt::{
    LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use nalgebra::{Dim, Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, Vector1, U1, U2};
use std::{fmt::Display, marker::PhantomData};
use vgcore::math::{DVec3, Handedness, SphericalCoord, Vec3};

/// Fitting procedure trying to find the width parameter for the microfacet area
/// (normal) distribution function (NDF) of the Trowbridge-Reitz (GGX) model.
///
/// Use Levenberg-Marquardt to fit this parameter to measured data.
/// The problem has 1 parameter and 1 residual.
pub struct AreaDistributionFittingProblem<'a> {
    /// The measured data to fit to.
    measured: &'a MeasuredMndfData,
    /// The target model family.
    family: ReflectionModelFamily,
    /// Isotropy of the model.
    isotropy: Isotropy,
    #[cfg(feature = "scaled-ndf-fitting")]
    scaled: bool,
}

impl<'a> Display for AreaDistributionFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MadfFittingProblem")
            .field("target_model", &self.family)
            .field("isotropy", &self.isotropy)
            .finish()
    }
}

impl<'a> AreaDistributionFittingProblem<'a> {
    /// Creates a new fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured data to fit to.
    /// * `family` - The target model family.
    /// * `isotropy` - Isotropy of the model.
    pub fn new(
        measured: &'a MeasuredMndfData,
        family: ReflectionModelFamily,
        isotropy: Isotropy,
        #[cfg(feature = "scaled-ndf-fitting")] scaled: bool,
    ) -> Self {
        Self {
            measured,
            family,
            isotropy,
            #[cfg(feature = "scaled-ndf-fitting")]
            scaled,
        }
    }
}

fn initiate_microfacet_ndf_models(
    num: u32,
    family: MicrofacetModelFamily,
    isotropy: Isotropy,
    #[cfg(feature = "scaled-ndf-fitting")] scaled: bool,
) -> Vec<Box<dyn MicrofacetAreaDistributionModel>> {
    match isotropy {
        Isotropy::Isotropic => {
            #[cfg(not(feature = "scaled-ndf-fitting"))]
            match family {
                MicrofacetModelFamily::TrowbridgeReitz => (0..num)
                    .map(|i| Box::new(TrowbridgeReitzNDF::new((i + 1) as f64 * 0.1)))
                    .collect(),
                MicrofacetModelFamily::BeckmannSpizzichino => (0..num)
                    .map(|i| Box::new(BeckmannSpizzichinoNDF::new((i + 1) as f64 * 0.1)))
                    .collect(),
            }
            #[cfg(feature = "scaled-ndf-fitting")]
            {
                let scale = if scaled { Some(1.0) } else { None };
                match family {
                    MicrofacetModelFamily::TrowbridgeReitz => (0..num)
                        .map(|i| {
                            Box::new(TrowbridgeReitzNDF::new((i + 1) as f64 * 0.1, scale)) as _
                        })
                        .collect(),
                    MicrofacetModelFamily::BeckmannSpizzichino => (0..num)
                        .map(|i| {
                            Box::new(BeckmannSpizzichinoNDF::new((i + 1) as f64 * 0.1, scale)) as _
                        })
                        .collect(),
                }
            }
        }
        Isotropy::Anisotropic => {
            #[cfg(not(feature = "scaled-ndf-fitting"))]
            match family {
                MicrofacetModelFamily::TrowbridgeReitz => (0..num)
                    .map(|i| {
                        Box::new(TrowbridgeReitzAnisotropicNDF::new(
                            (i + 1) as f64 * 0.1,
                            (i + 1) as f64 * 0.1,
                        )) as _
                    })
                    .collect(),
                MicrofacetModelFamily::BeckmannSpizzichino => (0..num)
                    .map(|i| {
                        Box::new(BeckmannSpizzichinoAnisotropicNDF::new(
                            (i + 1) as f64 * 0.1,
                            (i + 1) as f64 * 0.1,
                        )) as _
                    })
                    .collect(),
            }
            #[cfg(feature = "scaled-ndf-fitting")]
            {
                let scale = if scaled { Some(1.0) } else { None };
                match family {
                    MicrofacetModelFamily::TrowbridgeReitz => (0..num)
                        .map(|i| {
                            Box::new(TrowbridgeReitzAnisotropicNDF::new(
                                (i + 1) as f64 * 0.1,
                                (i + 1) as f64 * 0.1,
                                scale,
                            )) as _
                        })
                        .collect(),
                    MicrofacetModelFamily::BeckmannSpizzichino => (0..num)
                        .map(|i| {
                            Box::new(BeckmannSpizzichinoAnisotropicNDF::new(
                                (i + 1) as f64 * 0.1,
                                (i + 1) as f64 * 0.1,
                                scale,
                            )) as _
                        })
                        .collect(),
                }
            }
        }
    }
}

impl<'a> FittingProblem for AreaDistributionFittingProblem<'a> {
    type Model = Box<dyn MicrofacetAreaDistributionModel>;

    fn lsq_lm_fit(self) -> FittingReport<Self::Model> {
        let solver = LevenbergMarquardt::new();
        match self.family {
            ReflectionModelFamily::Microfacet(family) => {
                #[cfg(not(feature = "scaled-ndf-fitting"))]
                let mut result = {
                    let models = initiate_microfacet_ndf_models(10, family, self.isotropy);
                    match self.isotropy {
                        Isotropy::Isotropic => models
                            .into_iter()
                            .filter_map(|model| {
                                let problem = AreaDistributionFittingProblemProxy::<U1> {
                                    measured: self.measured,
                                    model,
                                    marker: Default::default(),
                                };
                                let (result, report) = solver.minimize(problem);
                                match report.termination {
                                    TerminationReason::Converged { .. } => {
                                        Some((result.model, report))
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
                        Isotropy::Anisotropic => models
                            .into_iter()
                            .filter_map(|model| {
                                let problem = AreaDistributionFittingProblemProxy::<U2> {
                                    measured: self.measured,
                                    model,
                                    marker: Default::default(),
                                };
                                let (result, report) = solver.minimize(problem);
                                match report.termination {
                                    TerminationReason::Converged { .. } => {
                                        Some((result.model, report))
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
                    }
                };
                #[cfg(feature = "scaled-ndf-fitting")]
                let mut result: Vec<_> = {
                    let models =
                        initiate_microfacet_ndf_models(10, family, self.isotropy, self.scaled);
                    match (self.isotropy, self.scaled) {
                        (Isotropy::Isotropic, false) => models
                            .into_iter()
                            .filter_map(|model| {
                                let problem = AreaDistributionFittingProblemProxy::<U1, UNSCALED> {
                                    measured: self.measured,
                                    model,
                                    marker: Default::default(),
                                };
                                let (result, report) = solver.minimize(problem);
                                match report.termination {
                                    TerminationReason::Converged { .. } => {
                                        Some((result.model, report))
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
                        (Isotropy::Isotropic, true) => models
                            .into_iter()
                            .filter_map(|model| {
                                let problem = AreaDistributionFittingProblemProxy::<U1, SCALED> {
                                    measured: self.measured,
                                    model,
                                    marker: Default::default(),
                                };
                                let (result, report) = solver.minimize(problem);
                                match report.termination {
                                    TerminationReason::Converged { .. } => {
                                        Some((result.model, report))
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
                        (Isotropy::Anisotropic, false) => models
                            .into_iter()
                            .filter_map(|model| {
                                let problem = AreaDistributionFittingProblemProxy::<U2, UNSCALED> {
                                    measured: self.measured,
                                    model,
                                    marker: Default::default(),
                                };
                                let (result, report) = solver.minimize(problem);
                                match report.termination {
                                    TerminationReason::Converged { .. } => {
                                        Some((result.model, report))
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
                        (Isotropy::Anisotropic, true) => models
                            .into_iter()
                            .filter_map(|model| {
                                let problem = AreaDistributionFittingProblemProxy::<U2, SCALED> {
                                    measured: self.measured,
                                    model,
                                    marker: Default::default(),
                                };
                                let (result, report) = solver.minimize(problem);
                                match report.termination {
                                    TerminationReason::Converged { .. } => {
                                        Some((result.model, report))
                                    }
                                    _ => None,
                                }
                            })
                            .collect(),
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
///
/// The fitting problem always assumes that the macro surface normal is
/// [0, 1, 0].
struct AreaDistributionFittingProblemProxy<'a, N: Dim, const Scaled: bool> {
    measured: &'a MeasuredMndfData,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    marker: PhantomData<N>,
}

#[cfg(not(feature = "scaled-ndf-fitting"))]
/// The inner fitting problem for the microfacet area distribution function of
/// different effective parameters count. `N` is the number of effective
/// parameters of the model, this is introduced because the levenberg_marquardt
/// crate requires the parameter count to be a type parameter.
struct AreaDistributionFittingProblemProxy<'a, N: Dim> {
    measured: &'a MeasuredMndfData,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    marker: PhantomData<N>,
}

/// Calculates the residuals of the evaluated model against the whole measured
/// data.
fn calculate_isotropic_mndf_residuals(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let model = model.as_isotropic().unwrap();
    let residuals = measured.samples.iter().enumerate().map(|(idx, meas)| {
        let theta_idx = idx % theta_step_count;
        let theta = measured.params.zenith.step(theta_idx);
        model.eval_with_cos_theta_m(theta.cos() as f64) - *meas as f64
    });

    OMatrix::<f64, Dyn, U1>::from_iterator(residuals.len(), residuals)
}

fn calculate_anisotropic_mndf_residuals(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let model = model.as_anisotropic().unwrap();
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let residuals = measured.samples.iter().enumerate().map(|(idx, meas)| {
        let theta_idx = idx % theta_step_count;
        let theta = measured.params.zenith.step(theta_idx);
        let phi_idx = idx / theta_step_count;
        let phi = measured.params.azimuth.step(phi_idx);
        model.eval_with_cos_theta_phi_m(theta.cos() as f64, phi.cos() as f64) - *meas as f64
    });
    OMatrix::<f64, Dyn, U1>::from_iterator(residuals.len(), residuals)
}

#[cfg(feature = "scaled-ndf-fitting")]
/// Calculates the residuals of the scaled evaluated model against the measured
/// data.
fn calculate_scaled_isotropic_mndf_residuals(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let model = model.as_isotropic().unwrap();
    let scale = model.scale().unwrap();
    let residuals = measured.samples.iter().enumerate().map(|(idx, meas)| {
        let theta_idx = idx % theta_step_count;
        let theta = measured.params.zenith.step(theta_idx);
        model.eval_with_cos_theta_m(theta.cos() as f64) * scale - *meas as f64
    });

    OMatrix::<f64, Dyn, U1>::from_iterator(residuals.len(), residuals)
}

#[cfg(feature = "scaled-ndf-fitting")]
/// Calculates the residuals of the scaled evaluated model against the measured
/// data.
fn calculate_scaled_anisotropic_mndf_residuals(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let model = model.as_anisotropic().unwrap();
    let theta_step_count = measured.params.zenith.step_count_wrapped();
    let scale = model.scale().unwrap();

    let residuals = measured.samples.iter().enumerate().map(|(idx, meas)| {
        let theta_idx = idx % theta_step_count;
        let theta = measured.params.zenith.step(theta_idx);
        let phi_idx = idx / theta_step_count;
        let phi = measured.params.azimuth.step(phi_idx);
        model.eval_with_cos_theta_phi_m(theta.cos() as f64, phi.cos() as f64) * scale - *meas as f64
    });
    OMatrix::<f64, Dyn, U1>::from_iterator(residuals.len(), residuals)
}

/// Extracts the cosine of all the measured zenith and azimuth angles.
fn extract_theta_phi_ms(
    len: usize,
    azimuth: RangeByStepSizeInclusive<Radians>,
    zenith: RangeByStepSizeInclusive<Radians>,
) -> Vec<(f64, f64)> {
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
        .collect()
}

/// Extracts the cosine of all the measured zenith angles.
fn extract_theta_ms(len: usize, zenith: RangeByStepSizeInclusive<Radians>) -> Vec<f64> {
    let theta_step_count = zenith.step_count_wrapped();
    (0..len)
        .map(|idx| {
            let theta_idx = idx % theta_step_count;
            zenith.step(theta_idx).cos() as f64
        })
        .collect()
}

/// Calculates the partial derivative of the fitting parameter.
fn calculate_isotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U1> {
    let model = model.as_isotropic().unwrap();
    let cos_theta_ms = extract_theta_ms(measured.samples.len(), measured.params.zenith);
    OMatrix::<f64, Dyn, U1>::from_vec(model.calc_param_pd(&cos_theta_ms))
}

fn calculate_anisotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U2> {
    let model = model.as_anisotropic().unwrap();
    let cos_theta_phi_ms = extract_theta_phi_ms(
        measured.samples.len(),
        measured.params.azimuth,
        measured.params.zenith,
    );
    OMatrix::<f64, Dyn, U2>::from_row_slice(&model.calc_params_pd(&cos_theta_phi_ms))
}

#[cfg(feature = "scaled-ndf-fitting")]
fn calculate_scaled_isotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U2> {
    let model = model.as_isotropic().unwrap();
    let cos_theta_ms = extract_theta_ms(measured.samples.len(), measured.params.zenith);
    OMatrix::<f64, Dyn, U2>::from_row_slice(&model.calc_param_pd_scaled(&cos_theta_ms))
}

#[cfg(feature = "scaled-ndf-fitting")]
fn calculate_scaled_anisotropic_mndf_jacobian(
    measured: &MeasuredMndfData,
    model: &Box<dyn MicrofacetAreaDistributionModel>,
) -> OMatrix<f64, Dyn, U3> {
    let model = model.as_anisotropic().unwrap();
    let cos_theta_phi_ms = extract_theta_phi_ms(
        measured.samples.len(),
        measured.params.azimuth,
        measured.params.zenith,
    );
    OMatrix::<f64, Dyn, U3>::from_row_slice(&model.calc_params_pd_scaled(&cos_theta_phi_ms))
}

// When the feature `scaled-ndf-fitting` is not enabled, the fitting problem
// is simplified to a single parameter fitting problem.
#[cfg(not(feature = "scaled-ndf-fitting"))]
impl<'a> LeastSquaresProblem<f64, Dyn, U1> for AreaDistributionFittingProblemProxy<'a, U1> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) {
        self.model.as_isotropic_mut().unwrap().set_param(x[0])
    }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.as_isotropic().unwrap().param()) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_isotropic_mndf_residuals(
            self.measured,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(calculate_isotropic_mndf_jacobian(
            self.measured,
            &self.model,
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
        self.model
            .as_anisotropic_mut()
            .unwrap()
            .set_params([x[0], x[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(
            self.model.as_anisotropic().unwrap().params(),
        )
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_isotropic_mndf_residuals(
            self.measured,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(calculate_anisotropic_mndf_jacobian(
            self.measured,
            &self.model,
        ))
    }
}

use crate::{
    fitting::{
        BeckmannSpizzichinoAnisotropicNDF, FittingProblem, Isotropy, TrowbridgeReitzAnisotropicNDF,
        TrowbridgeReitzNDF,
    },
    measure::measurement::MeasurementData,
};
#[cfg(feature = "scaled-ndf-fitting")]
use nalgebra::U3;
use vgcore::units::Radians;

use super::FittingReport;

#[cfg(feature = "scaled-ndf-fitting")]
/// Isotropic MADF fitting problem without the scaling factor.
impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for AreaDistributionFittingProblemProxy<'a, U1, UNSCALED>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) {
        self.model.as_isotropic_mut().unwrap().set_param(x[0])
    }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.as_isotropic().unwrap().param()) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_isotropic_mndf_residuals(
            self.measured,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(calculate_isotropic_mndf_jacobian(
            self.measured,
            &self.model,
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
        let model = self.model.as_isotropic_mut().unwrap();
        model.set_param(x[0]);
        model.set_scale(x[1]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        let model = self.model.as_isotropic().unwrap();
        Vector::<f64, U2, Self::ParameterStorage>::new(model.param(), model.scale().unwrap())
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_scaled_isotropic_mndf_residuals(
            self.measured,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(calculate_scaled_isotropic_mndf_jacobian(
            self.measured,
            &self.model,
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
        self.model
            .as_anisotropic_mut()
            .unwrap()
            .set_params([x[0], x[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(
            self.model.as_anisotropic().unwrap().params(),
        )
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(calculate_anisotropic_mndf_residuals(
            self.measured,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        Some(calculate_anisotropic_mndf_jacobian(
            self.measured,
            &self.model,
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
        let model = self.model.as_anisotropic_mut().unwrap();
        model.set_params([x[0], x[1]]);
        model.set_scale(x[2]);
    }

    fn params(&self) -> Vector<f64, U3, Self::ParameterStorage> {
        let model = self.model.as_anisotropic().unwrap();
        Vector::<f64, U3, Self::ParameterStorage>::new(
            model.params()[0],
            model.params()[1],
            model.scale().unwrap(),
        )
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(calculate_scaled_anisotropic_mndf_residuals(
            self.measured,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U3, Self::JacobianStorage>> {
        Some(calculate_scaled_anisotropic_mndf_jacobian(
            self.measured,
            &self.model,
        ))
    }
}

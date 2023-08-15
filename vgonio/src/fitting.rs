pub mod beckmann_spizzichino;
pub mod trowbridge_reitz;

pub use beckmann_spizzichino::*;
pub use trowbridge_reitz::*;

use crate::measure::microfacet::{MeasuredMadfData, MeasuredMmsfData};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{Dim, Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, Vector1, Vector2, U1, U2};
use std::{
    any::Any,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
};
use vgcore::{
    math,
    math::{madd, DVec3, Handedness, SphericalCoord, Vec3},
};

/// Family of reflection models.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReflectionModelFamily {
    /// Microfacet based reflection model.
    Microfacet(MicrofacetModelFamily),
}

/// Variation of microfacet based reflection model.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MicrofacetModelFamily {
    /// Trowbridge-Reitz (a.k.a. GGX) reflection model.
    TrowbridgeReitz,
    /// Beckmann-Spizzichino reflection model.
    BeckmannSpizzichino,
}

/// A model after fitting.
#[derive(Debug, Clone)]
pub enum FittedModel {
    Bsdf(Box<dyn BsdfModel>),
    Madf(Box<dyn MicrofacetAreaDistributionModel>),
    Mmsf(Box<dyn MicrofacetMaskingShadowingModel>),
}

impl FittedModel {
    /// Returns the fitted microfacet BSDF model.
    pub fn bsdf_model(&self) -> Option<Box<dyn BsdfModel>> {
        match self {
            FittedModel::Bsdf(bsdf) => Some(bsdf.clone()),
            _ => None,
        }
    }

    /// Returns the fitted microfacet area distribution function (NDF).
    pub fn madf_model(&self) -> Option<Box<dyn MicrofacetAreaDistributionModel>> {
        match self {
            FittedModel::Madf(madf) => Some(madf.clone()),
            _ => None,
        }
    }

    /// Returns the fitted microfacet masking-shadowing function (MMSF).
    pub fn mmsf_model(&self) -> Option<Box<dyn MicrofacetMaskingShadowingModel>> {
        match self {
            FittedModel::Mmsf(mmsf) => Some(mmsf.clone()),
            _ => None,
        }
    }

    /// Returns the family of the fitted model.
    pub fn family(&self) -> ReflectionModelFamily {
        match self {
            FittedModel::Bsdf(m) => m.family(),
            FittedModel::Madf(m) => m.family(),
            FittedModel::Mmsf(m) => m.family(),
        }
    }

    #[cfg(feature = "adf-fitting-scaling")]
    pub fn is_scaled(&self) -> bool {
        match self {
            FittedModel::Madf(m) => m.scale().is_some(),
            _ => false,
        }
    }
}

/// A microfacet area distribution function model.
pub trait MicrofacetAreaDistributionModel: Debug {
    /// Returns the name of the model.
    fn name(&self) -> &'static str;

    /// Returns the concrete model of the MADF.
    fn family(&self) -> ReflectionModelFamily;

    /// Returns if the model is isotropic.
    fn is_isotropic(&self) -> bool;

    /// Returns the parameters of the model.
    ///
    /// For isotropic models, elements in the returned array are the same.
    /// For anisotropic models, the returned array contains the parameters
    /// in the following order: [α_u, α_v].
    fn params(&self) -> [f64; 2];

    /// Sets the parameters of the model.
    fn set_params(&mut self, params: [f64; 2]);

    #[cfg(feature = "adf-fitting-scaling")]
    fn scale(&self) -> Option<f64>;

    #[cfg(feature = "adf-fitting-scaling")]
    fn set_scale(&mut self, scale_param: f64);

    /// Returns the number of effective parameters of the model.
    fn effective_params_count(&self) -> usize {
        if self.is_isotropic() {
            1
        } else {
            2
        }
    }

    /// Evaluates the model with the given microfacet normal and
    /// the normal vector of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `m` - The microfacet normal vector.
    /// * `n` - The normal vector of the macro-surface.
    fn eval(&self, m: DVec3, n: DVec3) -> f64 {
        self.eval_with_cos_theta_m(m.normalize().dot(n.normalize()))
    }

    /// Evaluates the model with the given angle between the microfacet normal
    /// and the normal vector of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `theta_m` - The angle between the microfacet normal and the normal
    ///   vector of the macro-surface.
    fn eval_with_theta_m(&self, theta_m: f64) -> f64 { self.eval_with_cos_theta_m(theta_m.cos()) }

    /// Evaluates the model with the given cosine of the angle between the
    /// microfacet normal and the normal vector of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `cos_theta_m` - The cosine of the angle between the microfacet normal
    ///  and the normal vector of the macro-surface.
    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64;

    /// Clones the model.
    fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel>;
}

impl Clone for Box<dyn MicrofacetAreaDistributionModel> {
    fn clone(&self) -> Self { self.clone_box() }
}

/// A microfacet masking-shadowing function model.
pub trait MicrofacetMaskingShadowingModel: Debug {
    /// Returns the name of the model.
    fn name(&self) -> &'static str;

    /// Returns the base model of the MMSF.
    fn family(&self) -> ReflectionModelFamily;

    /// Returns the parameter of the model.
    fn param(&self) -> f64;

    /// Sets the parameter of the model.
    fn set_param(&mut self, param: f64);

    /// Evaluates the model.
    ///
    /// # Arguments
    ///
    /// * `m` - The microfacet normal vector.
    /// * `n` - The normal vector of the macro-surface.
    /// * `v` - The view vector (any direction, either the incident or outgoing
    ///   direction).
    fn eval(&self, m: DVec3, n: DVec3, v: DVec3) -> f64 {
        let n = n.normalize();
        let cos_theta_v = v.normalize().dot(n);
        let cos_theta_m = m.normalize().dot(n);
        if cos_theta_m / cos_theta_v <= 0.0 {
            0.0
        } else {
            self.eval_with_cos_theta_v(cos_theta_v)
        }
    }

    /// Evaluates the model with the given cosine of the angle between the
    /// microfacet normal and the normal vector of the macro-surface, and
    /// the cosine of the angle between the view vector and the normal vector
    /// of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `cos_theta_m` - The cosine of the angle between the microfacet normal
    ///  and the normal vector of the macro-surface.
    /// * `cos_theta_v` - The cosine of the angle between the view vector and
    ///  the normal vector of the macro-surface.
    fn eval_with_cos_theta_m_v(&self, cos_theta_m: f64, cos_theta_v: f64) -> f64 {
        debug_assert!(
            cos_theta_m > 0.0 && cos_theta_m <= 1.0 && cos_theta_v > 0.0 && cos_theta_v <= 1.0
        );
        self.eval_with_cos_theta_v(cos_theta_v)
    }

    /// Evaluates the model with the given cosine of the angle between the
    /// microfacet normal and the normal vector of the macro-surface. This
    /// function assumes that the cosine of the angle between the microfacet
    /// normal and the macro-surface normal is positive.
    fn eval_with_cos_theta_v(&self, cos_theta_v: f64) -> f64;

    /// Clones the model.
    fn clone_box(&self) -> Box<dyn MicrofacetMaskingShadowingModel>;
}

impl Clone for Box<dyn MicrofacetMaskingShadowingModel> {
    fn clone(&self) -> Self { self.clone_box() }
}

/// A microfacet bidirectional scattering distribution function model.
pub trait BsdfModel: Debug {
    fn family(&self) -> ReflectionModelFamily;

    /// Clones the model.
    fn clone_box(&self) -> Box<dyn BsdfModel>;
}

impl Clone for Box<dyn BsdfModel> {
    fn clone(&self) -> Self { self.clone_box() }
}

pub trait FittingProblem {
    type Model;

    /// Non linear least squares fitting using Levenberg-Marquardt algorithm.
    fn lsq_lm_fit(self) -> (Self::Model, MinimizationReport<f64>);
}

/// Fitting procedure trying to find the width parameter for the microfacet area
/// (normal) distribution function (NDF) of the Trowbridge-Reitz (GGX) model.
///
/// Use Levenberg-Marquardt to fit this parameter to measured data.
/// The problem has 1 parameter and 1 residual.
pub struct MadfFittingProblem<'a> {
    /// The measured data to fit to.
    measured: &'a MeasuredMadfData,
    /// The normal vector of the microfacet.
    normal: Vec3,
    /// The target model.
    model: Box<dyn MicrofacetAreaDistributionModel>,
}

impl<'a> Display for MadfFittingProblem<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MadfFittingProblem")
            .field("target_model", &self.model.family())
            .finish()
    }
}

impl<'a> MadfFittingProblem<'a> {
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
    ) -> Self {
        Self {
            measured,
            normal: n,
            model: Box::new(model),
        }
    }
}

#[cfg(feature = "adf-fitting-scaling")]
/// The inner fitting problem for the microfacet area distribution function of
/// different effective parameters count.
struct InnerMadfFittingProblem<'a, N: Dim, const Scaled: bool> {
    measured: &'a MeasuredMadfData,
    normal: DVec3,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    marker: PhantomData<N>,
}

#[cfg(not(feature = "adf-fitting-scaling"))]
struct InnerMadfFittingProblem<'a, N: Dim> {
    measured: &'a MeasuredMadfData,
    normal: DVec3,
    model: Box<dyn MicrofacetAreaDistributionModel>,
    marker: PhantomData<N>,
}

/// Calculates the residuals of the evaluated model against the measured data.
fn calculate_single_param_madf_residuals(
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

#[cfg(not(feature = "adf-fitting-scaling"))]
impl<'a> LeastSquaresProblem<f64, Dyn, U1> for InnerMadfFittingProblem<'a, U1> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) { self.model.set_params([x[0], x[0]]) }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.params()[0]) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_single_param_madf_residuals(
            self.measured,
            self.normal,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(calculate_single_param_madf_jacobian(
            self.measured,
            self.params().x,
            self.normal,
            &self.model,
        ))
    }
}

#[cfg(feature = "adf-fitting-scaling")]
impl<'a> LeastSquaresProblem<f64, Dyn, U1> for InnerMadfFittingProblem<'a, U1, false> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f64>) { self.model.set_params([x[0], x[0]]) }

    fn params(&self) -> Vector1<f64> { Vector1::new(self.model.params()[0]) }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(calculate_single_param_madf_residuals(
            self.measured,
            self.normal,
            &self.model,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        Some(calculate_single_param_madf_jacobian(
            self.measured,
            self.params().x,
            self.normal,
            &self.model,
        ))
    }
}

#[cfg(feature = "adf-fitting-scaling")]
/// Least squares problem for fitting a MADF model together with a scaling
/// factor. The scaling factor is applied to the MADF model's output.
impl<'a> LeastSquaresProblem<f64, Dyn, U2> for InnerMadfFittingProblem<'a, U1, true> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector2<f64>) {
        self.model.set_params([x[0], x[0]]);
        self.model.set_scale(x[1]);
    }

    fn params(&self) -> Vector2<f64> {
        Vector2::new(self.model.params()[0], self.model.scale().unwrap())
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        let scale = self.model.scale().unwrap();
        let residuals = self
            .measured
            .samples
            .iter()
            .enumerate()
            .map(|(idx, meas)| {
                let phi_idx = idx / theta_step_count;
                let theta_idx = idx % theta_step_count;
                let theta = self.measured.params.zenith.step(theta_idx);
                let phi = self.measured.params.azimuth.step(phi_idx);
                let m = SphericalCoord::new(1.0, theta, phi)
                    .to_cartesian(Handedness::RightHandedYUp)
                    .as_dvec3();
                self.model.eval(m, self.normal) * scale - *meas as f64
            })
            .collect();
        Some(Matrix::<f64, Dyn, U1, Self::ResidualStorage>::from_vec(
            residuals,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        let alpha = self.params().x;
        let alpha2 = alpha * alpha;
        let scale = self.model.scale().unwrap();
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        let derivatives: Vec<f64> = match self.model.family() {
            ReflectionModelFamily::Microfacet(m) => match m {
                MicrofacetModelFamily::TrowbridgeReitz => self
                    .measured
                    .samples
                    .iter()
                    .enumerate()
                    .flat_map(move |(idx, meas)| {
                        let phi_idx = idx / theta_step_count;
                        let theta_idx = idx % theta_step_count;
                        let theta = self.measured.params.zenith.step(theta_idx);
                        let phi = self.measured.params.azimuth.step(phi_idx);
                        let m = SphericalCoord::new(1.0, theta, phi)
                            .to_cartesian(Handedness::RightHandedYUp)
                            .as_dvec3();
                        let cos_theta_m = m.dot(self.normal);
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
                            let denominator =
                                std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(3);
                            numerator / denominator
                        };
                        let d_scale = {
                            let numerator = alpha2 * sec_theta_m4;
                            let denominator =
                                std::f64::consts::PI * (alpha2 + tan_theta_m2).powi(2);
                            numerator / denominator
                        };
                        [d_alpha, d_scale]
                    })
                    .collect(),
                MicrofacetModelFamily::BeckmannSpizzichino => self
                    .measured
                    .samples
                    .iter()
                    .enumerate()
                    .flat_map(move |(idx, meas)| {
                        let phi_idx = idx / self.measured.params.zenith.step_count_wrapped();
                        let theta_idx = idx % self.measured.params.zenith.step_count_wrapped();
                        let theta = self.measured.params.zenith.step(theta_idx);
                        let phi = self.measured.params.azimuth.step(phi_idx);
                        let m = SphericalCoord::new(1.0, theta, phi)
                            .to_cartesian(Handedness::RightHandedYUp)
                            .as_dvec3();
                        let cos_theta_m = m.dot(self.normal);
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
        Some(Matrix::<f64, Dyn, U2, Self::JacobianStorage>::from_row_slice(&derivatives))
    }
}

impl<'a> FittingProblem for MadfFittingProblem<'a> {
    type Model = Box<dyn MicrofacetAreaDistributionModel>;

    fn lsq_lm_fit(self) -> (Self::Model, MinimizationReport<f64>) {
        let effective_params_count = self.model.effective_params_count();
        if effective_params_count == 1 {
            #[cfg(not(feature = "adf-fitting-scaling"))]
            {
                let problem = InnerMadfFittingProblem::<U1> {
                    measured: self.measured,
                    normal: self.normal.as_dvec3(),
                    model: self.model.clone_box(),
                    marker: Default::default(),
                };
                let solver = LevenbergMarquardt::new();
                let (result, report) = solver.minimize(problem);
                (result.model, report)
            }
            #[cfg(feature = "adf-fitting-scaling")]
            {
                if self.model.scale().is_some() {
                    let problem = InnerMadfFittingProblem::<U1, true> {
                        measured: self.measured,
                        normal: self.normal.as_dvec3(),
                        model: self.model.clone_box(),
                        marker: Default::default(),
                    };
                    let solver = LevenbergMarquardt::new();
                    let (result, report) = solver.minimize(problem);
                    (result.model, report)
                } else {
                    let problem = InnerMadfFittingProblem::<U1, false> {
                        measured: self.measured,
                        normal: self.normal.as_dvec3(),
                        model: self.model.clone_box(),
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

pub struct MmsfFittingProblem<'a> {
    inner: InnerMmsfFittingProblem<'a>,
}

impl<'a> MmsfFittingProblem<'a> {
    pub fn new<M: MicrofacetMaskingShadowingModel + 'static>(
        measured: &'a MeasuredMmsfData,
        model: M,
        normal: Vec3,
    ) -> Self {
        Self {
            inner: InnerMmsfFittingProblem {
                measured,
                normal,
                model: Box::new(model),
            },
        }
    }
}

struct InnerMmsfFittingProblem<'a> {
    measured: &'a MeasuredMmsfData,
    normal: Vec3,
    model: Box<dyn MicrofacetMaskingShadowingModel>,
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1> for InnerMmsfFittingProblem<'a> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector<f64, U1, Self::ParameterStorage>) {
        self.model.set_param(x[0]);
    }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> {
        Vector::<f64, U1, Self::ParameterStorage>::from_vec(vec![self.model.param()])
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        let phi_step_count = self.measured.params.azimuth.step_count_wrapped();
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        // Only the first phi_step_count * theta_step_count samples are used
        let residuals = self
            .measured
            .samples
            .iter()
            .take(phi_step_count * theta_step_count)
            .enumerate()
            .map(|(idx, meas)| {
                let phi_v_idx = idx / theta_step_count;
                let theta_v_idx = idx % theta_step_count;
                let theta_v = self.measured.params.zenith.step(theta_v_idx);
                let cos_theta_v = theta_v.cos() as f64;
                self.model.eval_with_cos_theta_v(cos_theta_v) - *meas as f64
            })
            .collect();
        Some(Matrix::<f64, Dyn, U1, Self::ResidualStorage>::from_vec(
            residuals,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        let alpha = self.params().x;
        let alpha2 = alpha * alpha;
        let phi_step_count = self.measured.params.azimuth.step_count_wrapped();
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        let derivatives = match self.model.family() {
            ReflectionModelFamily::Microfacet(m) => match m {
                MicrofacetModelFamily::TrowbridgeReitz => self
                    .measured
                    .samples
                    .iter()
                    .take(phi_step_count * theta_step_count)
                    .enumerate()
                    .map(|(idx, meas)| {
                        let phi_v_idx = idx / theta_step_count;
                        let theta_v_idx = idx % theta_step_count;
                        let theta_v = self.measured.params.zenith.step(theta_v_idx);
                        let phi_v = self.measured.params.azimuth.step(phi_v_idx);
                        let v = SphericalCoord::new(1.0, theta_v, phi_v)
                            .to_cartesian(Handedness::RightHandedYUp);
                        let cos_theta_v = v.dot(self.normal) as f64;
                        let cos_theta_v2 = cos_theta_v * cos_theta_v;
                        let tan_theta_v2 = if cos_theta_v2 == 0.0 {
                            f64::INFINITY
                        } else {
                            1.0 / cos_theta_v2 - cos_theta_v2
                        };
                        let a = (alpha2 * tan_theta_v2 + 1.0).sqrt();
                        let numerator = 2.0 * alpha * tan_theta_v2;
                        let denominator = a * a * a + 2.0 * a * a + a;
                        numerator / denominator
                    })
                    .collect(),
                MicrofacetModelFamily::BeckmannSpizzichino => self
                    .measured
                    .samples
                    .iter()
                    .take(phi_step_count * theta_step_count)
                    .enumerate()
                    .map(|(idx, meas)| {
                        let phi_v_idx = idx / theta_step_count;
                        let theta_v_idx = idx % theta_step_count;
                        let theta_v = self.measured.params.zenith.step(theta_v_idx);
                        let phi_v = self.measured.params.azimuth.step(phi_v_idx);
                        let v = SphericalCoord::new(1.0, theta_v, phi_v)
                            .to_cartesian(Handedness::RightHandedYUp);
                        let cos_theta_v = v.dot(self.normal) as f64;
                        let cos_theta_v2 = cos_theta_v * cos_theta_v;
                        let tan_theta_v = if cos_theta_v2 == 0.0 {
                            f64::INFINITY
                        } else {
                            (1.0 - cos_theta_v2).sqrt() / cos_theta_v
                        };

                        let cot_theta_v = 1.0 / tan_theta_v;
                        let a = cot_theta_v / alpha;
                        let e = (-a * a).exp();
                        let erf = libm::erf(a);
                        let sqrt_pi = std::f64::consts::PI.sqrt();
                        let b = 1.0 + erf + e * alpha * tan_theta_v / sqrt_pi;
                        let numerator = 2.0 * e * tan_theta_v;
                        let denominator = sqrt_pi * b * b;
                        numerator / denominator
                    })
                    .collect(),
            },
        };
        Some(Matrix::<f64, Dyn, U1, Self::JacobianStorage>::from_vec(
            derivatives,
        ))
    }
}

impl<'a> FittingProblem for MmsfFittingProblem<'a> {
    type Model = Box<dyn MicrofacetMaskingShadowingModel>;

    fn lsq_lm_fit(self) -> (Self::Model, MinimizationReport<f64>) {
        let solver = LevenbergMarquardt::new();
        let (result, report) = solver.minimize(self.inner);
        (result.model, report)
    }
}

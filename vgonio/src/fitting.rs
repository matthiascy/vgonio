pub mod beckmann_spizzichino;
mod mndf;
pub mod trowbridge_reitz;

pub use beckmann_spizzichino::*;
pub use mndf::*;
pub use trowbridge_reitz::*;

use crate::measure::microfacet::{MeasuredMmsfData, MeasuredMndfData};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{
    Dim, Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, Vector1, Vector2, U1, U2, U3,
};
use std::{
    any::Any,
    fmt::{Debug, Display},
};
use vgcore::math::{DVec3, Handedness, SphericalCoord, Vec3};

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
    Mndf {
        model: Box<dyn MicrofacetAreaDistributionModel>,
        mode: AreaDistributionFittingMode,
    },
    Mmsf(Box<dyn MicrofacetMaskingShadowingModel>),
}

impl FittedModel {
    /// Returns the family of the fitted model.
    pub fn family(&self) -> ReflectionModelFamily {
        match self {
            FittedModel::Bsdf(m) => m.family(),
            FittedModel::Mndf { model, .. } => model.family(),
            FittedModel::Mmsf(m) => m.family(),
        }
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    pub fn is_scaled(&self) -> bool {
        match self {
            FittedModel::Mndf { model, .. } => model.scale().is_some(),
            _ => false,
        }
    }

    pub fn is_isotropic(&self) -> bool {
        match self {
            FittedModel::Bsdf(_) | FittedModel::Mmsf(_) => true,
            FittedModel::Mndf { model, .. } => model.is_isotropic(),
        }
    }

    pub fn mode(&self) -> Option<AreaDistributionFittingMode> {
        match self {
            FittedModel::Bsdf(_) | FittedModel::Mmsf(_) => None,
            FittedModel::Mndf { mode, .. } => Some(*mode),
        }
    }
}

/// A collection of fitted models without repetition.
#[derive(Debug, Clone)]
pub struct FittedModels(Vec<FittedModel>);

impl FittedModels {
    pub fn new() -> Self { Self(Vec::new()) }

    #[cfg(not(feature = "scaled-ndf-fitting"))]
    pub fn contains(
        &self,
        family: ReflectionModelFamily,
        mode: AreaDistributionFittingMode,
        isotropic: bool,
    ) -> bool {
        self.0.iter().any(|f| {
            f.family() == family && f.mode() == Some(mode) && f.is_isotropic() == isotropic
        })
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    pub fn contains(
        &self,
        family: ReflectionModelFamily,
        mode: AreaDistributionFittingMode,
        isotropic: bool,
        scaled: bool,
    ) -> bool {
        self.0.iter().any(|f| {
            f.family() == family
                && f.is_scaled() == scaled
                && f.mode() == Some(mode)
                && f.is_isotropic() == isotropic
        })
    }

    pub fn push(&mut self, model: FittedModel) { self.0.push(model); }
}

impl AsRef<[FittedModel]> for FittedModels {
    fn as_ref(&self) -> &[FittedModel] { self.0.as_ref() }
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
    /// in the following order: [α_x, α_y].
    fn params(&self) -> [f64; 2];

    /// Sets the parameters of the model.
    fn set_params(&mut self, params: [f64; 2]);

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Returns the scaling factor of the model.
    ///
    /// If the model is not scaled, this function returns `None`. Otherwise, it
    /// returns the scaling factor.
    fn scale(&self) -> Option<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Sets the scaling factor of the model.
    fn set_scale(&mut self, scale: f64);

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

    /// Calculates the partial derivatives of the model with respect to
    /// the single alpha parameter.
    fn calc_param_pd_isotropic(&self, cos_theta_ms: &[f64]) -> Vec<f64>;

    /// Calculates the partial derivatives of the model with respect to
    /// the two alpha parameters. The returned array contains the partial
    /// derivatives in the following order: [∂/∂α_x, ∂/∂α_y] for each cosine
    /// of the angle between the microfacet normal and the normal vector of
    /// the macro-surface.
    fn calc_param_pd_anisotropic(&self, cos_theta_ms: &[f64]) -> Vec<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Calculates the partial derivatives of the model with respect to
    /// the single alpha parameter and the scaling factor.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the model with respect to the single alpha
    /// parameter and the scaling factor. The returned array contains the
    /// partial derivatives in the following order: [∂/∂α, ∂/∂scale] for each
    /// cosine of the angle between the microfacet normal and the normal vector
    /// of the macro-surface.
    fn calc_param_pd_isotropic_scaled(&self, cos_theta_ms: &[f64]) -> Vec<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Calculates the partial derivatives of the model with respect to
    /// the two alpha parameters and the scaling factor. The returned array
    /// contains the partial derivatives in the following order:
    /// [∂/∂α_x, ∂/∂α_y, ∂/∂scale] for each cosine of the angle between the
    /// microfacet normal and the normal vector of the macro-surface.
    fn calc_param_pd_anisotropic_scaled(&self, cos_theta_ms: &[f64]) -> Vec<f64>;

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

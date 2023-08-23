mod beckmann_spizzichino;
mod mmsf;
mod mndf;
mod trowbridge_reitz;

pub use beckmann_spizzichino::*;
pub use mmsf::*;
pub use mndf::*;
pub use trowbridge_reitz::*;

use levenberg_marquardt::MinimizationReport;
use std::fmt::Debug;
use vgcore::math::DVec3;

/// Indicates if something is isotropic or anisotropic.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Isotropy {
    /// Uniformity in all directions.
    Isotropic,
    /// Non-uniformity in some directions.
    Anisotropic,
}

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
    Mndf(Box<dyn MicrofacetAreaDistributionModel>),
    Mmsf(Box<dyn MicrofacetMaskingShadowingModel>),
}

impl FittedModel {
    /// Returns the family of the fitted model.
    pub fn family(&self) -> ReflectionModelFamily {
        match self {
            FittedModel::Bsdf(m) => m.family(),
            FittedModel::Mndf(m) => m.family(),
            FittedModel::Mmsf(m) => m.family(),
        }
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    pub fn is_scaled(&self) -> bool {
        match self {
            FittedModel::Mndf(m) => m.scale().is_some(),
            _ => false,
        }
    }

    pub fn isotropy(&self) -> Isotropy {
        match self {
            FittedModel::Bsdf(_) | FittedModel::Mmsf(_) => Isotropy::Isotropic,
            FittedModel::Mndf(m) => m.isotropy(),
        }
    }
}

/// A collection of fitted models without repetition.
#[derive(Debug, Clone)]
pub struct FittedModels(Vec<FittedModel>);

impl FittedModels {
    pub fn new() -> Self { Self(Vec::new()) }

    #[cfg(not(feature = "scaled-ndf-fitting"))]
    pub fn contains(&self, family: ReflectionModelFamily, isotropy: Isotropy) -> bool {
        self.0
            .iter()
            .any(|f| f.family() == family && f.isotropy() == isotropy)
    }

    #[cfg(feature = "scaled-ndf-fitting")]
    pub fn contains(
        &self,
        family: ReflectionModelFamily,
        isotropy: Isotropy,
        scaled: bool,
    ) -> bool {
        self.0
            .iter()
            .any(|f| f.family() == family && f.is_scaled() == scaled && f.isotropy() == isotropy)
    }

    pub fn push(&mut self, model: FittedModel) { self.0.push(model); }
}

impl AsRef<[FittedModel]> for FittedModels {
    fn as_ref(&self) -> &[FittedModel] { self.0.as_ref() }
}

#[cfg(feature = "scaled-ndf-fitting")]
macro impl_get_set_scale($self:ident) {
    fn scale(&$self) -> Option<f64> { $self.scale }

    fn set_scale(&mut $self, scale: f64) {
        #[cfg(debug_assertions)]
        if $self.scale.is_none() {
            panic!("Trying to set the scale on a non-scaled Beckmann Spizzichino NDF");
        }
        $self.scale.replace(scale);
    }
}

/// Isotropic microfacet area distribution function model.
pub trait IsotropicMicrofacetAreaDistributionModel: Debug {
    /// Returns the name of the model.
    fn name(&self) -> &'static str;

    /// Returns the family of the reflection model.
    fn family(&self) -> ReflectionModelFamily;

    /// Returns the value of the parameter.
    fn param(&self) -> f64;

    /// Sets the value of the parameter.
    fn set_param(&mut self, param: f64);

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Returns the scaling factor of the model.
    ///
    /// If the model is not scaled, this function returns `None`. Otherwise, it
    /// returns the scaling factor.
    fn scale(&self) -> Option<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Sets the scaling factor of the model.
    fn set_scale(&mut self, scale: f64);

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

    /// Returns the partial derivatives of the model with respect to the
    /// parameter.
    fn calc_param_pd(&self, cos_theta_ms: &[f64]) -> Vec<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Returns the partial derivatives of the model with respect to the
    /// parameter and the scaling factor.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the model with respect to the parameter and
    /// the scaling factor. The returned array contains the partial derivatives
    /// in the following order: [∂/∂α, ∂/∂scale] for each cosine of the angle
    /// between the microfacet normal and the normal vector of the
    /// macro-surface.
    fn calc_param_pd_scaled(&self, cos_theta_ms: &[f64]) -> Vec<f64>;

    /// Clones the model.
    fn clone_box(&self) -> Box<dyn IsotropicMicrofacetAreaDistributionModel>;
}

/// Anisotropic microfacet area distribution function model.
pub trait AnisotropicMicrofacetAreaDistributionModel: Debug {
    /// Returns the name of the model.
    fn name(&self) -> &'static str;

    /// Returns the family of the reflection model.
    fn family(&self) -> ReflectionModelFamily;

    /// Returns the value of the parameter.
    fn params(&self) -> [f64; 2];

    /// Sets the value of the parameter.
    fn set_params(&mut self, param: [f64; 2]);

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Returns the scaling factor of the model.
    ///
    /// If the model is not scaled, this function returns `None`. Otherwise, it
    /// returns the scaling factor.
    fn scale(&self) -> Option<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Sets the scaling factor of the model.
    fn set_scale(&mut self, scale: f64);

    /// Evaluates the model with the given microfacet normal and
    /// the normal vector of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `m` - The microfacet normal vector. Should be normalized.
    /// * `n` - The normal vector of the macro-surface. Should be normalized.
    fn eval(&self, m: DVec3, n: DVec3) -> f64 {
        debug_assert!(m.is_normalized() && n.is_normalized());
        let cos_theta_m = m.dot(n);
        let cos_phi_m = (m.z.atan2(m.x) - n.z.atan2(n.x)).cos();
        self.eval_with_cos_theta_phi_m(cos_theta_m, cos_phi_m)
    }

    /// Evaluates the model with the given angle between the microfacet normal
    /// and the normal vector of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `theta_m` - The polar angle difference between the microfacet normal
    ///   and the normal vector of the macro-surface.
    /// * `phi_m` - The azimuthal angle difference between the microfacet normal
    ///  and the normal vector of the macro-surface.
    fn eval_with_theta_phi_m(&self, theta_m: f64, phi_m: f64) -> f64 {
        self.eval_with_cos_theta_phi_m(theta_m.cos(), phi_m.cos())
    }

    /// Evaluates the model with the given cosine of the angle between the
    /// microfacet normal and the normal vector of the macro-surface.
    ///
    /// # Arguments
    ///
    /// * `cos_theta_m` - The cosine of the polar angle difference between the
    ///   microfacet normal and the normal vector of the macro-surface.
    /// * `cos_phi_m` - The cosine of the azimuthal angle difference between the
    ///   microfacet normal and the normal vector of the macro-surface.
    fn eval_with_cos_theta_phi_m(&self, cos_theta_m: f64, cos_phi_m: f64) -> f64;

    /// Returns the partial derivatives of the model with respect to the
    /// parameter.
    ///
    /// # Arguments
    ///
    /// * `cos_theta_phi_ms` - The cosines of angle difference between the
    ///  microfacet normal and the normal vector of the macro-surface.
    ///  The array contains the cosines in the following order: [(cos θ, cos φ)]
    ///  for each cosine of the angle between the microfacet.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the model with respect to the parameter.
    /// The returned array contains the partial derivatives in the following
    /// order: [∂/∂αx, ∂/∂αy] for each cosine of the angle between the
    /// microfacet
    fn calc_params_pd(&self, cos_theta_phi_ms: &[(f64, f64)]) -> Vec<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    /// Returns the partial derivatives of the model with respect to the
    /// parameter and the scaling factor.
    ///
    /// # Arguments
    ///
    /// * `cos_theta_phi_ms` - The cosines of angle difference between the
    /// microfacet normal and the normal vector of the macro-surface.
    /// The array contains the cosines in the following order: [(cos θ, cos φ)]
    /// for each cosine of the angle between the microfacet.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the model with respect to the parameter and
    /// the scaling factor. The returned array contains the partial derivatives
    /// in the following order: [∂/∂αx, ∂/∂αy, ∂/∂scale].
    fn calc_params_pd_scaled(&self, cos_theta_phi_ms: &[(f64, f64)]) -> Vec<f64>;

    /// Clones the model.
    fn clone_box(&self) -> Box<dyn AnisotropicMicrofacetAreaDistributionModel>;
}

/// A microfacet area distribution function model.
pub trait MicrofacetAreaDistributionModel: Debug {
    /// Returns the name of the model.
    fn name(&self) -> &'static str;

    /// Returns the concrete model of the MADF.
    fn family(&self) -> ReflectionModelFamily;

    /// Returns if the model is isotropic.
    fn is_isotropic(&self) -> bool;

    /// Returns the isotropy of the model.
    fn isotropy(&self) -> Isotropy {
        if self.is_isotropic() {
            Isotropy::Isotropic
        } else {
            Isotropy::Anisotropic
        }
    }

    /// Clones the model.
    fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel>;

    /// Returns as the isotropic model if the model is isotropic.
    fn as_isotropic(&self) -> Option<&dyn IsotropicMicrofacetAreaDistributionModel>;

    /// Returns as the isotropic model if the model is isotropic.
    fn as_isotropic_mut(&mut self) -> Option<&mut dyn IsotropicMicrofacetAreaDistributionModel>;

    /// Returns as the anisotropic model if the model is anisotropic.
    fn as_anisotropic(&self) -> Option<&dyn AnisotropicMicrofacetAreaDistributionModel>;

    /// Returns as the anisotropic model if the model is anisotropic.
    fn as_anisotropic_mut(&mut self)
        -> Option<&mut dyn AnisotropicMicrofacetAreaDistributionModel>;

    #[cfg(feature = "scaled-ndf-fitting")]
    fn scale(&self) -> Option<f64>;

    #[cfg(feature = "scaled-ndf-fitting")]
    fn set_scale(&mut self, scale: f64);
}

impl Clone for Box<dyn MicrofacetAreaDistributionModel> {
    fn clone(&self) -> Self { self.clone_box() }
}

macro impl_microfacet_area_distribution_model_for_isotropic_model($t:ty) {
    impl MicrofacetAreaDistributionModel for $t {
        fn name(&self) -> &'static str {
            <Self as IsotropicMicrofacetAreaDistributionModel>::name(self)
        }

        fn family(&self) -> ReflectionModelFamily {
            <Self as IsotropicMicrofacetAreaDistributionModel>::family(self)
        }

        fn is_isotropic(&self) -> bool { true }

        fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel> { Box::new(*self) }

        fn as_isotropic(&self) -> Option<&dyn IsotropicMicrofacetAreaDistributionModel> {
            Some(self)
        }

        fn as_isotropic_mut(
            &mut self,
        ) -> Option<&mut dyn IsotropicMicrofacetAreaDistributionModel> {
            Some(self)
        }

        fn as_anisotropic(&self) -> Option<&dyn AnisotropicMicrofacetAreaDistributionModel> { None }

        fn as_anisotropic_mut(
            &mut self,
        ) -> Option<&mut dyn AnisotropicMicrofacetAreaDistributionModel> {
            None
        }

        #[cfg(feature = "scaled-ndf-fitting")]
        fn scale(&self) -> Option<f64> {
            <Self as IsotropicMicrofacetAreaDistributionModel>::scale(self)
        }

        #[cfg(feature = "scaled-ndf-fitting")]
        fn set_scale(&mut self, scale: f64) {
            <Self as IsotropicMicrofacetAreaDistributionModel>::set_scale(self, scale);
        }
    }
}

macro impl_microfacet_area_distribution_model_for_anisotropic_model($t:ty) {
    impl MicrofacetAreaDistributionModel for $t {
        fn name(&self) -> &'static str {
            <Self as AnisotropicMicrofacetAreaDistributionModel>::name(self)
        }

        fn family(&self) -> ReflectionModelFamily {
            <Self as AnisotropicMicrofacetAreaDistributionModel>::family(self)
        }

        fn is_isotropic(&self) -> bool { false }

        fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel> { Box::new(*self) }

        fn as_isotropic(&self) -> Option<&dyn IsotropicMicrofacetAreaDistributionModel> { None }

        fn as_isotropic_mut(
            &mut self,
        ) -> Option<&mut dyn IsotropicMicrofacetAreaDistributionModel> {
            None
        }

        fn as_anisotropic(&self) -> Option<&dyn AnisotropicMicrofacetAreaDistributionModel> {
            Some(self)
        }

        fn as_anisotropic_mut(
            &mut self,
        ) -> Option<&mut dyn AnisotropicMicrofacetAreaDistributionModel> {
            Some(self)
        }

        #[cfg(feature = "scaled-ndf-fitting")]
        fn scale(&self) -> Option<f64> {
            <Self as AnisotropicMicrofacetAreaDistributionModel>::scale(self)
        }

        #[cfg(feature = "scaled-ndf-fitting")]
        fn set_scale(&mut self, scale: f64) {
            <Self as AnisotropicMicrofacetAreaDistributionModel>::set_scale(self, scale);
        }
    }
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

/// Report of a fitting process.
pub struct FittingReport<M> {
    /// The best model found.
    best: usize,
    /// The reports of each iteration of the fitting process.
    pub reports: Vec<(M, MinimizationReport<f64>)>,
}

impl<M> FittingReport<M> {
    pub fn best_model(&self) -> &M { &self.reports[self.best].0 }

    pub fn best_model_report(&self) -> &(M, MinimizationReport<f64>) { &self.reports[self.best] }

    pub fn print_fitting_report(&self)
    where
        M: Debug,
    {
        println!("Fitting report:");
        println!("  Best model: {:?}", self.best_model());
        println!("  Reports:");
        for (m, r) in self.reports.iter() {
            println!(
                "    - Model: {:?}, objective_function: {}",
                m, r.objective_function
            );
        }
    }
}

/// A fitting problem.
pub trait FittingProblem {
    type Model;

    /// Non linear least squares fitting using Levenberg-Marquardt algorithm.
    fn lsq_lm_fit(self) -> FittingReport<Self::Model>;
}

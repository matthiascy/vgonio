#![feature(decl_macro)]
#![feature(new_uninit)]

use base::Isotropy;
use std::fmt::Debug;

pub mod brdf;
pub mod dist;

use base::{math::Vec3, optics::ior::RefractiveIndex};

/// Family of reflection models.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReflectionModelFamily {
    /// Microfacet based reflection model.
    Microfacet,
}

// TODO: merge microfacet distribution modle kind and microfacet based bsdf
// model kind into a single enum.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MicrofacetDistributionModelKind {
    /// Beckmann microfacet distribution.
    Beckmann,
    /// Trowbridge-Reitz microfacet distribution.
    TrowbridgeReitz,
}

impl MicrofacetDistributionModelKind {
    pub fn to_str(&self) -> &'static str {
        match self {
            MicrofacetDistributionModelKind::Beckmann => "Beckmann",
            MicrofacetDistributionModelKind::TrowbridgeReitz => "Trowbridge-Reitz",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MicrofacetBasedBrdfModelKind {
    /// BRDF model based on Trowbridge-Reitz(GGX) microfacet distribution.
    TrowbridgeReitz,
    /// BRDF model based on Beckmann microfacet distribution.
    Beckmann,
}

impl MicrofacetBasedBrdfModelKind {
    pub fn to_str(&self) -> &'static str {
        match self {
            MicrofacetBasedBrdfModelKind::TrowbridgeReitz => "Trowbridge-Reitz",
            MicrofacetBasedBrdfModelKind::Beckmann => "Beckmann",
        }
    }
}

pub trait MicrofacetDistributionModel: Debug + Send {
    /// Returns the kind of the distribution.
    fn kind(&self) -> MicrofacetDistributionModelKind;

    /// Returns whether the distribution is isotropic or anisotropic.
    fn isotropy(&self) -> Isotropy;

    /// Returns the roughness parameter αx of the distribution model.
    fn alpha_x(&self) -> f64;

    /// Sets the roughness parameter αx of the distribution model.
    fn set_alpha_x(&mut self, alpha_x: f64);

    /// Returns the roughness parameter αy of the distribution model.
    fn alpha_y(&self) -> f64;

    /// Sets the roughness parameter αy of the distribution model.
    fn set_alpha_y(&mut self, alpha_y: f64);

    /// Evaluates the microfacet area distribution function with the given
    /// interested microfacet normal. m is assumed to be normalised and
    /// defined in the right-handed Y-up coordinate system.
    ///
    /// # Arguments
    ///
    /// * `cos_theta` - The cosine of the polar angle of the microfacet normal.
    /// * `cos_phi` - The cosine of the azimuthal angle of the microfacet
    ///   normal.
    fn eval_adf(&self, cos_theta: f64, cos_phi: f64) -> f64;

    // /// Evaluates the slop distribution function. TODO
    // fn eval_sdf(&self, x: f64, y: f64) -> f64;

    /// Evaluates the Smith masking-shadowing function with the incident and
    /// outgoing directions.
    fn eval_msf(&self, m: Vec3, i: Vec3, o: Vec3) -> f64 {
        self.eval_msf1(m, i) * self.eval_msf1(m, o)
    }

    /// Evaluates the Smith masking-shadowing function with either the incident
    /// or outgoing direction.
    fn eval_msf1(&self, m: Vec3, v: Vec3) -> f64;

    /// Clones the distribution model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn MicrofacetDistributionModel>;
}

impl Clone for Box<dyn MicrofacetDistributionModel> {
    fn clone(&self) -> Self { self.clone_box() }
}

pub trait MicrofacetDistributionFittingModel: MicrofacetDistributionModel {
    /// Computes the partial derivatives of the microfacet area distribution
    /// function with respect to the roughness parameters of the distribution
    /// model. The derivatives are evaluated with the Microfacet Area
    /// Distribution Function, for derivatives evaluated with the Microfacet
    /// Masking-Shadowing Function, see `params_partial_derivatives_msf`.
    ///
    /// # Arguments
    ///
    /// * `cos_thetas` - The cosines of the polar angles of the microfacet
    ///  normals.
    /// * `cos_phis` - The cosines of the azimuthal angles of the microfacet
    ///  normals.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the microfacet area distribution function
    /// with respect to the roughness parameters of the distribution model.
    /// If the distribution is isotropic, the returned vector contains the
    /// partial derivatives with respect to the single roughness parameter
    /// ∂f/∂α for microfacet normals with the given polar angles and
    /// azimuthal angles. If the distribution is anisotropic, the returned
    /// vector contains the partial derivatives with respect to the roughness
    /// parameters: ∂f/∂αx and ∂f/∂αy for microfacet normals with the given
    /// the polar angles and azimuthal angles.
    fn adf_partial_derivatives(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Box<[f64]>;

    /// Computes the partial derivatives of the masking-shadowing function G1
    /// term with respect to the roughness parameters of the distribution
    /// model. For derivatives evaluated with the Microfacet Area
    /// Dipstribution Function, see `params_partial_derivatives_adf`.
    ///
    /// # Arguments
    ///
    /// * `m` - The microfacet normal.
    /// * `i` - The incident direction.
    /// * `o` - The outgoing direction.
    fn msf_partial_derivative(&self, m: Vec3, i: Vec3, o: Vec3) -> f64;
}

macro impl_common_methods() {
    fn isotropy(&self) -> Isotropy {
        if (self.alpha_x - self.alpha_y).abs() < 1.0e-6 {
            Isotropy::Isotropic
        } else {
            Isotropy::Anisotropic
        }
    }

    fn alpha_x(&self) -> f64 { self.alpha_x }

    fn set_alpha_x(&mut self, alpha_x: f64) { self.alpha_x = alpha_x; }

    fn alpha_y(&self) -> f64 { self.alpha_y }

    fn set_alpha_y(&mut self, alpha_y: f64) { self.alpha_y = alpha_y; }
}

pub trait MicrofacetBasedBrdfModel: Debug + Send {
    /// Returns the kind of the BSDF model.
    fn kind(&self) -> MicrofacetBasedBrdfModelKind;

    /// Returns the isotropy of the model.
    fn isotropy(&self) -> Isotropy;

    /// Returns the roughness parameter αx of the model.
    fn alpha_x(&self) -> f64;

    /// Sets the roughness parameter αx of the model.
    fn set_alpha_x(&mut self, alpha_x: f64);

    /// Returns the roughness parameter αy of the model.
    fn alpha_y(&self) -> f64;

    /// Sets the roughness parameter αy of the model.
    fn set_alpha_y(&mut self, alpha_y: f64);

    /// Evaluates the BRDF model.
    fn eval(&self, wi: Vec3, wo: Vec3, ior_i: &RefractiveIndex, ior_t: &RefractiveIndex) -> f64;

    /// Evaluates the BRDF model with the given spectrum.
    fn eval_spectrum(
        &self,
        wi: Vec3,
        wo: Vec3,
        iors_i: &[RefractiveIndex],
        iors_t: &[RefractiveIndex],
    ) -> Box<[f64]>;

    // TODO: eval with spherical coordinates?

    /// Clones the model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn MicrofacetBasedBrdfModel>;
}

impl Clone for Box<dyn MicrofacetBasedBrdfModel> {
    fn clone(&self) -> Box<dyn MicrofacetBasedBrdfModel> { self.clone_box() }
}

pub trait MicrofacetBasedBrdfFittingModel: MicrofacetBasedBrdfModel {
    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model.
    ///
    /// # Arguments
    ///
    /// * `wis` - The incident directions.
    /// * `wos` - The outgoing directions.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the BRDF model with respect to the roughness
    /// parameters of the model in the order of αx and αy for each incident and
    /// outgoing direction pair.
    ///
    /// # Note
    ///
    /// The returned vector has the length of 2 times the length of `wis` times
    /// the length of `wos`. For each incident direction wi, the derivatives
    /// with respect to αx and αy are evaluated for each outgoing direction wo.
    fn partial_derivatives(
        &self,
        wis: &[Vec3],
        wos: &[Vec3],
        ior_i: &RefractiveIndex,
        ior_t: &RefractiveIndex,
    ) -> Box<[f64]>;
}

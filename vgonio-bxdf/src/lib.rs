#![feature(decl_macro)]

use std::fmt::Debug;
use vgcore::Isotropy;

mod beckmann;
mod trowbridge_reitz;

pub use beckmann::BeckmannDistribution;
pub use trowbridge_reitz::TrowbridgeReitzDistribution;
use vgcore::math::Vec3;

/// Family of reflection models.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ReflectionModelFamily {
    /// Microfacet based reflection model.
    Microfacet,
}

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

pub trait MicrofacetDistributionModel: Debug {
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
    fn adf_partial_derivatives(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Vec<f64>;

    /// Computes the partial derivatives of the masking-shadowing function G1
    /// term with respect to the roughness parameters of the distribution
    /// model. For derivatives evaluated with the Microfacet Area
    /// Distribution Function, see `params_partial_derivatives_adf`.
    ///
    /// # Arguments
    ///
    /// * `m` - The microfacet normal.
    /// * `i` - The incident direction.
    /// * `o` - The outgoing direction.
    fn msf_partial_derivative(&self, m: Vec3, i: Vec3, o: Vec3) -> f64;
}

macro impl_microfacet_distribution_common_methods() {
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

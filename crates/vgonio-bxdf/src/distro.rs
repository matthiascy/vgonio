//! Microfacet distribution function models.

mod beckmann;
mod trowbridge_reitz;

use base::{math::Vec3, Isotropy};
pub use beckmann::*;
use std::fmt::Debug;
pub use trowbridge_reitz::*;

/// Different kinds of microfacet distribution functions.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MicrofacetDistroKind {
    /// Beckmann microfacet distribution.
    #[cfg_attr(feature = "cli", clap(alias = "bk", name = "beckmann"))]
    #[default]
    Beckmann,
    /// Trowbridge-Reitz microfacet distribution.
    #[cfg_attr(feature = "cli", clap(alias = "tr", name = "trowbridge"))]
    TrowbridgeReitz,
}

impl MicrofacetDistroKind {
    pub fn to_str(&self) -> &'static str {
        match self {
            MicrofacetDistroKind::Beckmann => "Beckmann",
            MicrofacetDistroKind::TrowbridgeReitz => "Trowbridge-Reitz",
        }
    }
}

/// Common interface for microfacet distribution functions.
pub trait MicrofacetDistribution: Debug + Send + Sync {
    /// The type of the parameters of the distribution model.
    type Params = [f64; 2];

    /// Returns the roughness parameters of the distribution model.
    fn params(&self) -> Self::Params;

    /// Sets the roughness parameters of the distribution model.
    fn set_params(&mut self, params: &Self::Params);

    /// Returns the kind of the distribution.
    fn kind(&self) -> MicrofacetDistroKind;

    /// Returns whether the distribution is isotropic or anisotropic.
    fn isotropy(&self) -> Isotropy;

    /// Evaluates the lambda auxiliary function.
    fn eval_lambda(&self, w: Vec3) -> f64;

    /// Evaluates the microfacet area distribution function with the given
    /// interested microfacet normal. m is assumed to be normalised and
    /// defined in the right-handed Y-up coordinate system.
    ///
    /// # Arguments
    ///
    /// * `cos_theta` - The cosine of the polar angle of the microfacet normal.
    /// * `cos_phi` - The cosine of the azimuthal angle of the microfacet
    ///   normal.
    fn eval_ndf(&self, cos_theta: f64, cos_phi: f64) -> f64;

    // /// Evaluates the slop distribution function. TODO
    // fn eval_sdf(&self, x: f64, y: f64) -> f64;

    /// Evaluates the Smith masking-shadowing function with the incident and
    /// outgoing directions. This assumes that masking and shadowing are
    /// statistically independent.
    fn eval_msf(&self, wm: Vec3, wi: Vec3, wo: Vec3) -> f64 {
        self.eval_msf1(wm, wi) * self.eval_msf1(wm, wo)
    }

    /// TODO: evaluate msf with lambda
    /// Evaluates the Smith masking-shadowing function with either the incident
    /// or outgoing direction.
    ///
    /// $ G_1(\mathbf{\omega}) = \frac{1}{1 + \Lambda(\mathbf{\omega})} $
    fn eval_msf1(&self, wm: Vec3, w: Vec3) -> f64 {
        if wm.dot(w) <= 1.0e-6 {
            return 0.0;
        }
        let lambda = self.eval_lambda(w);
        if lambda.is_infinite() {
            return 0.0;
        }
        1.0 / (1.0 + lambda)
    }

    /// Clones the distribution model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn MicrofacetDistribution<Params = Self::Params>>;

    // TODO: do not need provide full pair of directions
    #[cfg(feature = "fitting")]
    /// Computes the partial derivatives of the microfacet area distribution
    /// function with respect to the roughness parameters of the distribution
    /// model. The derivatives are evaluated with the Microfacet Area
    /// Distribution Function, for derivatives evaluated with the Microfacet
    /// Masking-Shadowing Function, see `pd_msf`.
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
    ///
    /// # Note
    ///
    /// The partial derivatives are evaluated WITHOUT Fresnel factor. You may
    /// need to multiply the partial derivatives by the Fresnel factor if you
    /// want to use them in the fitting process.
    fn pd_ndf(&self, cos_thetas: &[f64], cos_phis: &[f64]) -> Box<[f64]>;

    #[cfg(feature = "fitting")]
    /// Computes the partial derivatives of the NDF with respect to the
    /// roughness parameters of the distribution model, evaluated with the
    /// isotropic distribution model. For derivatives evaluated with the
    /// anisotropic distribution model, see `pd_ndf`.
    fn pd_ndf_iso(&self, cos_thetas: &[f64]) -> Box<[f64]>;

    #[cfg(feature = "fitting")]
    /// Computes the partial derivatives of the masking-shadowing function G1
    /// term with respect to the roughness parameters of the distribution
    /// model. For derivatives evaluated with the Microfacet Area
    /// Distribution Function, see `pd_ndf`.
    ///
    /// # Arguments
    ///
    /// * `wms` - The microfacet normals.
    /// * `ws` - The incident or outgoing directions.
    ///
    /// # Note
    ///
    /// The partial derivatives are evaluated WITHOUT Fresnel factor. You may
    /// need to multiply the partial derivatives by the Fresnel factor if you
    /// want to use them in the fitting process.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the masking-shadowing function G1 term with
    /// respect to the roughness parameters of the distribution model. The
    /// derivatives are evaluated first with the αx and then αy for each
    /// microfacet normal and the incident or outgoing direction.
    ///
    /// To calculate the partial derivatives, `wms` is iterated first, then
    /// `ws` is iterated. The size of the returned vector is `wms.len() *
    /// ws.len() * 2` if the distribution is anisotropic, and `wms.len() *
    /// ws.len()` if the distribution is isotropic.
    fn pd_msf1(&self, wms: &[Vec3], w: &[Vec3]) -> Box<[f64]>;
}

impl<P: Clone> Clone for Box<dyn MicrofacetDistribution<Params = P>> {
    fn clone(&self) -> Self { self.clone_box() }
}

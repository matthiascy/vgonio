#[cfg(feature = "bxdf_fitting_api")]
use crate::optics::Ior;
use crate::{
    bxdf::{distro::MicrofacetDistroKind, BrdfFamily},
    Symmetry,
};
use glam::Vec3;
use std::fmt::Debug;

/// Common interface for BRDFs (analytical BRDF models).
pub trait AnalyticalBrdf: Send + Sync + Debug + 'static {
    /// The type of the parameters of the BRDF model.
    type Params;

    /// The name of the BRDF model.
    fn name(&self) -> &str;

    /// Returns the kind of the BRDF.
    fn family(&self) -> BrdfFamily;
    /// Returns the kind of the microfacet distribution function.
    fn distro(&self) -> Option<MicrofacetDistroKind> { None }
    /// Tells whether the BRDF model is isotropic or not.
    fn is_isotropic(&self) -> bool;
    /// Returns the symmetry of the BRDF model.
    fn symmetry(&self) -> Symmetry {
        if self.is_isotropic() {
            Symmetry::Isotropic
        } else {
            Symmetry::Anisotropic
        }
    }

    /// Returns the parameters of the BRDF model.
    fn params(&self) -> Self::Params;

    /// Sets the parameters of the BRDF model.
    fn set_params(&mut self, params: &Self::Params);

    /// Evaluates the BRDF ($f_r$) with the classical parameterization for any
    /// incident and outgoing direction located on the hemisphere.
    ///
    /// # Arguments
    ///
    /// * `vi` - The incident direction (normalised).
    /// * `vo` - The outgoing direction (normalised).
    fn eval(&self, vi: &Vec3, vo: &Vec3) -> f64;

    #[rustfmt::skip]
    /// Evaluates the BRDF ($f_r$) with the Rusinkiewicz parameterization.
    ///
    /// Szymon M Rusinkiewicz. A new change of variables for efficient BRDF
    /// representation. In Rendering Techniques '98, pages 11-22. Springer, 1998.
    ///
    /// See [`hd2io`] and [`io2hd`] for conversion between the incident and outgoing direction and
    /// the half and difference vector (incident direction in the frame of reference in which the
    /// halfway vector is at the North Pole).
    ///
    /// # Arguments
    ///
    /// * `vh` - The half vector.
    /// * `vd` - The difference vector.
    fn eval_hd(&self, vh: &Vec3, vd: &Vec3) -> f64;

    /// Evaluates the projected BRDF with the classical parameterization.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `vi` - The incident direction.
    /// * `vo` - The outgoing direction.
    fn evalp(&self, vi: &Vec3, vo: &Vec3) -> f64;

    /// Evaluates the projected BRDF with the Rusinkiewicz parameterization.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `vh` - The half vector.
    /// * `vd` - The difference vector.
    fn evalp_hd(&self, vh: &Vec3, vd: &Vec3) -> f64;

    /// Evaluates the projected BRDF with importance sampling.
    fn evalp_is(&self, u: f32, v: f32, vo: &Vec3, vi: &mut Vec3, pdf: &mut f32) -> f64;

    /// Importance sample f_r * cos_theta_i using two uniform variates.
    ///
    /// # Arguments
    ///
    /// * `u` - The first uniform variate.
    /// * `v` - The second uniform variate.
    /// * `vo` - The outgoing direction.
    fn sample(&self, u: f32, v: f32, vo: &Vec3) -> f64;

    /// Evaluates the PDF of a sample.
    fn pdf(&self, vi: &Vec3, vo: &Vec3) -> f64;

    #[cfg(feature = "bxdf_fitting_api")]
    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model.
    ///
    /// # Arguments
    ///
    /// * `vi` - The incident directions.
    /// * `vo` - The outgoing directions.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the BRDF model with respect to the roughness
    /// parameters of the model for each incident and outgoing direction pair.
    ///
    /// # Note
    ///
    /// The returned vector has the length of the number of parameters times the
    /// length of `i` times the length of `o`. For each incident direction
    /// `i`, the derivatives with respect to params are evaluated for each
    /// outgoing direction `o`.
    fn pds(&self, vi: &[Vec3], vo: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]>;

    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model for a single incident and outgoing
    /// direction pair.
    #[cfg(feature = "bxdf_fitting_api")]
    fn pd(&self, vi: &Vec3, vo: &Vec3, ior_i: &Ior, ior_t: &Ior) -> [f64; 2];

    #[cfg(feature = "bxdf_fitting_api")]
    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model for isotropic materials.
    ///
    /// # Arguments
    ///
    /// * `vi` - The incident directions.
    /// * `vo` - The outgoing directions.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the BRDF model with respect to the roughness
    /// parameters of the model for each incident and outgoing direction pair.
    ///
    /// # Note
    ///
    /// For each incident direction `vi`, the derivatives with respect to params
    ///
    /// are evaluated for each outgoing direction `vo`.
    /// The returned vector has the length of number of parameters times the
    /// length of `vi` times the length of `vo`.
    fn pds_iso(&self, vi: &[Vec3], vo: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]>;

    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model for a single incident and outgoing
    /// direction pair for isotropic materials.
    #[cfg(feature = "bxdf_fitting_api")]
    fn pd_iso(&self, vi: &Vec3, vo: &Vec3, ior_i: &Ior, ior_t: &Ior) -> f64;

    /// Enables cloning the BRDF model from a Boxed trait object.
    ///
    /// # Note
    ///
    /// This method is used to implement the `Clone` trait for the `Box<dyn
    /// Bxdf>` type.
    fn clone_box(&self) -> Box<dyn AnalyticalBrdf<Params = Self::Params>>;
}

impl<P: 'static + Clone> Clone for Box<dyn AnalyticalBrdf<Params = P>> {
    fn clone(&self) -> Box<dyn AnalyticalBrdf<Params = P>> { self.clone_box() }
}

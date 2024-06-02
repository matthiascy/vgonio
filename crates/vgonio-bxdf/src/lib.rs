#![feature(decl_macro)]
#![feature(new_uninit)]
#![feature(associated_type_defaults)]
#![feature(downcast_unchecked)]

extern crate core;

pub mod brdf;
pub mod distro;

use crate::brdf::Bxdf;
use base::{
    math::{cos_theta, Vec3},
    optics::{fresnel, ior::Ior},
};

// pub trait MicrofacetBrdfModel: Debug + Send + Sync {
//     /// Returns the kind of the BSDF model.
//     fn kind(&self) -> MicrofacetDistroKind;
//
//     /// Returns the isotropy of the model.
//     fn isotropy(&self) -> Isotropy;
//
//     /// Returns the roughness parameter αx of the model.
//     fn alpha_x(&self) -> f64;
//
//     /// Sets the roughness parameter αx of the model.
//     fn set_alpha_x(&mut self, alpha_x: f64);
//
//     /// Returns the roughness parameter αy of the model.
//     fn alpha_y(&self) -> f64;
//
//     /// Sets the roughness parameter αy of the model.
//     fn set_alpha_y(&mut self, alpha_y: f64);
//
//     /// Evaluates the BRDF model.
//     fn eval(
//         &self,
//         wi: Vec3,
//         wo: Vec3,
//         ior_i: &RefractiveIndexRecord,
//         ior_t: &RefractiveIndexRecord,
//     ) -> f64;
//
//     /// Evaluates the BRDF model with the given spectrum.
//     fn eval_spectrum(
//         &self,
//         wi: Vec3,
//         wo: Vec3,
//         iors_i: &[RefractiveIndexRecord],
//         iors_t: &[RefractiveIndexRecord],
//     ) -> Box<[f64]>;
//
//     // TODO: eval with spherical coordinates?
//
//     /// Clones the model into a boxed trait object.
//     fn clone_box(&self) -> Box<dyn MicrofacetBrdfModel>;
// }
//
// impl Clone for Box<dyn MicrofacetBrdfModel> {
//     fn clone(&self) -> Box<dyn MicrofacetBrdfModel> { self.clone_box() }
// }
//
// pub trait MicrofacetBrdfFittingModel: MicrofacetBrdfModel {
//     /// Computes the partial derivatives of the BRDF model with respect to
// the     /// roughness parameters of the model.
//     ///
//     /// # Arguments
//     ///
//     /// * `wis` - The incident directions.
//     /// * `wos` - The outgoing directions.
//     ///
//     /// # Returns
//     ///
//     /// The partial derivatives of the BRDF model with respect to the
// roughness     /// parameters of the model in the order of αx and αy for each
// incident and     /// outgoing direction pair.
//     ///
//     /// # Note
//     ///
//     /// The returned vector has the length of 2 times the length of `wis`
// times     /// the length of `wos`. For each incident direction wi, the
// derivatives     /// with respect to αx and αy are evaluated for each outgoing
// direction wo.     fn partial_derivatives(
//         &self,
//         wis: &[Vec3],
//         wos: &[Vec3],
//         ior_i: &RefractiveIndexRecord,
//         ior_t: &RefractiveIndexRecord,
//     ) -> Box<[f64]>;
//
//     fn partial_derivatives_isotropic(
//         &self,
//         wis: &[Vec3],
//         wos: &[Vec3],
//         ior_i: &RefractiveIndexRecord,
//         ior_t: &RefractiveIndexRecord,
//     ) -> Box<[f64]>;
//
//     fn as_ref(&self) -> &dyn MicrofacetBrdfModel;
// }

/// Structure for evaluating the reflectance combining the BRDF evaluation and
/// Fresnel term.
pub struct Scattering;

impl Scattering {
    /// Evaluates the reflectance of the given BRDF model.
    ///
    /// # Arguments
    ///
    /// * `brdf` - The BRDF model.
    /// * `wi` - The incident direction, assumed to be normalized, pointing away
    /// from the surface.
    /// * `wo` - The outgoing direction, assumed to be normalized, pointing away
    /// from the surface.
    /// * `ior_i` - The refractive index of the incident medium.
    /// * `ior_t` - The refractive index of the transmitted medium.
    pub fn eval_reflectance<P: 'static>(
        brdf: &dyn Bxdf<Params = P>,
        wi: &Vec3,
        wo: &Vec3,
        ior_i: &Ior,
        ior_t: &Ior,
    ) -> f64 {
        fresnel::reflectance(cos_theta(&(-*wi)), ior_i, ior_t) as f64 * brdf.eval(wi, wo)
    }

    pub fn eval_reflectance_spectrum<P: 'static>(
        brdf: &dyn Bxdf<Params = P>,
        wi: &Vec3,
        wo: &Vec3,
        iors_i: &[Ior],
        iors_t: &[Ior],
    ) -> Box<[f64]> {
        debug_assert_eq!(iors_i.len(), iors_t.len(), "IOR pair count mismatch");
        let mut reflectances = Box::new_uninit_slice(iors_i.len());
        for ((ior_i, ior_t), refl) in iors_i
            .iter()
            .zip(iors_t.iter())
            .zip(reflectances.iter_mut())
        {
            refl.write(Scattering::eval_reflectance(brdf, wi, wo, ior_i, ior_t));
        }
        unsafe { reflectances.assume_init() }
    }
}

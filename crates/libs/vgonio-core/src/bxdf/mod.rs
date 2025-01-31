use glam::Vec3;

mod distro;
mod measured;
mod proxy;
mod traits;

use crate::{
    math::cos_theta,
    optics::{fresnel, Ior},
};
pub use distro::*;
pub use measured::*;
pub use proxy::*;
pub use traits::*;

/// Different kinds of BRDFs.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrdfFamily {
    /// Microfacet-based BRDF.
    Microfacet,
    /// Lambertian BRDF.
    Lambert,
    /// MERL BRDF.
    Merl,
    /// UTIA BRDF.
    Utia,
}

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
    ///   from the surface.
    /// * `wo` - The outgoing direction, assumed to be normalized, pointing away
    ///   from the surface.
    /// * `ior_i` - The refractive index of the incident medium.
    /// * `ior_t` - The refractive index of the transmitted medium.
    pub fn eval_reflectance<P: 'static>(
        brdf: &dyn AnalyticalBrdf<Params = P>,
        vi: &Vec3,
        vo: &Vec3,
        ior_i: &Ior,
        ior_t: &Ior,
    ) -> f64 {
        fresnel::reflectance(cos_theta(&(-*vi)), ior_i, ior_t) as f64 * brdf.eval(vi, vo)
    }

    /// Evaluates the reflectance of the given BRDF model for a spectrum.
    ///
    /// # Arguments
    ///
    /// * `brdf` - The BRDF model.
    /// * `vi` - The incident direction, assumed to be normalized, pointing away
    ///   from the surface.
    /// * `vo` - The outgoing direction, assumed to be normalized, pointing away
    ///   from the surface.
    /// * `iors_i` - The refractive indices of the incident media.
    /// * `iors_t` - The refractive indices of the transmitted media.
    pub fn eval_reflectance_spectrum<P: 'static>(
        brdf: &dyn AnalyticalBrdf<Params = P>,
        vi: &Vec3,
        vo: &Vec3,
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
            refl.write(Scattering::eval_reflectance(brdf, vi, vo, ior_i, ior_t));
        }
        unsafe { reflectances.assume_init() }
    }
}

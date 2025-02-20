use glam::Vec3;
use std::fmt::{Display, Formatter};

mod distro;
mod measured;
mod proxy;
mod traits;

pub mod brdf;
#[cfg(feature = "bxdf-fitting")]
pub mod fitting;

use crate::{
    math::cos_theta,
    optics::{fresnel, Ior},
};
pub use distro::*;
pub use measured::*;
pub use proxy::*;
pub use traits::*;

/// Variants of the Bidirectional Scattering Distribution Function (BSDF).
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum BsdfKind {
    /// Bidirectional reflectance distribution function.
    Brdf = 0x00,

    /// Bidirectional transmittance distribution function.
    Btdf = 0x01,

    /// Bidirectional scattering-surface distribution function.
    Bssdf = 0x02,

    /// Bidirectional scattering-surface reflectance distribution function.
    Bssrdf = 0x03,

    /// Bidirectional scattering-surface transmittance distribution function.
    Bsstdf = 0x04,
}

impl Display for BsdfKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BsdfKind::Brdf => {
                write!(f, "brdf")
            },
            BsdfKind::Btdf => {
                write!(f, "btdf")
            },
            BsdfKind::Bssdf => {
                write!(f, "bssdf")
            },
            BsdfKind::Bssrdf => {
                write!(f, "bssrdf")
            },
            BsdfKind::Bsstdf => {
                write!(f, "bsstdf")
            },
        }
    }
}

impl From<u8> for BsdfKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => BsdfKind::Brdf,
            0x01 => BsdfKind::Btdf,
            0x02 => BsdfKind::Bssdf,
            0x03 => BsdfKind::Bssrdf,
            0x04 => BsdfKind::Bsstdf,
            _ => panic!("Invalid BSDF kind: {}", value),
        }
    }
}

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

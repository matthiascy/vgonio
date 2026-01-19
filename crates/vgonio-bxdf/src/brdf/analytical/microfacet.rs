//! Microfacet BRDF models.
//!
//! Implements Torrance-Sparrow style microfacet reflectance:
//! `f_r(i,o) = F(i,h) * D(h) * G(i,o,h) / (4 * cos_theta_i * cos_theta_o)`.
//!
//! Here, `i` is the incident direction, `o` is the outgoing direction, and
//! `h` is the half-vector between them. `F` is the Fresnel reflectance,
//! `D` is the microfacet normal distribution function (NDF), and
//! `G` is the geometry (shadowing-masking) function.
//!
//! The runtime BRDF evaluation keeps Fresnel separate so callers can decide
//! how to evaluate/weight spectra (see `Scattering::eval_reflectance`), while
//! the fitting/derivative helpers do include the Fresnel reflectance they are
//! passed. See Torrance and Sparrow (1967), Cook and Torrance (1982), Walter
//! et al. (2007), and Heitz (2014) for derivation and sampling guidance.
mod beckmann;
mod trowbridge_reitz;

use crate::distro::{MicrofacetDistribution, MicrofacetDistroKind};
pub use beckmann::*;
use std::fmt::Debug;
pub use trowbridge_reitz::*;

/// Microfacet BRDF model, also known as Torrance-Sparrow model.
pub struct MicrofacetBrdf<D: MicrofacetDistribution> {
    /// The microfacet distribution.
    distro: D,
}

impl<D: MicrofacetDistribution> MicrofacetBrdf<D> {
    /// Returns the kind of the microfacet distribution.
    pub fn kind(&self) -> MicrofacetDistroKind { self.distro.kind() }
}

impl<D: MicrofacetDistribution> From<D> for MicrofacetBrdf<D> {
    fn from(distro: D) -> Self { MicrofacetBrdf { distro } }
}

impl<D: MicrofacetDistribution> Debug for MicrofacetBrdf<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MicrofacetBrdf {{ distro: {:?} }}", self.distro)
    }
}

impl<D: MicrofacetDistribution + Clone> Clone for MicrofacetBrdf<D> {
    fn clone(&self) -> Self {
        MicrofacetBrdf {
            distro: self.distro.clone(),
        }
    }
}

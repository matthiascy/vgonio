//! Microfacet BRDF models.
mod beckmann;
mod trowbridge_reitz;

pub use beckmann::*;
use std::fmt::Debug;
pub use trowbridge_reitz::*;
use vgonio_core::bxdf::{MicrofacetDistribution, MicrofacetDistroKind};

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

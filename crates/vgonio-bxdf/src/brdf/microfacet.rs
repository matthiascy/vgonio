mod beckmann;
mod trowbridge_reitz;

use crate::{
    brdf::Bxdf,
    distro::{MicrofacetDistribution, MicrofacetDistroKind},
};
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

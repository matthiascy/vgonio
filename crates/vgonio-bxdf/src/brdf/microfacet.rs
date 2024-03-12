use crate::MicrofacetDistribution;

mod beckmann;
mod trowbridge_reitz;

use crate::brdf::Brdf;
pub use beckmann::*;
pub use trowbridge_reitz::*;

pub struct MicrofacetBrdf<D: MicrofacetDistribution> {
    /// The microfacet distribution.
    distro: D,
}

impl<D: MicrofacetDistribution> MicrofacetBrdf<D> {
    pub fn new(distro: D) -> Self { MicrofacetBrdf { distro } }
}

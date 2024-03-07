use crate::MicrofacetDistribution;

mod beckmann;
mod trowbridge_reitz;

pub use beckmann::*;
pub use trowbridge_reitz::*;

pub struct MicrofacetBrdf<D: MicrofacetDistribution> {
    /// The microfacet distribution.
    distro: D,
}

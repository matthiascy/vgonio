//! Microfacet related measurement.
//!
//! This module contains the code for measuring the microfacet distribution
//! (NDF, normal distribution function) and the shadowing-masking function (G,
//! geometric attenuation function).

mod distribution;
pub(crate) mod params;
mod visibility;

pub use distribution::*;
pub use visibility::*;

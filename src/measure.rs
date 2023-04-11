//! Acquisition related.

pub mod bsdf;
mod collector;
pub(crate) mod emitter; // TODO: maybe make private
pub mod measurement;
pub mod microfacet;
pub mod rtc;

pub use collector::{Collector, CollectorScheme, Patch};
pub use emitter::Emitter;

/// Enumeration of the different ways to trace rays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RtcMethod {
    /// Ray tracing using Intel's Embree library.
    #[cfg(feature = "embree")]
    Embree,
    /// Ray tracing using Nvidia's OptiX library.
    #[cfg(feature = "optix")]
    Optix,
    /// Customised grid ray tracing method.
    Grid,
}

//! Ray tracing measurement module.

#[cfg(feature = "embree")]
mod embree;

mod grid;

pub use grid::GridRayTracing;

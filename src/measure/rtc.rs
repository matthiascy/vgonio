#[cfg(feature = "embree")]
mod embree;

mod grid;

pub use grid::GridRayTracing;

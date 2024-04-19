use base::{math::Sph2, partition::SphericalPartition};
use jabr::array::DyArr;

/// A BRDF that has been measured
#[derive(Clone)]
pub struct MeasuredBrdf {
    /// Sampled BRDF data with three dimensions: ωi, ωo, λ.
    samples: DyArr<f32, 3>,
    /// Incident directions in spherical coordinates.
    wi: Vec<Sph2>,
    /// Spherical domain partition of the outgoing direction.
    part: SphericalPartition,
}

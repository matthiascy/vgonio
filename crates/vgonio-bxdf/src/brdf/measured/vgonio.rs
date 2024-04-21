//! BRDF from the VGonio simulator.
use crate::brdf::measured::{BrdfParameterisation, MeasuredBrdf, ParametrisationKind};
use base::{math::Sph2, partition::SphericalPartition};
use jabr::array::DyArr;

#[derive(Debug, Clone)]
pub struct VgonioBrdfParameterisation {
    /// The incident directions of the BRDF.
    incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF.
    outgoing: SphericalPartition,
}

impl BrdfParameterisation for VgonioBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::IncidentDirection }
}

/// BRDF from the VGonio simulator.
///
/// Sampled BRDF data has three dimensions: ωi, ωo, λ.
pub type VgonioBrdf = MeasuredBrdf<VgonioBrdfParameterisation, 3>;

impl VgonioBrdf {}

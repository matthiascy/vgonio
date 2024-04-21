//! BRDF measured in the paper "Investigation and Simulation of Diffraction on
//! Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and R. Marroquim.
use crate::brdf::measured::{BrdfParameterisation, MeasuredBrdf, ParametrisationKind};
use base::math::Sph2;
use jabr::array::DyArr;

/// BRDFs measured in the paper "Investigation and Simulation of Diffraction on
/// Rough Surfaces" are in-plane BRDFs where the incident direction and the
/// outgoing direction are in the same plane. Moreover, there are no
/// measurements at the positions where the incident and outgoing directions are
/// the same.
#[derive(Debug, Clone, Copy)]
pub struct ClausenBrdfParameterisation {
    /// The incident directions of the BRDF.
    incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF for each incident direction.
    /// Directions are stored in a 2D array with dimensions: ωi, ωo.
    outgoing: DyArr<Sph2, 2>,
    /// The number of outgoing directions per incident direction.
    num_outgoing_per_incoming: usize,
}

impl BrdfParameterisation for ClausenBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::IncidentDirection }
}

/// In-plane BRDF measured in the paper "Investigation and Simulation of
/// Diffraction on Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and
/// R. Marroquim.
///
/// BRDF samples are stored in a 3D array with dimensions: ωi, ωo, λ.
pub type ClausenBrdf = MeasuredBrdf<ClausenBrdfParameterisation, 3>;

impl ClausenBrdf {
    /// Return the number of incident directions in the measured BRDF.
    pub fn n_wi(&self) -> usize { self.param.incoming.len() }

    /// Return the number of outgoing directions for each incident direction.
    pub fn n_wo(&self) -> usize { self.param.num_outgoing_per_incoming }
}

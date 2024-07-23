use crate::brdf::measured::{BrdfParameterisation, MeasuredBrdf, ParametrisationKind};

/// Parameterisation for a measured BRDF from the MERL database: <http://www.merl.com/brdf/>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerlBrdfParameterisation {}

impl BrdfParameterisation for MerlBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::HalfVector }
}

/// BRDF from the MERL database: <http://www.merl.com/brdf/>
pub type MerlBrdf = MeasuredBrdf<MerlBrdfParameterisation, 4>;

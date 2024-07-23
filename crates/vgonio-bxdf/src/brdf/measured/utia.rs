use crate::brdf::measured::{BrdfParameterisation, MeasuredBrdf, ParametrisationKind};

/// Represent the BRDF parameterisation from UTIA: <http://btf.utia.cas.cz/>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtiaBrdfParameterisation {}

impl BrdfParameterisation for UtiaBrdfParameterisation {
    fn kind() -> ParametrisationKind { todo!() }
}

/// BRDF from UTIA: <http://btf.utia.cas.cz/>
pub type UtiaBrdf = MeasuredBrdf<UtiaBrdfParameterisation, 4>;

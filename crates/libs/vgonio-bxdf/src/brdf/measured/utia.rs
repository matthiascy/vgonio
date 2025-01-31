//! BRDF data measured by UTIA at Czech Technical University.

use crate::brdf::measured::{BrdfParam, MeasuredBrdf};
use vgonio_core::bxdf::BrdfParamKind;

/// Represent the BRDF parameterisation from UTIA: <http://btf.utia.cas.cz/>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtiaBrdfParameterisation {}

impl BrdfParam for UtiaBrdfParameterisation {
    fn kind() -> BrdfParamKind { todo!() }
}

/// BRDF from UTIA: <http://btf.utia.cas.cz/>
pub type UtiaBrdf = MeasuredBrdf<UtiaBrdfParameterisation, 4>;

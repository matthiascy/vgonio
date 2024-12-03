/// Parametrisation of the BRDF measured in RGL (https://rgl.epfl.ch/pages/lab/material-database) at EPFL by Jonathan Dupuy and Wenzel Jakob.
pub type RglBrdf = MeasuredBrdf<RglBrdfParameterisation, 3>;

pub struct RglBrdfParameterisation {}

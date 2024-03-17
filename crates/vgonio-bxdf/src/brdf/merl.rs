use crate::brdf::Bxdf;
use std::{error::Error, path::Path};

/// BRDF from the MERL database: http://www.merl.com/brdf/
pub struct MerlBrdf {
    /// Sampled BRDF data.
    samples: Box<[f64]>,
}

// impl MerlBrdf {
//     /// Load a BRDF from the MERL database.
//     pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
//         todo!("Load BRDF from MERL database")
//     }
// }

// impl Brdf for MerlBrdf {
//
// }

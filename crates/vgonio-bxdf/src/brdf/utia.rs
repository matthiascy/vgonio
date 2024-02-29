use std::path::Path;

/// BRDF from UTIA: http://btf.utia.cas.cz/
pub struct UtiaBrdf {
    /// Measured BRDF samples.
    samples: Box<[f64]>,
    norm: f64,
}

// impl UtiaBrdf {
//     pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
//         todo!("Load BRDF from file")
//     }
//
//     fn normalize(&mut self) { todo!() }
// }

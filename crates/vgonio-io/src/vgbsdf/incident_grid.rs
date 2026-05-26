//! Schema for `incident_grid.toml`.
//!
//! ```toml
//! # wi positions in measurement order — index = layer index in the EXR files.
//! # Each entry is (θ, φ) in radians, on the upper hemisphere.
//! positions = [
//!     [0.0, 0.0],
//!     [0.2617993, 0.0],     # 15°, 0°
//!     [0.2617993, 1.5707963], # 15°, 90°
//!     # ...
//! ]
//! ```
use serde::{Deserialize, Serialize};

/// TOML schema describing the grid of measured incident directions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncidentGridToml {
    /// One entry per incident direction wi, in measurement order.
    /// Each entry is [theta_rad, phi_rad].
    pub positions: Vec<[f32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let g = IncidentGridToml {
            positions: vec![[0.0, 0.0], [0.261_799, 0.0], [0.261_799, 1.570_796]],
        };
        let s = toml::to_string_pretty(&g).unwrap();
        let r: IncidentGridToml = toml::from_str(&s).unwrap();
        assert_eq!(g, r);
    }
}

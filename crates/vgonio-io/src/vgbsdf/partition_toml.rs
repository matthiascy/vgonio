//! Schema for the partition TOML file, which describes how the hemisphere is partitioned into
//! patches for the BSDF.
//!
//! ```toml
//! scheme = "beckers"                    # or "equal_angle"
//! domain = "upper"                      # or "whole", "lower"
//! n_patches = 1024
//!
//! # For Beckers scheme:
//! [beckers]
//! theta_precision_rad = 0.0392699       # π / 80, the input to new_beckers
//! n_rings = 80
//!
//! # Ring breakdown — one entry per ring. Index = ring number.
//! [[beckers.rings]]
//! theta_min_rad = 0.0
//! theta_max_rad = 0.0392699
//! phi_step_rad  = 6.2831853             # patch_count = 1 (innermost)
//! patch_count   = 1
//! base_index    = 0
//!
//! [[beckers.rings]]
//! theta_min_rad = 0.0392699
//! theta_max_rad = 0.0785398
//! phi_step_rad  = 1.0471975
//! patch_count   = 6
//! base_index    = 1
//! # ...
//! ```
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionToml {
    /// "beckers" | "equal_angle".
    pub scheme: String,
    /// "upper" | "lower" | "whole".
    pub domain: String,
    pub n_patches: u32,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beckers: Option<BeckersBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equal_angle: Option<EqualAngleBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BeckersBlock {
    pub theta_precision_rad: f32,
    pub n_rings: u32,
    pub rings: Vec<RingToml>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EqualAngleBlock {
    pub theta_precision_rad: f32,
    pub phi_precision_rad: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RingToml {
    pub theta_min_rad: f32,
    pub theta_max_rad: f32,
    pub phi_step_rad: f32,
    pub patch_count: u32,
    pub base_index: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_beckers() -> PartitionToml {
        PartitionToml {
            scheme: "beckers".into(),
            domain: "upper".into(),
            n_patches: 7,
            beckers: Some(BeckersBlock {
                theta_precision_rad: 0.039_269_9,
                n_rings: 2,
                rings: vec![
                    RingToml {
                        theta_min_rad: 0.0,
                        theta_max_rad: 0.039_269_9,
                        phi_step_rad: std::f32::consts::TAU,
                        patch_count: 1,
                        base_index: 0,
                    },
                    RingToml {
                        theta_min_rad: 0.039_269_9,
                        theta_max_rad: 0.078_539_8,
                        phi_step_rad: std::f32::consts::TAU / 6.0,
                        patch_count: 6,
                        base_index: 1,
                    },
                ],
            }),
            equal_angle: None,
        }
    }

    fn sample_equal_angle() -> PartitionToml {
        PartitionToml {
            scheme: "equal_angle".into(),
            domain: "upper".into(),
            n_patches: 360,
            beckers: None,
            equal_angle: Some(EqualAngleBlock {
                // Mirror the (theta, phi) components of `SphericalPartition::precision: Sph2`.
                theta_precision_rad: std::f32::consts::FRAC_PI_2 / 9.0,
                phi_precision_rad: std::f32::consts::TAU / 40.0,
            }),
        }
    }

    #[test]
    fn partition_round_trip_beckers() {
        let p = sample_beckers();
        let s = toml::to_string_pretty(&p).unwrap();
        let r: PartitionToml = toml::from_str(&s).unwrap();
        assert_eq!(p, r);
        assert!(s.contains("scheme = \"beckers\""));
        assert!(s.contains("[[beckers.rings]]"));
    }

    #[test]
    fn partition_round_trip_equal_angle() {
        let p = sample_equal_angle();
        let s = toml::to_string_pretty(&p).unwrap();
        let r: PartitionToml = toml::from_str(&s).unwrap();
        assert_eq!(p, r);
        // Beckers section must be omitted, not emitted as empty.
        assert!(!s.contains("[beckers]"));
    }
}

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
use vgn_core::utils::partition::{PartitionScheme, SphericalDomain, SphericalPartition};
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

impl PartitionToml {
    pub fn from_partition(partition: &SphericalPartition) -> Self {
        let scheme = match partition.scheme {
            PartitionScheme::Beckers => "beckers",
            PartitionScheme::EqualAngle => "equal_angle",
        }
        .into();
        let domain = match partition.domain {
            SphericalDomain::Upper => "upper",
            SphericalDomain::Lower => "lower",
            SphericalDomain::Whole => "whole",
        }
        .into();

        let (beckers, equal_angle) = match partition.scheme {
            PartitionScheme::Beckers => {
                let rings = partition
                    .rings
                    .iter()
                    .map(|ring| RingToml {
                        theta_min_rad: ring.theta_min,
                        theta_max_rad: ring.theta_max,
                        phi_step_rad: ring.phi_step,
                        patch_count: ring.patch_count as u32,
                        base_index: ring.base_index as u32,
                    })
                    .collect();
                (
                    Some(BeckersBlock {
                        theta_precision_rad: partition.precision.theta.as_f32(),
                        n_rings: partition.rings.len() as u32,
                        rings,
                    }),
                    None,
                )
            },
            PartitionScheme::EqualAngle => (
                None,
                Some(EqualAngleBlock {
                    theta_precision_rad: partition.precision.theta.as_f32(),
                    phi_precision_rad: partition.precision.phi.as_f32(),
                }),
            ),
        };

        Self {
            scheme,
            domain,
            n_patches: partition.n_patches() as u32,
            beckers,
            equal_angle,
        }
    }

    pub fn to_partition(&self) -> Result<SphericalPartition, String> {
        use vgn_core::{math::Sph2, units::rad};

        let domain = match self.domain.as_str() {
            "upper" => SphericalDomain::Upper,
            "lower" => SphericalDomain::Lower,
            "whole" => SphericalDomain::Whole,
            other => return Err(format!("unknown domain {other:?}")),
        };

        match self.scheme.as_str() {
            "beckers" => {
                let b = self
                    .beckers
                    .as_ref()
                    .ok_or("scheme=beckers requires [beckers] block")?;
                Ok(SphericalPartition::new_beckers(
                    domain,
                    rad!(b.theta_precision_rad),
                ))
            },
            "equal_angle" => {
                let e = self
                    .equal_angle
                    .as_ref()
                    .ok_or("scheme=equal_angle requires [equal_angle] block")?;
                Ok(SphericalPartition::new_equal_angle(
                    domain,
                    Sph2::new(rad!(e.theta_precision_rad), rad!(e.phi_precision_rad)),
                ))
            },
            other => Err(format!("unknown scheme {other:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use vgn_core::{
        units::rad,
        utils::partition::{SphericalDomain, SphericalPartition},
    };

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

    #[test]
    fn beckers_round_trip_via_partition() {
        let original = SphericalPartition::new_beckers(
            SphericalDomain::Upper,
            rad!(std::f32::consts::FRAC_PI_2 / 9.0), // 10°
        );
        let toml = PartitionToml::from_partition(&original);
        let rebuilt = toml.to_partition().unwrap();

        assert_eq!(original.n_patches(), rebuilt.n_patches());
        assert_eq!(original.scheme, rebuilt.scheme);
        // Patch-mapping identity check
        let mut a = vec![0i32; 64 * 64];
        let mut b = vec![0i32; 64 * 64];
        original.compute_pixel_patch_indices(64, 64, &mut a);
        rebuilt.compute_pixel_patch_indices(64, 64, &mut b);
        assert_eq!(
            a, b,
            "patch mapping must be identical after TOML round-trip"
        );
    }

    #[test]
    fn equal_angle_round_trip_via_partition() {
        use vgn_core::math::Sph2;
        let original = SphericalPartition::new_equal_angle(
            SphericalDomain::Upper,
            Sph2::new(
                rad!(std::f32::consts::FRAC_PI_2 / 9.0),
                rad!(std::f32::consts::TAU / 36.0),
            ),
        );
        let toml = PartitionToml::from_partition(&original);
        let rebuilt = toml.to_partition().unwrap();

        assert_eq!(original.n_patches(), rebuilt.n_patches());
    }
}

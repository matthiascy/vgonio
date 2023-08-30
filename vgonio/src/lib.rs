//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
#![feature(trait_upcasting)]
#![feature(decl_macro)]
#![feature(vec_push_within_capacity)]
#![feature(assert_matches)]
#![warn(missing_docs)]

extern crate core;
mod app;
mod error;
pub mod fitting;
mod io;
pub mod measure;
pub mod optics;
mod range;

pub use range::*;

use crate::measure::Patch;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    str::FromStr,
};
use vgcore::{
    error::VgonioError,
    math::Handedness,
    units::{rad, Radians},
};

/// Main entry point for the application.
pub fn run() -> Result<(), VgonioError> {
    use app::args::CliArgs;
    use clap::Parser;

    let args = CliArgs::parse();
    let config = app::init(&args, std::time::SystemTime::now())?;

    match args.command {
        None => app::gui::run(config),
        Some(cmd) => app::cli::run(cmd, config),
    }
}

/// Medium of the surface.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Medium {
    /// Vacuum.
    Vacuum = 0x00,
    /// Air.
    Air = 0x01,
    /// Aluminium.
    Aluminium = 0x02,
    /// Copper.
    Copper = 0x03,
}

impl From<u8> for Medium {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Vacuum,
            0x01 => Self::Air,
            0x02 => Self::Aluminium,
            0x03 => Self::Copper,
            _ => panic!("Invalid medium kind: {}", value),
        }
    }
}

/// Describes the kind of material.
pub enum MaterialKind {
    /// Material is a conductor.
    Conductor,
    /// Material is a dielectric.
    Insulator,
}

impl Medium {
    /// Returns the catalog of the medium.
    pub fn kind(&self) -> MaterialKind {
        match self {
            Self::Air | Self::Vacuum => MaterialKind::Insulator,
            Self::Aluminium | Self::Copper => MaterialKind::Conductor,
        }
    }
}

impl FromStr for Medium {
    type Err = VgonioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "air" => Ok(Self::Air),
            "vacuum" => Ok(Self::Vacuum),
            "al" => Ok(Self::Aluminium),
            "cu" => Ok(Self::Copper),
            &_ => Err(VgonioError::new("Unknown medium", None)),
        }
    }
}

/// The domain of the spherical coordinate.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SphericalDomain {
    /// Simulation happens only on upper part of the sphere.
    #[default]
    #[serde(rename = "upper_hemisphere")]
    Upper = 0x01,

    /// Simulation happens only on lower part of the sphere.
    #[serde(rename = "lower_hemisphere")]
    Lower = 0x02,

    /// Simulation happens on the whole sphere.
    #[serde(rename = "whole_sphere")]
    Whole = 0x00,
}

impl Display for SphericalDomain {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upper => write!(f, "upper hemisphere"),
            Self::Lower => write!(f, "lower hemisphere"),
            Self::Whole => write!(f, "whole sphere"),
        }
    }
}

impl TryFrom<u8> for SphericalDomain {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(Self::Upper),
            0x02 => Ok(Self::Lower),
            0x00 => Ok(Self::Whole),
            _ => Err(format!("unknown spherical domain: {}", value)),
        }
    }
}

impl SphericalDomain {
    /// Range of zenith angle in radians of the upper hemisphere.
    pub const ZENITH_RANGE_UPPER_DOMAIN: (Radians, Radians) = (Radians::ZERO, Radians::HALF_PI);
    /// Range of zenith angle in radians of the lower hemisphere.
    pub const ZENITH_RANGE_LOWER_DOMAIN: (Radians, Radians) = (Radians::HALF_PI, Radians::PI);
    /// Range of zenith angle in radians of the whole sphere.
    pub const ZENITH_RANGE_WHOLE_DOMAIN: (Radians, Radians) = (Radians::ZERO, Radians::TWO_PI);

    /// Clamps the given azimuthal and zenith angle to shape's boundaries.
    ///
    /// # Arguments
    ///
    /// `zenith` - zenith angle in radians.
    /// `azimuth` - azimuthal angle in radians.
    ///
    /// # Returns
    ///
    /// `(zenith, azimuth)` - clamped zenith and azimuth angles in radians.
    #[inline]
    pub fn clamp(&self, zenith: Radians, azimuth: Radians) -> (Radians, Radians) {
        (self.clamp_zenith(zenith), self.clamp_azimuth(azimuth))
    }

    /// Clamps the given zenith angle to shape's boundaries.
    #[inline]
    pub fn clamp_zenith(&self, zenith: Radians) -> Radians {
        let (zenith_min, zenith_max) = match self {
            SphericalDomain::Upper => Self::ZENITH_RANGE_UPPER_DOMAIN,
            SphericalDomain::Lower => Self::ZENITH_RANGE_LOWER_DOMAIN,
            SphericalDomain::Whole => Self::ZENITH_RANGE_WHOLE_DOMAIN,
        };

        zenith.clamp(zenith_min, zenith_max)
    }

    /// Clamps the given azimuthal angle to shape's boundaries.
    #[inline]
    pub fn clamp_azimuth(&self, azimuth: Radians) -> Radians {
        azimuth.clamp(Radians::ZERO, Radians::TWO_PI)
    }
}

/// Partition of the collector spherical shape, each patch served as a detector.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SphericalPartition {
    /// The collector is partitioned into a number of regions with the same
    /// angular interval.
    EqualAngle {
        /// Range of interest of the polar angle θ.
        zenith: RangeByStepSizeInclusive<Radians>,

        /// Range of interest of the azimuthal angle φ.
        azimuth: RangeByStepSizeInclusive<Radians>,
    } = 0x00,

    /// The collector is partitioned into a number of regions with the same
    /// area (solid angle), azimuthal angle φ is divided into equal intervals;
    /// polar angle θ is divided non uniformly to guarantee equal area patches.
    EqualArea {
        /// Range of interest of the polar angle θ.
        zenith: RangeByStepCountInclusive<Radians>,

        /// Range of interest interval of the azimuthal angle φ.
        azimuth: RangeByStepSizeInclusive<Radians>,
    } = 0x01,

    /// The collector is partitioned into a number of regions with the same
    /// projected area (projected solid angle).
    EqualProjectedArea {
        /// Range of interest of the polar angle θ.
        zenith: RangeByStepCountInclusive<Radians>,

        /// Range of interest interval of the azimuthal angle φ.
        azimuth: RangeByStepSizeInclusive<Radians>,
    } = 0x02,
}

impl Default for SphericalPartition {
    fn default() -> Self {
        SphericalPartition::EqualAngle {
            zenith: RangeByStepSizeInclusive::new(
                Radians::ZERO,
                Radians::HALF_PI,
                Radians::from_degrees(5.0),
            ),
            azimuth: RangeByStepSizeInclusive::new(
                Radians::ZERO,
                Radians::TWO_PI,
                Radians::from_degrees(15.0),
            ),
        }
    }
}

impl SphericalPartition {
    /// Returns human-friendly description of the partition.
    pub fn kind_str(&self) -> &'static str {
        match self {
            SphericalPartition::EqualAngle { .. } => "equal angular interval",
            SphericalPartition::EqualArea { .. } => "equal area (solid angle)",
            SphericalPartition::EqualProjectedArea { .. } => {
                "equal projected area (projected solid angle)"
            }
        }
    }

    /// Returns human-friendly description of the polar angle range.
    pub fn zenith_range_str(&self) -> String {
        match self {
            SphericalPartition::EqualAngle { zenith: theta, .. } => {
                format!(
                    "{}° - {}°, step size {}°",
                    theta.start, theta.stop, theta.step_size
                )
            }
            SphericalPartition::EqualArea { zenith: theta, .. }
            | SphericalPartition::EqualProjectedArea { zenith: theta, .. } => {
                format!(
                    "{}° - {}°, samples count {}",
                    theta.start, theta.stop, theta.step_count
                )
            }
        }
    }

    /// Returns human-friendly description of the azimuthal angle range.
    pub fn azimuth_range_str(&self) -> String {
        match self {
            SphericalPartition::EqualAngle { azimuth: phi, .. }
            | SphericalPartition::EqualArea { azimuth: phi, .. }
            | SphericalPartition::EqualProjectedArea { azimuth: phi, .. } => {
                format!(
                    "{}° - {}°, step size {}°",
                    phi.start, phi.stop, phi.step_size
                )
            }
        }
    }

    /// Checks if the partition is equal angle.
    pub const fn is_equal_angle(&self) -> bool {
        matches!(self, SphericalPartition::EqualAngle { .. })
    }

    /// Checks if the partition is equal area.
    pub const fn is_equal_area(&self) -> bool {
        matches!(self, SphericalPartition::EqualArea { .. })
    }

    /// Checks if the partition is equal projected area.
    pub const fn is_equal_projected_area(&self) -> bool {
        matches!(self, SphericalPartition::EqualProjectedArea { .. })
    }
}

impl SphericalPartition {
    /// Generate patches over the unit spherical shape.
    ///
    /// The patches are generated in the order of first azimuth, then zenith.
    pub fn generate_patches(&self) -> Vec<Patch> {
        // REVIEW: this function is not very efficient, it can be improved
        match self {
            SphericalPartition::EqualAngle { zenith, azimuth } => {
                let num_patch_zenith = zenith.step_count() - 1;
                let num_patch_azimuth = azimuth.step_count() - 1;
                log::trace!(
                    "[SphericalPartition] Generating {} EqualAngle patches over domain: {:?}",
                    num_patch_azimuth * num_patch_zenith,
                    self,
                );
                let mut patches = Vec::with_capacity(num_patch_zenith * num_patch_azimuth);
                for i_phi in 0..num_patch_azimuth {
                    for i_theta in 0..num_patch_zenith {
                        patches.push(Patch::new_partitioned(
                            (
                                i_theta as f32 * zenith.step_size + zenith.start,
                                (i_theta + 1) as f32 * zenith.step_size + zenith.start,
                            ),
                            (
                                i_phi as f32 * azimuth.step_size + azimuth.start,
                                (i_phi + 1) as f32 * azimuth.step_size + azimuth.start,
                            ),
                            Handedness::RightHandedYUp,
                        ));
                    }
                }
                patches
            }
            SphericalPartition::EqualArea { zenith, azimuth } => {
                log::trace!(
                    "[SphericalPartition] Generating EqualArea patches over domain: {:?}",
                    self
                );
                let theta_start = zenith.start;
                let theta_stop = zenith.stop;
                let count = zenith.step_count;
                let phi_start = azimuth.start;
                let phi_stop = azimuth.stop;
                let phi_step = azimuth.step_size;
                // Uniformly divide the azimuthal angle. Suppose r == 1
                // Spherical cap area = 2πrh, where r is the radius of the sphere on which
                // resides the cap, and h is the height from the top of the cap
                // to the bottom of the cap.
                let h_start = 1.0 - theta_start.cos();
                let h_stop = 1.0 - theta_stop.cos();
                let h_step = (h_stop - h_start) / count as f32;

                let n_theta = count;
                let n_phi = ((phi_stop - phi_start) / phi_step).ceil() as usize;

                let mut patches = Vec::with_capacity(n_theta * n_phi);
                for i_phi in 0..n_phi {
                    for i_theta in 0..n_theta {
                        patches.push(Patch::new_partitioned(
                            (
                                (1.0 - (h_step * i_theta as f32 + h_start)).acos().into(),
                                (1.0 - (h_step * (i_theta + 1) as f32 + h_start))
                                    .acos()
                                    .into(),
                            ),
                            (
                                phi_start + i_phi as f32 * phi_step,
                                phi_start + (i_phi + 1) as f32 * phi_step,
                            ),
                            Handedness::RightHandedYUp,
                        ));
                    }
                }
                patches
            }
            SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                log::trace!(
                    "[SphericalPartition] Generating EqualProjectedArea patches over domain: {:?}",
                    self
                );
                let n_theta = zenith.step_count;
                let n_phi = azimuth.step_count();
                // Non-uniformly divide the radius of the disk into n_theta parts to
                // ensure that the projected area of each patch is equal.
                // Suppose R = 1, the radius of each ring is r_i = sqrt(1 / n_theta)
                let ring_radius = &(0..=n_theta)
                    .map(|i| (i as f32 / n_theta as f32).sqrt())
                    .collect::<Vec<_>>();
                // Generate the patches
                (0..n_phi)
                    .flat_map(move |i_phi| {
                        (0..n_theta).map(move |i| {
                            let r_start = ring_radius[i];
                            let r_stop = ring_radius[i + 1];
                            let theta_start = (1.0 - r_start).acos();
                            let theta_stop = (1.0 - r_stop).acos();
                            let phi_start = azimuth.start + i_phi as f32 * azimuth.step_size;
                            let phi_stop = azimuth.start + (i_phi + 1) as f32 * azimuth.step_size;
                            Patch::new_partitioned(
                                (rad!(theta_start), rad!(theta_stop)),
                                (phi_start, phi_stop),
                                Handedness::RightHandedYUp,
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            }
        }
    }
}

/// Machine epsilon for `f32`.
pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

/// Returns the gamma factor for a floating point number.
pub const fn gamma_f32(n: f32) -> f32 { (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON) }

#[cfg(test)]
mod tests {
    use crate::SphericalDomain;
    use vgcore::units::deg;

    #[test]
    fn spherical_domain_clamp() {
        let domain = SphericalDomain::Upper;
        let angle = deg!(91.0);
        let clamped = domain.clamp_zenith(angle.into());
        assert_eq!(clamped, deg!(90.0));

        let domain = SphericalDomain::Lower;
        let angle = deg!(191.0);
        let clamped = domain.clamp_zenith(angle.into());
        assert_eq!(clamped, deg!(180.0));
    }

    /// Bumps a floating-point value up to the next representable value.
    #[inline]
    pub fn next_f32_up(f: f32) -> f32 {
        if f.is_infinite() && f > 0.0 {
            f
        } else if f == -0.0 {
            0.0
        } else {
            let bits = f.to_bits();
            if f >= 0.0 {
                f32::from_bits(bits + 1)
            } else {
                f32::from_bits(bits - 1)
            }
        }
    }

    /// Bumps a floating-point value down to the next representable value.
    #[inline]
    pub fn next_f32_down(f: f32) -> f32 {
        if f.is_infinite() && f < 0.0 {
            f
        } else if f == -0.0 {
            0.0
        } else {
            let bits = f.to_bits();
            if f > 0.0 {
                f32::from_bits(bits - 1)
            } else {
                f32::from_bits(bits + 1)
            }
        }
    }
}

use glam::Vec3;

use rand_distr::num_traits::abs;

/// Machine epsilon for double precision floating point numbers.
pub const MACHINE_EPSILON_F64: f64 = f64::EPSILON * 0.5;

/// Machine epsilon for single precision floating point numbers.
pub const MACHINE_EPSILON_F32: f32 = f32::EPSILON * 0.5;

/// Compute the conservative bounding of $(1 \pm \epsilon_{m})^n$ for a given
/// $n$.
pub const fn gamma(n: u32) -> f32 {
    (n as f32 * MACHINE_EPSILON_F32) / (1.0 - n as f32 * MACHINE_EPSILON_F32)
}

/// Equality test of two floating point numbers.
///
/// # Arguments
///
/// * `a`: The first number.
/// * `b`: The second number.
///
/// returns: bool
pub fn ulp_eq(a: f32, b: f32) -> bool {
    let diff = abs(a - b);
    let a_abs = abs(a);
    let b_abs = abs(b);
    if a == b {
        true
    } else if a == 0.0 || b == 0.0 || a_abs + b_abs < f32::MIN_POSITIVE {
        diff < (f32::MIN_POSITIVE * f32::EPSILON)
    } else {
        (diff / f32::min(a_abs + b_abs, f32::MAX)) < f32::EPSILON
    }
}

#[test]
fn test_approx_eq() {
    assert!(ulp_eq(0.0, 0.0));
    assert!(ulp_eq(1.0, 1.0 + MACHINE_EPSILON_F32));
    assert!(ulp_eq(1.0, 1.0 + 1e-7 * 0.5));
    assert!(ulp_eq(1.0, 1.0 - 1e-7 * 0.5));
    assert!(!ulp_eq(1.0, 1.0 + 1e-6));
    assert!(!ulp_eq(1.0, 1.0 - 1e-6));
}

mod numeric;
mod range;

pub use numeric::*;
pub use range::*;
use std::fmt::Display;

use crate::{measure::Patch, units::Radians};
use serde::{Deserialize, Serialize};

/// Spherical coordinate in radians.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SphericalCoord {
    /// Zenith angle (polar angle) in radians. 0 is the zenith, pi is the nadir.
    /// The zenith angle is the angle between the positive z-axis and the point
    /// on the sphere. The zenith angle is always between 0 and pi. 0 ~ pi/2 is
    /// the upper hemisphere, pi/2 ~ pi is the lower hemisphere.
    pub zenith: Radians,
    /// Azimuth angle (azimuthal angle) in radians. It is always between 0 and
    /// 2pi: 0 is the positive x-axis, pi/2 is the positive y-axis, pi is the
    /// negative x-axis, 3pi/2 is the negative y-axis.
    pub azimuth: Radians,
}

impl SphericalCoord {
    /// Convert to a cartesian coordinate.
    pub fn into_cartesian(self) -> glam::Vec3 {
        let theta = self.zenith;
        let phi = self.azimuth;
        glam::Vec3::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        )
    }
}

/// The domain of the spherical coordinate.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SphericalDomain {
    /// Simulation happens only on upper part of the sphere.
    #[serde(rename = "upper_hemisphere")]
    Upper,

    /// Simulation happens only on lower part of the sphere.
    #[serde(rename = "lower_hemisphere")]
    Lower,

    /// Simulation happens on the whole sphere.
    #[serde(rename = "whole_sphere")]
    Whole,
}

impl Display for SphericalDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upper => write!(f, "upper hemisphere"),
            Self::Lower => write!(f, "lower hemisphere"),
            Self::Whole => write!(f, "whole sphere"),
        }
    }
}

impl SphericalDomain {
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
        let (zenith_min, zenith_max) = match self {
            SphericalDomain::Upper => (Radians::ZERO, Radians::HALF_PI),
            SphericalDomain::Lower => (Radians::HALF_PI, Radians::PI),
            SphericalDomain::Whole => (Radians::ZERO, Radians::PI),
        };

        (
            zenith.clamp(zenith_min, zenith_max),
            azimuth.clamp(Radians::ZERO, Radians::TWO_PI),
        )
    }

    /// Clamps the given zenith angle to shape's boundaries.
    pub fn clamp_zenith(&self, zenith: Radians) -> Radians {
        let (zenith_min, zenith_max) = match self {
            SphericalDomain::Upper => (Radians::ZERO, Radians::HALF_PI),
            SphericalDomain::Lower => (Radians::HALF_PI, Radians::PI),
            SphericalDomain::Whole => (Radians::ZERO, Radians::PI),
        };

        zenith.clamp(zenith_min, zenith_max)
    }

    /// Clamps the given azimuthal angle to shape's boundaries.
    pub fn clamp_azimuth(&self, azimuth: Radians) -> Radians {
        azimuth.clamp(Radians::ZERO, Radians::TWO_PI)
    }
}

/// Partition of the collector spherical shape, each patch served as a detector.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SphericalPartition {
    /// The collector is partitioned into a number of regions with the same
    /// angular interval.
    EqualAngle {
        /// Range of interest of the polar angle θ.
        zenith: RangeByStepSize<Radians>,

        /// Range of interest of the azimuthal angle φ.
        azimuth: RangeByStepSize<Radians>,
    },

    /// The collector is partitioned into a number of regions with the same
    /// area (solid angle), azimuthal angle φ is divided into equal intervals;
    /// polar angle θ is divided non uniformly to guarantee equal area patches.
    EqualArea {
        /// Range of interest of the polar angle θ.
        zenith: RangeByStepCount<Radians>,

        /// Range of interest interval of the azimuthal angle φ.
        azimuth: RangeByStepSize<Radians>,
    },

    /// The collector is partitioned into a number of regions with the same
    /// projected area (projected solid angle).
    EqualProjectedArea {
        /// Range of interest of the polar angle θ.
        zenith: RangeByStepCount<Radians>,

        /// Range of interest interval of the azimuthal angle φ.
        azimuth: RangeByStepSize<Radians>,
    },
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
}

// todo: improve the implementation of this function
impl SphericalPartition {
    /// Generate patches over the spherical shape. The angle range of the
    /// partition is limited by `SphericalDomain`.
    pub fn generate_patches_over_domain(&self, _domain: &SphericalDomain) -> Vec<Patch> {
        // REVIEW: this function is not very efficient, it can be improved
        match self {
            SphericalPartition::EqualAngle { zenith, azimuth } => {
                let n_zenith = zenith.step_count();
                let n_azimuth = azimuth.step_count();

                let mut patches = Vec::with_capacity(n_zenith * n_azimuth);
                for i_theta in 0..n_zenith - 1 {
                    for i_phi in 0..n_azimuth - 1 {
                        patches.push(Patch::new(
                            (
                                i_theta as f32 * zenith.step_size + zenith.start,
                                (i_theta + 1) as f32 * zenith.step_size + zenith.start,
                            ),
                            (
                                i_phi as f32 * azimuth.step_size + azimuth.start,
                                (i_phi + 1) as f32 * azimuth.step_size + azimuth.start,
                            ),
                        ));
                    }
                }
                patches
            }
            SphericalPartition::EqualArea {
                zenith:
                    RangeByStepCount {
                        start: theta_start,
                        stop: theta_stop,
                        step_count: count,
                    },
                azimuth:
                    RangeByStepSize {
                        start: phi_start,
                        stop: phi_stop,
                        step_size: phi_step,
                    },
            } => {
                // TODO: revision
                // Uniformly divide the azimuthal angle. Suppose r == 1
                // Spherical cap area = 2πrh, where r is the radius of the sphere on which
                // resides the cap, and h is the height from the top of the cap
                // to the bottom of the cap.
                let h_start = 1.0 - theta_start.cos();
                let h_stop = 1.0 - theta_stop.cos();
                let h_step = (h_stop - h_start) / *count as f32;

                let n_theta = *count;
                let n_phi = ((*phi_stop - *phi_start) / *phi_step).ceil() as usize;

                let mut patches = Vec::with_capacity(n_theta * n_phi);
                for i_theta in 0..n_theta {
                    for i_phi in 0..n_phi {
                        patches.push(Patch::new(
                            (
                                (1.0 - (h_step * i_theta as f32 + h_start)).acos().into(),
                                (1.0 - (h_step * (i_theta + 1) as f32 + h_start))
                                    .acos()
                                    .into(),
                            ),
                            (
                                *phi_start + i_phi as f32 * *phi_step,
                                *phi_start + (i_phi + 1) as f32 * *phi_step,
                            ),
                        ));
                    }
                }
                patches
            }
            SphericalPartition::EqualProjectedArea {
                zenith:
                    RangeByStepCount {
                        start: theta_start,
                        stop: theta_stop,
                        step_count: count,
                    },
                azimuth:
                    RangeByStepSize {
                        start: phi_start,
                        stop: phi_stop,
                        step_size: phi_step,
                    },
            } => {
                // TODO: revision
                // Non-uniformly divide the radius of the disk after the projection.
                // Disk area is linearly proportional to squared radius.
                // Calculate radius range.
                let r_start = theta_start.sin();
                let r_stop = theta_stop.sin();
                let r_start_sqr = r_start * r_start;
                let r_stop_sqr = r_stop * r_stop;
                let factor = 1.0 / *count as f32;
                let n_theta = *count;
                let n_phi = ((*phi_stop - *phi_start) / *phi_step).ceil() as usize;

                let mut patches = Vec::with_capacity(n_theta * n_phi);

                let calc_theta = |i: usize| -> f32 {
                    let r_sqr =
                        r_start_sqr + (r_stop_sqr - r_start_sqr) * factor * (i as f32 + 0.5);
                    let r = r_sqr.sqrt();
                    r.asin()
                };

                for i in 0..n_theta {
                    // Linearly interpolate squared radius range.
                    // Projected area is proportional to squared radius.
                    //                 1st           2nd           3rd
                    // O- - - - | - - - I - - - | - - - I - - - | - - - I - - -|
                    //     r_start_sqr                               r_stop_sqr
                    let theta = calc_theta(i);
                    let theta_next = calc_theta(i + 1);
                    for i_phi in 0..((*phi_stop - *phi_start) / *phi_step).ceil() as usize {
                        patches.push(Patch::new(
                            (theta.into(), theta_next.into()),
                            (
                                *phi_start + i_phi as f32 * *phi_step,
                                *phi_start + (i_phi as f32 + 1.0) * *phi_step,
                            ),
                        ));
                    }
                }
                patches
            }
        }
    }
}

pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

pub const fn gamma_f32(n: f32) -> f32 { (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON) }

/// Data encoding while storing the data to the disk.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
pub enum DataEncoding {
    /// The data is encoded as ascii text (plain text).
    Ascii,
    /// The data is encoded as binary data.
    Binary,
}

impl Display for DataEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataEncoding::Ascii => write!(f, "ascii"),
            DataEncoding::Binary => write!(f, "binary"),
        }
    }
}

impl DataEncoding {
    /// Returns true if the data is encoded as ascii text.
    pub fn is_ascii(&self) -> bool {
        match self {
            DataEncoding::Ascii => true,
            DataEncoding::Binary => false,
        }
    }

    /// Returns true if the data is encoded as binary data.
    pub fn is_binary(&self) -> bool {
        match self {
            DataEncoding::Ascii => false,
            DataEncoding::Binary => true,
        }
    }
}

/// Coordinate system handedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Handedness {
    /// Right-handed, Z-up coordinate system.
    RightHandedZUp,
    /// Right-handed, Y-up coordinate system.
    RightHandedYUp,
}

impl Handedness {
    /// Returns the up vector of the reference coordinate system.
    pub const fn up(self) -> Vec3 {
        match self {
            Self::RightHandedZUp => Vec3::Z,
            Self::RightHandedYUp => Vec3::Y,
        }
    }
}

#[test]
fn spherical_domain_clamp() {
    use crate::units::degrees;

    let domain = SphericalDomain::Upper;
    let angle = degrees!(91.0);
    let clamped = domain.clamp_zenith(angle.into());
    assert_eq!(clamped, degrees!(90.0));

    let domain = SphericalDomain::Lower;
    let angle = degrees!(191.0);
    let clamped = domain.clamp_zenith(angle.into());
    assert_eq!(clamped, degrees!(180.0));
}

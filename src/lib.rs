//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(is_some_and)]
#![feature(vec_push_within_capacity)]
#![warn(missing_docs)]
#![feature(stdsimd)] // to enable _mm_rcp14_ss

extern crate core;

mod app;
mod error;
mod io;
mod math;
pub mod measure;
pub mod msurf;
pub mod optics;
pub mod specs;
pub mod units;

use crate::{
    error::Error,
    math::{cartesian_to_spherical, spherical_to_cartesian},
    measure::Patch,
    units::{Angle, AngleUnit, Length, LengthUnit, Radians},
};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    ops::Sub,
    str::FromStr,
};

/// Main entry point for the application.
pub fn run() -> Result<(), Error> {
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
    /// Air.
    Air,
    /// Vacuum.
    Vacuum,
    /// Aluminium.
    Aluminium,
    /// Copper.
    Copper,
}

/// Describes the kind of material.
pub enum MaterialKind {
    /// Material is a conductor.
    Conductor,
    /// Material is a dielectric.
    Insulator,
}

impl Medium {
    pub fn kind(&self) -> MaterialKind {
        match self {
            Self::Air | Self::Vacuum => MaterialKind::Insulator,
            Self::Aluminium | Self::Copper => MaterialKind::Conductor,
        }
    }
}

impl FromStr for Medium {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "air" => Ok(Self::Air),
            "vacuum" => Ok(Self::Vacuum),
            "al" => Ok(Self::Aluminium),
            "cu" => Ok(Self::Copper),
            &_ => Err(Error::Any("unknown medium".to_string())),
        }
    }
}

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
/// todo: specify the error bound.
///
/// # Arguments
///
/// * `a`: The first number.
/// * `b`: The second number.
///
/// returns: bool
pub fn ulp_eq(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let a_abs = a.abs();
    let b_abs = b.abs();
    if a == b {
        true
    } else if a == 0.0 || b == 0.0 || a_abs < f32::MIN_POSITIVE || b_abs < f32::MIN_POSITIVE {
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

/// Spherical coordinate in radians.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SphericalCoord {
    /// Radius of the sphere.
    pub radius: f32,
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
    /// Create a new spherical coordinate.
    pub fn new(radius: f32, zenith: Radians, azimuth: Radians) -> Self {
        Self {
            radius,
            zenith,
            azimuth,
        }
    }

    /// Convert to a cartesian coordinate.
    pub fn to_cartesian(&self, handedness: Handedness) -> Vec3 {
        let theta = self.zenith;
        let phi = self.azimuth;
        spherical_to_cartesian(self.radius, theta, phi, handedness)
    }

    /// Convert from a cartesian coordinate.
    pub fn from_cartesian(cartesian: Vec3, radius: f32, handedness: Handedness) -> Self {
        cartesian_to_spherical(cartesian, radius, handedness)
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
                        patches.push(Patch::new_partitioned(
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
                            Handedness::RightHandedYUp,
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
                        patches.push(Patch::new_partitioned(
                            (theta.into(), theta_next.into()),
                            (
                                *phi_start + i_phi as f32 * *phi_step,
                                *phi_start + (i_phi as f32 + 1.0) * *phi_step,
                            ),
                            Handedness::RightHandedYUp,
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

/// Coordinate system handedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

// TODO: inclusive range

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given step.
#[derive(Debug, Copy, Clone)]
pub struct RangeByStepSize<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Increment between two consecutive values of the range.
    pub step_size: T,
}

impl<T: Copy + Clone> PartialEq for RangeByStepSize<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step_size == other.step_size
    }
}

impl<T: Copy + Clone> Eq for RangeByStepSize<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> RangeByStepSize<T> {
    /// Create a new range.
    pub fn new(start: T, stop: T, step_size: T) -> Self {
        Self {
            start,
            stop,
            step_size,
        }
    }

    /// Maps a function over the start and stop of the range.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> RangeByStepSize<U> {
        RangeByStepSize {
            start: f(self.start),
            stop: f(self.stop),
            step_size: f(self.step_size),
        }
    }

    /// Returns the span of the range.
    pub fn span(&self) -> T
    where
        T: Sub<Output = T>,
    {
        self.stop - self.start
    }
}

impl<T: Copy> From<[T; 3]> for RangeByStepSize<T> {
    fn from(vals: [T; 3]) -> Self {
        Self {
            start: vals[0],
            stop: vals[1],
            step_size: vals[2],
        }
    }
}

impl<T: Copy> From<RangeByStepSize<T>> for [T; 3] {
    fn from(range: RangeByStepSize<T>) -> Self { [range.start, range.stop, range.step_size] }
}

impl<T: Copy> From<(T, T, T)> for RangeByStepSize<T> {
    fn from(vals: (T, T, T)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            step_size: vals.2,
        }
    }
}

impl<T: Copy> From<RangeByStepSize<T>> for (T, T, T) {
    fn from(range: RangeByStepSize<T>) -> Self { (range.start, range.stop, range.step_size) }
}

impl<T: Default + Copy> Default for RangeByStepSize<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            step_size: T::default(),
        }
    }
}

impl<T> Display for RangeByStepSize<T>
where
    T: Display + Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ~ {} per {}", self.start, self.stop, self.step_size)
    }
}

impl<T> Serialize for RangeByStepSize<T>
where
    T: Serialize + Copy + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!(
            "{} ~ {} / {}",
            self.start, self.stop, self.step_size
        ))
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepSize<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split('~');
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {value}"))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .split('/');
        let stop = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range stop value: {value}"))?;
        let step_size = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range step size value: {value}"))?;
        Ok(Self {
            start,
            stop,
            step_size,
        })
    }
}

impl<'d, T> Deserialize<'d> for RangeByStepSize<T>
where
    T: Deserialize<'d> + Copy + FromStr,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        struct RangeByStepSizeVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> serde::de::Visitor<'de> for RangeByStepSizeVisitor<T>
        where
            T: Copy + Deserialize<'de> + FromStr,
        {
            type Value = RangeByStepSize<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a range by step size in the from of 'start to stop per
step'"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                RangeByStepSize::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepSizeVisitor::<T>(core::marker::PhantomData))
    }
}

macro impl_step_count_for_range_by_step_size($self:ident) {{
    #[cfg(debug_assertions)]
    assert!(
        $self.step_size.value() > 0.0,
        "step_size must not be greater than 0.0"
    );
    let count = $self.span() / $self.step_size;
    if count == 0.0 {
        1
    } else {
        count.ceil() as usize
    }
}}

impl RangeByStepSize<f32> {
    /// Returns the number of steps in this range.
    pub fn step_count(&self) -> usize {
        #[cfg(debug_assertions)]
        assert!(
            self.step_size > 0.0,
            "step_size must not be greater than 0.0"
        );
        let count = self.span() / self.step_size;
        if count == 0.0 {
            1
        } else {
            count.ceil() as usize
        }
    }
}

impl<A: AngleUnit> RangeByStepSize<Angle<A>> {
    /// Returns the number of steps in this range of angles.
    pub fn step_count(&self) -> usize { impl_step_count_for_range_by_step_size!(self) }
}

impl<A: LengthUnit> RangeByStepSize<Length<A>> {
    /// Returns the number of steps in this range of lengths.
    pub fn step_count(&self) -> usize { impl_step_count_for_range_by_step_size!(self) }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given number of steps.
#[derive(Debug, Copy, Clone)]
pub struct RangeByStepCount<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Number of samples.
    pub step_count: usize,
}

impl<T: Copy + Clone> PartialEq for RangeByStepCount<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step_count == other.step_count
    }
}

impl<T: Copy + Clone> Eq for RangeByStepCount<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> RangeByStepCount<T> {
    /// Create a new range.
    pub fn new(start: T, stop: T, step_count: usize) -> Self {
        Self {
            start,
            stop,
            step_count,
        }
    }

    /// Maps a function over the start and stop of the range.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> RangeByStepCount<U> {
        RangeByStepCount {
            start: f(self.start),
            stop: f(self.stop),
            step_count: self.step_count,
        }
    }

    /// Returns the span of the range.
    pub const fn span(&self) -> T
    where
        T: ~const Sub<Output = T>,
    {
        self.stop - self.start
    }
}

impl<T: Copy> From<(T, T, usize)> for RangeByStepCount<T> {
    fn from(vals: (T, T, usize)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            step_count: vals.2,
        }
    }
}

impl<T: Copy> From<RangeByStepCount<T>> for (T, T, usize) {
    fn from(range: RangeByStepCount<T>) -> Self { (range.start, range.stop, range.step_count) }
}

impl<T: Default + Copy> Default for RangeByStepCount<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            step_count: 1,
        }
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepCount<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split('~');
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {value}"))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .split(',');
        let stop = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range stop value: {value}"))?;
        let step_count = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid range step count: {value}"))?;
        Ok(Self {
            start,
            stop,
            step_count,
        })
    }
}

impl<T> Serialize for RangeByStepCount<T>
where
    T: Copy + Serialize + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!(
            "{} ~ {}, {}",
            self.start, self.stop, self.step_count
        ))
    }
}

impl<'d, T> Deserialize<'d> for RangeByStepCount<T>
where
    T: Deserialize<'d> + Copy + FromStr,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        struct RangeByStepCountVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> serde::de::Visitor<'de> for RangeByStepCountVisitor<T>
        where
            T: Copy + Deserialize<'de> + FromStr,
        {
            type Value = RangeByStepCount<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a range by step size in the from of 'start to stop per
step'"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                RangeByStepCount::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepCountVisitor::<T>(core::marker::PhantomData))
    }
}

impl RangeByStepCount<f32> {
    /// Returns the step between two consecutive values in the range.
    #[inline]
    pub const fn step_size(&self) -> f32 { self.span() / self.step_count as f32 }
}

impl<A: LengthUnit> RangeByStepCount<Length<A>> {
    /// Returns the size of the step bewteen two consecutive values in the
    /// range.
    #[inline]
    pub fn step_size(&self) -> Length<A> { self.span() / self.step_count as f32 }
}

#[test]
fn range_by_step_size_try_from_str() {
    let range = RangeByStepSize::<f32>::try_from("0.0 ~ 1.0 / 0.1").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_size, 0.1);
}

#[test]
fn range_by_step_size_try_from_str_with_angle() {
    use crate::units::{radians, Radians, URadian};
    let range = RangeByStepSize::<Radians>::try_from("0.0rad ~ 1.0rad / 0.1rad").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_size, Angle::<URadian>::new(0.1));
}

#[test]
fn range_by_step_count_try_from_str() {
    let range = RangeByStepCount::<f32>::try_from("0.0 ~ 1.0, 19").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_count, 19);
}

#[test]
fn range_by_step_count_try_from_str_with_angle() {
    use crate::units::{radians, Radians};
    let range = RangeByStepCount::<Radians>::try_from("0.0rad ~ 1.0rad, 4").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_count, 4);
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

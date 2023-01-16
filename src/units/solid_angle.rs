use crate::units::Radians;
use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
};

// TODO(yang): implement operators for SolidAngle

/// A type representing a solid angle.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SolidAngle(f32);

impl Deref for SolidAngle {
    type Target = f32;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for SolidAngle {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl SolidAngle {
    /// Creates a new solid angle from a value in steradians.
    pub fn new(value: f32) -> Self { Self(value) }

    /// Calculates the solid angle subtended by a cap on a unit sphere delimited
    /// by polar and azimuthal angles.
    pub fn from_angle_ranges(zenith: (Radians, Radians), azimuth: (Radians, Radians)) -> Self {
        solid_angle_of_region(zenith, azimuth)
    }

    /// Returns the solid angle value without units.
    pub fn value(&self) -> f32 { self.0 }
}

impl Display for SolidAngle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} sr", self.0)
    }
}

/// Helper macro to create a new `SolidAngle` from a value in `sr`.
pub macro steradians($value:expr) {
    $crate::units::SolidAngle::new($value)
}

/// Calculates the solid angle subtended by a region delimited by zenith and
/// azimuth angles.
pub fn solid_angle_of_region(
    zenith: (Radians, Radians),
    azimuth: (Radians, Radians),
) -> SolidAngle {
    SolidAngle((zenith.0.cos() - zenith.1.cos()) * (azimuth.1.value() - azimuth.0.value()))
}

/// Calculates the solid angle subtended by a spherical cap.
pub fn solid_angle_of_spherical_cap(zenith: Radians) -> SolidAngle {
    SolidAngle(2.0 * std::f32::consts::PI * (1.0 - zenith.cos()))
}

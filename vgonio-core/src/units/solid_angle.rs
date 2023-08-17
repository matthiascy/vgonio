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

    /// Returns the solid angle value in steradians.
    pub fn as_f32(&self) -> f32 { self.0 }

    /// Returns the solid angle value in steradians.
    pub fn as_f64(&self) -> f64 { self.0 as f64 }
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

/// Helper macro to create a new `SolidAngle` from a value in `sr`.
pub macro sr($value:expr) {
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

/// Calculates the solid angle subtended by a spherical strip.
///
/// A spherical strip is a region delimited by two polar angles. The upper and
/// lower bounds are given in terms of the angle values, not the position on the
/// sphere. The polar angles are measured from the top of the sphere, not the
/// center of the sphere.
///
/// # Arguments
///
/// * `upper_bound` - The upper bound of the spherical strip, in radians.
/// * `lower_bound` - The lower bound of the spherical strip, in radians.
pub fn solid_angle_of_spherical_strip(upper_bound: Radians, lower_bound: Radians) -> SolidAngle {
    debug_assert!(upper_bound.value() > lower_bound.value());
    SolidAngle(2.0 * std::f32::consts::PI * (lower_bound.cos() - upper_bound.cos()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_spherical_strip_solid_angle() {
        use super::solid_angle_of_spherical_strip;
        use crate::units::Radians;

        let lower_bound = Radians::new(0.0);
        let upper_bound = Radians::new(std::f32::consts::PI / 2.0);
        let solid_angle = solid_angle_of_spherical_strip(upper_bound, lower_bound);
        assert_eq!(solid_angle.value(), std::f32::consts::TAU);
    }
}

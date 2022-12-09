use crate::acq::Radians;
use std::ops::{Deref, DerefMut};

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
    pub fn new(value: f32) -> Self { Self(value) }

    /// Calculate the solid angle subtended by a region of the sphere delimited
    /// by two ranges of angles.
    pub fn from_angle_ranges(zenith: (Radians, Radians), azimuth: (Radians, Radians)) -> Self {
        let solid_angle = (zenith.0.cos() - zenith.1.cos()) * (azimuth.1 - azimuth.0);
        Self(solid_angle.value())
    }
}

pub macro steradians($value:expr) {
    $crate::acq::SolidAngle::new($value)
}

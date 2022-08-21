use std::ops::{Add, Div, Mul, Sub};

/// Machine epsilon for double precision floating point numbers.
pub const MACHINE_EPSILON_F64: f64 = f64::EPSILON * 0.5;

/// Machine epsilon for single precision floating point numbers.
pub const MACHINE_EPSILON_F32: f32 = f32::EPSILON * 0.5;

/// Compute the conservative bounding of $(1 \pm \epsilon_{m})^n$ for a given
/// $n$.
pub const fn gamma(n: u32) -> f32 {
    (n as f32 * MACHINE_EPSILON_F32) / (1.0 - n as f32 * MACHINE_EPSILON_F32)
}

#[derive(Debug, Copy, Clone)]
pub enum Angle {
    Rad(f32),
    Deg(f32),
}

impl Angle {
    pub fn as_rad(&self) -> f32 {
        match self {
            Angle::Rad(rad) => *rad,
            Angle::Deg(deg) => deg.to_radians(),
        }
    }

    pub fn as_deg(&self) -> f32 {
        match self {
            Angle::Rad(rad) => rad.to_degrees(),
            Angle::Deg(deg) => *deg,
        }
    }
}

impl Add for Angle {
    type Output = Angle;

    fn add(self, rhs: Angle) -> Self::Output { Angle::Rad(self.as_rad() + rhs.as_rad()) }
}

impl Sub for Angle {
    type Output = Angle;

    fn sub(self, rhs: Angle) -> Self::Output { Angle::Rad(self.as_rad() - rhs.as_rad()) }
}

impl Mul<f32> for Angle {
    type Output = Angle;

    fn mul(self, rhs: f32) -> Self::Output { Angle::Rad(self.as_rad() * rhs) }
}

impl Div<f32> for Angle {
    type Output = Angle;

    fn div(self, rhs: f32) -> Self::Output { Angle::Rad(self.as_rad() / rhs) }
}

impl Mul<Angle> for f32 {
    type Output = Angle;

    fn mul(self, rhs: Angle) -> Self::Output {
        match rhs {
            Angle::Rad(rad) => Angle::Rad(rad * self),
            Angle::Deg(deg) => Angle::Deg(deg * self),
        }
    }
}

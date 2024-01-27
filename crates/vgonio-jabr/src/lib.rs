#![feature(const_fn_floating_point_arithmetic)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self { Vec3 { x, y, z } }

    pub fn norm(&self) -> f64 { self.norm_sqr().sqrt() }

    pub const fn norm_sqr(&self) -> f64 { self.x * self.x + self.y * self.y + self.z * self.z }

    pub const fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub const fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            -(self.x * other.z - self.z * other.x),
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn normalize(&self) -> Vec3 {
        let norm = self.norm();
        Vec3::new(self.x / norm, self.y / norm, self.z / norm)
    }
}

macro_rules! impl_ops {
    ($($trait:ident, $op:ident);*) => {
        $(
            impl $trait for Vec3 {
                type Output = Vec3;

                fn $op(self, other: Vec3) -> Vec3 {
                    Vec3::new(self.x.$op(other.x), self.y.$op(other.y), self.z.$op(other.z))
                }
            }

            impl $trait<&Vec3> for Vec3 {
                type Output = Vec3;

                fn $op(self, other: &Vec3) -> Vec3 {
                    Vec3::new(self.x.$op(other.x), self.y.$op(other.y), self.z.$op(other.z))
                }
            }

            impl $trait<Vec3> for &Vec3 {
                type Output = Vec3;

                fn $op(self, other: Vec3) -> Vec3 {
                    Vec3::new(self.x.$op(other.x), self.y.$op(other.y), self.z.$op(other.z))
                }
            }

            impl $trait<&Vec3> for &Vec3 {
                type Output = Vec3;

                fn $op(self, other: &Vec3) -> Vec3 {
                    Vec3::new(self.x.$op(other.x), self.y.$op(other.y), self.z.$op(other.z))
                }
            }
        )*
    };
}

impl_ops! {
    Add, add;
    Sub, sub;
    Mul, mul
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, other: f64) -> Vec3 {
        let inv = 1.0 / other;
        Vec3::new(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl Div<f64> for &Vec3 {
    type Output = Vec3;

    fn div(self, other: f64) -> Vec3 {
        let inv = 1.0 / other;
        Vec3::new(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, other: f64) -> Vec3 { Vec3::new(self.x * other, self.y * other, self.z * other) }
}

impl Mul<f64> for &Vec3 {
    type Output = Vec3;

    fn mul(self, other: f64) -> Vec3 { Vec3::new(self.x * other, self.y * other, self.z * other) }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 { Vec3::new(self * other.x, self * other.y, self * other.z) }
}

impl Mul<&Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, other: &Vec3) -> Vec3 { Vec3::new(self * other.x, self * other.y, self * other.z) }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output { Vec3::new(-self.x, -self.y, -self.z) }
}

impl Neg for &Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output { Vec3::new(-self.x, -self.y, -self.z) }
}

pub use Vec3 as Pnt3;

pub use Vec3 as Clr3;

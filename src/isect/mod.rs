//! Intersection algorithms.

use crate::math::Vec3;

mod axis;
mod triangle;
mod aabb;

pub use axis::Axis;
pub use aabb::Aabb;

pub use triangle::ray_tri_intersect_moller_trumbore;
pub use triangle::ray_tri_intersect_woop;

/// Ray/Triangle intersection result.
#[derive(Debug)]
pub struct RayTriIsect {
    /// Barycentric coordinates of the intersection point.
    pub u: f32,

    /// Barycentric coordinates of the intersection point.
    pub v: f32,

    /// Normal of the triangle at the intersection point.
    pub n: Vec3,

    /// Intersection point.
    pub p: Vec3,
}

impl RayTriIsect {
    // TODO:
    pub fn new(p: Vec3, p_err: Vec3, n: Vec3, u: f32, v: f32) -> Self {
        Self {
            u,
            v,
            n,
            p
        }
    }
}

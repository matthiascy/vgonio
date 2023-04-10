//! Ray tracing measurement module.

use approx::RelativeEq;
use glam::Vec3;

#[cfg(feature = "embree")]
pub mod embr;

#[cfg(feature = "optix")]
pub mod optix;

pub mod grid;

mod aabb;
mod axis;
mod triangle;

use crate::common::ulp_eq;
pub use aabb::*;
pub use axis::Axis;
pub use triangle::*;

// TODO: ray packet and associated wavelengths

/// Representation of a ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// The origin of the ray.
    pub org: Vec3,

    /// The direction of the ray.
    pub dir: Vec3,
}

impl Ray {
    /// Creates a new ray (direction will be normalised).
    pub fn new(o: Vec3, d: Vec3) -> Self {
        let d = d.normalize();
        Self { org: o, dir: d }
    }

    /// Checks if the ray intersects the given axis-aligned bounding box.
    pub fn does_intersect_aabb(&self, bbox: &Aabb) -> bool { ray_aabb_intersects(self, bbox) }

    /// Checks if the ray intersects the given axis-aligned bounding box also
    /// if the ray is inside the bounding box.
    pub fn intersects_aabb(&self, bbox: &Aabb) -> Option<RayAabbIsect> {
        ray_aabb_intersection(self, bbox)
    }
}

/// Representation of a ray hit result.
#[derive(Debug, Copy, Clone)]
pub struct Hit {
    /// Geometric normal at the hit point.
    pub normal: Vec3,
    /// Hit point.
    pub point: Vec3,
    /// Barycentric coordinates of the hit point.
    pub u: f32,
    /// Barycentric coordinates of the hit point.
    pub v: f32,
    /// Geometric ID of the hit primitive.
    pub geom_id: u32,
    /// Primitive ID of the hit primitive.
    pub prim_id: u32,
}

impl Default for Hit {
    fn default() -> Self {
        Self {
            normal: Vec3::ZERO,
            point: Vec3::ZERO,
            u: f32::INFINITY,
            v: f32::INFINITY,
            geom_id: u32::MAX,
            prim_id: u32::MAX,
        }
    }
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.geom_id == other.geom_id
            && self.prim_id == other.prim_id
            && self.u.relative_eq(&other.u, f32::EPSILON, 2e-7)
            && self.v.relative_eq(&other.v, f32::EPSILON, 2e-7)
            && self.normal.abs_diff_eq(other.normal, 2e-7)
            && self.point.abs_diff_eq(other.point, 2e-7)
    }
}

impl Hit {
    /// Checks if the hit is valid.
    pub fn is_valid(&self) -> bool { self.geom_id != u32::MAX && self.prim_id != u32::MAX }

    /// Invalidates the hit.
    pub fn invalidate(&mut self) {
        self.geom_id = u32::MAX;
        self.prim_id = u32::MAX;
    }
}

// todo: make this more general, not only for triangles
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
    /// Constructs a new `RayTriIsect`. TODO: add error bounds.
    pub fn new(p: Vec3, p_err: Vec3, n: Vec3, u: f32, v: f32) -> Self { Self { u, v, n, p } }
}

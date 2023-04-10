//! Ray tracing measurement module.

use glam::Vec3;

#[cfg(feature = "embree")]
pub mod embr;

#[cfg(feature = "optix")]
pub mod optix;

pub mod grid;

mod aabb;
mod axis;
mod triangle;

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

    // /// Intersects the ray with an axis-aligned bounding box.
    // pub fn intersect_aabb(&self, bbox: &Aabb) -> Option<Vec3> {
    //     let (t_in, t_out) = ray_aabb_intersection(self, bbox);
    // }
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

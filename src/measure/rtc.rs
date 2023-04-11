//! Ray tracing measurement module.

use approx::RelativeEq;
use glam::{Vec3, Vec3A};
use std::ops::{Add, Deref, DerefMut, Mul};

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

/// Ray/Triangle intersection result.
///
/// The parameterization of a triangle uses the first vertex p0 as base point,
/// the vector p1 - p0 as u-direction and the vector p2 - p0 as v-direction.
/// Thus vertex attributes can be interpolated using the barycentric coordinates
/// u and v: t_uv = (1 - u - v) * t0 + u * t1 + v * t2.
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

    /// Interpolates the given vertex attributes according to the barycentric
    /// coordinates.
    pub fn interpolate<T>(&self, p0: T, p1: T, p2: T) -> T
    where
        T: Add<Output = T> + Mul<f32, Output = T> + Copy,
    {
        p0 * (1.0 - self.u - self.v) + p1 * self.u + p2 * self.v
    }
}

/// Hit information used for avoiding self-intersections.
#[derive(Debug, Clone, Copy)]
struct LastHit {
    /// Geometry ID of the last hit primitive.
    pub geom_id: u32,
    /// Primitive ID of the last hit primitive.
    pub prim_id: u32,
    /// Normal of the last hit primitive.
    pub normal: Vec3A,
}

/// Records the status of a traced ray.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryNode {
    /// The origin of the ray.
    pub org: Vec3A,
    /// The direction of the ray.
    pub dir: Vec3A,
    /// The cosine of the incident angle (always positive),
    /// only has value if the ray has hit the micro-surface.
    pub cos: Option<f32>,
}

/// Records the trajectory of a ray from the moment it is spawned.
///
/// The trajectory always starts with the ray that is spawned.
#[derive(Debug, Clone)]
pub struct Trajectory(pub(crate) Vec<TrajectoryNode>);

impl Deref for Trajectory {
    type Target = Vec<TrajectoryNode>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for Trajectory {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl Trajectory {
    /// Returns `true` if the ray did not hit anything.
    pub fn is_missed(&self) -> bool { self.0.len() <= 1 }

    /// Returns the last ray of the trajectory if the ray hit the micro-surface
    /// or was absorbed, `None` in case if the ray did not hit anything.
    pub fn last(&self) -> Option<&TrajectoryNode> {
        if self.is_missed() {
            None
        } else {
            self.0.last()
        }
    }

    pub fn last_mut(&mut self) -> Option<&mut TrajectoryNode> {
        if self.is_missed() {
            None
        } else {
            self.0.last_mut()
        }
    }
}

/// Maximum number of rays that can be traced in a single stream.
const MAX_RAY_STREAM_SIZE: usize = 1024;

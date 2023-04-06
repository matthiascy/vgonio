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

pub use aabb::Aabb;
pub use axis::Axis;
pub use triangle::*;

/// Representation of a ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// The origin of the ray.
    pub org: Vec3,

    /// The minimum distance along the ray direction to consider.
    pub tmin: f32,

    /// The direction of the ray.
    pub dir: Vec3,

    /// The maximum distance along the ray direction to consider.
    pub tmax: f32,
}

impl Ray {
    /// Creates a new ray (direction will be normalised).
    pub fn new(o: Vec3, d: Vec3) -> Self {
        let d = d.normalize();
        Self {
            org: o,
            dir: d,
            tmin: 0.0,
            tmax: f32::INFINITY,
        }
    }

    /// Creates a new ray with bounds (direction will be normalised).
    pub fn with_bounds(o: Vec3, d: Vec3, tmin: f32, tmax: f32) -> Self {
        let d = d.normalize();
        Self {
            org: o,
            dir: d,
            tmin,
            tmax,
        }
    }

    /// Intersects the ray with an axis-aligned bounding box.
    pub fn intersect_aabb(&self, bbox: &Aabb) -> Option<Vec3> { ray_aabb_intersection(bbox, self) }
}

/// Representation of a ray hit result.
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

fn ray_aabb_intersects_inner(ray: &Ray, bbox: &Aabb) -> (f32, f32) {
    let mut t_enter = ray.tmin;
    let mut t_exit = ray.tmax;

    for i in 0..3 {
        let inv_d = 1.0 / ray.dir[i];
        let (t_near, mut t_far) = {
            let near = (bbox.min[i] - ray.org[i]) * inv_d;
            let far = (bbox.max[i] - ray.org[i]) * inv_d;

            (near.min(far), near.max(far))
        };

        t_far *= 1.0 + 2.0 * crate::common::gamma_f32(3.0);

        t_enter = t_near.max(t_enter);
        t_exit = t_far.min(t_exit);
    }

    (t_enter, t_exit)
}

/// Tests intersection between AABB and Ray using slab method.
/// The idea is to treat the AABB as the space inside of three pairs of
/// parallel planes. The ray is clipped by each pair of parallel planes,
/// and if any portion of the ray remains, it intersects the box.
pub fn ray_aabb_intersection_p(bbox: &Aabb, ray: &Ray) -> bool {
    let (t_enter, t_exit) = ray_aabb_intersects_inner(ray, bbox);
    t_exit > t_enter && t_exit >= 0.0
}

/// Find the intersection between a ray and a axis-aligned bounding box.
pub fn ray_aabb_intersection(bbox: &Aabb, ray: &Ray) -> Option<Vec3> {
    let (t_enter, t_exit) = ray_aabb_intersects_inner(ray, bbox);

    (t_exit > t_enter && t_exit >= 0.0).then(|| {
        if t_enter < 0.0 {
            // ray origin is inside the box
            ray.org + ray.dir * t_exit
        } else {
            ray.org + ray.dir * t_enter
        }
    })
}

#[test]
fn test_ray_aabb_intersection() {
    let ray = Ray::with_bounds(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0).normalize(),
        0.0,
        10.0,
    );
    let bbox = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0));
    assert!(ray_aabb_intersection_p(&bbox, &ray));

    {
        let ray = Ray::with_bounds(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
            10.0,
        );
        let bbox = Aabb::new(Vec3::new(2.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0));
        assert!(ray_aabb_intersection_p(&bbox, &ray));
        assert_eq!(
            ray_aabb_intersection(&bbox, &ray),
            Some(Vec3::new(2.0, 0.0, 0.0))
        );

        let ray1 = Ray::with_bounds(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
            10.0,
        );
        assert!(!ray_aabb_intersection_p(&bbox, &ray1));
        assert!(ray_aabb_intersection(&bbox, &ray1).is_none());
    }
}

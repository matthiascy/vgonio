//! Ray tracing measurement module.

use glam::Vec3;

#[cfg(feature = "embree")]
pub mod embr;

#[cfg(feature = "optix")]
pub mod optix;

pub mod grid;

mod aabb;
pub mod isect;
mod axis;

pub use axis::Axis;
pub use aabb::Aabb;

/// Representation of a ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// The origin of the ray.
    pub o: Vec3,

    /// The direction of the ray.
    pub d: Vec3,
}

impl Ray {
    /// Creates a new ray (direction will be normalised).
    pub fn new(o: Vec3, d: Vec3) -> Self {
        let d = d.normalize();
        Self { o, d }
    }

    /// Intersects the ray with an axis-aligned bounding box.
    pub fn intersect_aabb(&self, bbox: &Aabb) -> Option<Vec3> {
        ray_aabb_intersection(bbox, self, 0.0, f32::INFINITY)
    }
}

fn ray_aabb_intersects_inner(ray: &Ray, bbox: &Aabb, t_min: f32, t_max: f32) -> (f32, f32) {
    let mut t_enter = t_min;
    let mut t_exit = t_max;

    for i in 0..3 {
        let inv_d = 1.0 / ray.d[i];
        let (t_near, mut t_far) = {
            let near = (bbox.min[i] - ray.o[i]) * inv_d;
            let far = (bbox.max[i] - ray.o[i]) * inv_d;

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
pub fn ray_aabb_intersection_p(bbox: &Aabb, ray: &Ray, t_min: f32, t_max: f32) -> bool {
    let (t_enter, t_exit) = ray_aabb_intersects_inner(ray, bbox, t_min, t_max);
    t_exit > t_enter && t_exit >= 0.0
}

/// Find the intersection between a ray and a axis-aligned bounding box.
pub fn ray_aabb_intersection(bbox: &Aabb, ray: &Ray, t_min: f32, t_max: f32) -> Option<Vec3> {
    let (t_enter, t_exit) = ray_aabb_intersects_inner(ray, bbox, t_min, t_max);

    (t_exit > t_enter && t_exit >= 0.0).then(|| {
        if t_enter < 0.0 {
            // ray origin is inside the box
            ray.o + ray.d * t_exit
        } else {
            ray.o + ray.d * t_enter
        }
    })
}

#[test]
fn test_ray_aabb_intersection() {
    let ray = Ray::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0).normalize(),
    );
    let bbox = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0));
    assert!(ray_aabb_intersection_p(&bbox, &ray, 0.0, 10.0));

    {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let bbox = Aabb::new(Vec3::new(2.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0));
        assert!(ray_aabb_intersection_p(&bbox, &ray, 0.0, 10.0));
        assert_eq!(
            ray_aabb_intersection(&bbox, &ray, 0.0, 10.0),
            Some(Vec3::new(2.0, 0.0, 0.0))
        );

        let ray1 = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(!ray_aabb_intersection_p(&bbox, &ray1, 0.0, 10.0));
        assert!(ray_aabb_intersection(&bbox, &ray1, 0.0, 10.0).is_none());
    }
}

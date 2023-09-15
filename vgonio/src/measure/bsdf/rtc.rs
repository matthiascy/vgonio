//! Ray tracing measurement module.

use approx::RelativeEq;
use std::{
    fmt::{Debug, Formatter},
    ops::{Add, Deref, DerefMut, Mul},
};
use vgcore::math::{Aabb, Vec3, Vec3A};

#[cfg(feature = "embree")]
pub mod embr;

#[cfg(feature = "optix")]
pub mod optix;

pub mod grid;

mod triangle;
pub use triangle::*;
use vgcore::math;

// TODO: ray packet and associated wavelengths

/// Enumeration of the different ways to trace rays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RtcMethod {
    /// Ray tracing using Intel's Embree library.
    #[cfg(feature = "embree")]
    Embree,
    /// Ray tracing using Nvidia's OptiX library.
    #[cfg(feature = "optix")]
    Optix,
    /// Customised grid ray tracing method.
    Grid,
}

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
    pub fn new(p: Vec3, _p_err: Vec3, n: Vec3, u: f32, v: f32) -> Self { Self { u, v, n, p } }

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
#[derive(Clone, Copy)]
pub struct RayTrajectoryNode {
    /// The origin of the ray.
    pub org: Vec3A,
    /// The direction of the ray.
    pub dir: Vec3A,
    /// The cosine of the incident angle (always positive),
    /// only has value if the ray has hit the micro-surface.
    pub cos: Option<f32>,
}

impl Debug for RayTrajectoryNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TrajectoryNode {{ org: {}, dir: {}, cos: {:?} }}",
            self.org, self.dir, self.cos
        )
    }
}

/// Records the trajectory of a ray from the moment it is spawned.
///
/// The trajectory always starts with the ray that is spawned.
#[derive(Debug, Clone)]
pub struct RayTrajectory(pub(crate) Vec<RayTrajectoryNode>);

impl Deref for RayTrajectory {
    type Target = Vec<RayTrajectoryNode>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for RayTrajectory {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl RayTrajectory {
    /// Returns `true` if the ray did not hit anything.
    pub fn is_missed(&self) -> bool { self.0.len() <= 1 }

    /// Returns the last ray of the trajectory if the ray hit the micro-surface
    /// or was absorbed, `None` in case if the ray did not hit anything.
    pub fn last(&self) -> Option<&RayTrajectoryNode> {
        if self.is_missed() {
            None
        } else {
            self.0.last()
        }
    }

    /// Returns a mutable reference to the last ray of the trajectory if the ray
    /// hit the micro-surface or was absorbed, `None` in case if the ray did
    /// not hit anything.
    pub fn last_mut(&mut self) -> Option<&mut RayTrajectoryNode> {
        if self.is_missed() {
            None
        } else {
            self.0.last_mut()
        }
    }
}

/// Maximum number of rays that can be traced in a single stream.
const MAX_RAY_STREAM_SIZE: usize = 1024;

/// Intersection result of a ray and an AABB.
///
/// The distance (ray parameter) is the distance from the ray origin to the
/// nearest intersection point. Only the ray with the positive distance
/// will be considered as a hit.
#[derive(Debug, Clone, Copy)]
pub enum RayAabbIsect {
    /// The ray is inside the AABB.
    Inside(f32),
    /// The ray is outside the AABB.
    Outside(f32),
}

impl RayAabbIsect {
    /// Returns true if the ray is inside the AABB.
    pub fn is_inside(&self) -> bool { matches!(self, RayAabbIsect::Inside(_)) }

    /// Returns true if the ray is outside the AABB.
    pub fn is_outside(&self) -> bool { matches!(self, RayAabbIsect::Outside(_)) }

    /// Returns the distance from the ray origin to the nearest intersection
    /// point.
    pub fn distance(&self) -> f32 {
        match self {
            RayAabbIsect::Inside(d) | RayAabbIsect::Outside(d) => *d,
        }
    }
}

/// Tests intersection between AABB and Ray using slab method.
pub fn ray_aabb_intersects(ray: &Ray, bbox: &Aabb) -> bool {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;

    for i in 0..3 {
        if ray.dir[i] == 0.0 && (ray.org[i] < bbox.min[i] || ray.org[i] > bbox.max[i]) {
            return false;
        }

        let (near, mut far) = {
            let inv_d = 1.0 / ray.dir[i];
            let near = (bbox.min[i] - ray.org[i]) * inv_d;
            let far = (bbox.max[i] - ray.org[i]) * inv_d;
            if near > far {
                (far, near)
            } else {
                (near, far)
            }
        };

        far *= 1.0 + 2.0 * crate::gamma_f32(3.0);

        t_near = t_near.max(near);
        t_far = t_far.min(far);

        if t_near > t_far {
            return false;
        }
    }

    t_far >= 0.0
}

/// Tests intersection between AABB and Ray using slab method.
/// The idea is to treat the AABB as the space inside of three pairs of
/// parallel planes. The ray is clipped by each pair of parallel planes,
/// and if any portion of the ray remains, it intersects the box.
pub fn ray_aabb_intersection(ray: &Ray, bbox: &Aabb) -> Option<RayAabbIsect> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;

    for i in 0..3 {
        if ray.dir[i] == 0.0 && (ray.org[i] < bbox.min[i] || ray.org[i] > bbox.max[i]) {
            return None;
        }

        let (near, mut far) = {
            let inv_d = math::rcp_f32(ray.dir[i]);
            let near = (bbox.min[i] - ray.org[i]) * inv_d;
            let far = (bbox.max[i] - ray.org[i]) * inv_d;
            if near > far {
                (far, near)
            } else {
                (near, far)
            }
        };

        far *= 1.0 + 2.0 * crate::gamma_f32(3.0);

        t_near = t_near.max(near);
        t_far = t_far.min(far);

        if t_near > t_far {
            return None;
        }
    }

    (t_far >= 0.0).then_some({
        if t_near >= 0.0 {
            RayAabbIsect::Outside(t_near)
        } else {
            RayAabbIsect::Inside(t_far)
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::measure::bsdf::rtc::{ray_aabb_intersection, ray_aabb_intersects, Ray};
    use vgcore::math::{Aabb, Vec3};

    #[test]
    fn aabb_ray_intersection() {
        let ray = Ray::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0).normalize(),
        );
        let bbox = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0));
        assert!(ray_aabb_intersection(&ray, &bbox).is_some());

        let bbox2 = Aabb::new(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(-1.0, -1.0, -1.0));
        assert!(ray_aabb_intersection(&ray, &bbox2).is_none());
        assert!(!ray_aabb_intersects(&ray, &bbox2));

        {
            let bbox = Aabb::new(Vec3::new(2.0, -1.0, -1.0), Vec3::new(3.0, 1.0, 1.0));

            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            assert!(ray_aabb_intersection(&ray, &bbox).is_some());

            println!("ray: {:?}", ray);
            let ray1 = Ray::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
            let isect = ray_aabb_intersection(&ray1, &bbox);
            println!("{:?}", isect);
            assert!(isect.is_none());
            assert!(!ray_aabb_intersects(&ray1, &bbox));
        }
    }

    #[test]
    fn aabb_ray_intersection_flat() {
        let aabb = Aabb::new(Vec3::new(2.0, 2.0, 0.0), Vec3::new(3.0, 3.0, 0.0));

        let normal_ray = Ray::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0).normalize(),
        );

        assert!(
            ray_aabb_intersects(&normal_ray, &aabb),
            "normal ray should intersect"
        );
        assert!(
            ray_aabb_intersection(&normal_ray, &aabb).is_some(),
            "normal ray should intersect"
        );

        {
            println!("parallel ray miss 0");
            let parallel_ray_miss0 = Ray::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0).normalize(),
            );

            assert!(
                !ray_aabb_intersects(&parallel_ray_miss0, &aabb),
                "parallel ray miss 0 should not intersect"
            );
            assert!(
                ray_aabb_intersection(&parallel_ray_miss0, &aabb).is_none(),
                "parallel ray miss 0 should not intersect"
            );

            let parallel_ray_miss00 = Ray::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0).normalize(),
            );
            assert!(
                !ray_aabb_intersects(&parallel_ray_miss00, &aabb),
                "parallel ray miss 00 should not intersect"
            );
            assert!(
                ray_aabb_intersection(&parallel_ray_miss00, &aabb).is_none(),
                "parallel ray miss 00 should not intersect"
            );
        }

        {
            println!("parallel ray miss 1");
            let parallel_ray_miss1 = Ray::new(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0).normalize(),
            );

            assert!(
                !ray_aabb_intersects(&parallel_ray_miss1, &aabb),
                "parallel ray miss 1 should not intersect"
            );
            assert!(
                ray_aabb_intersection(&parallel_ray_miss1, &aabb).is_none(),
                "parallel ray miss 1 should not intersect"
            );
        }

        {
            println!("parallel ray hit 0");
            let parallel_ray_hit0 = Ray::new(
                Vec3::new(0.0, 2.5, 0.0),
                Vec3::new(1.0, 0.0, 0.0).normalize(),
            );

            assert!(
                ray_aabb_intersects(&parallel_ray_hit0, &aabb),
                "parallel ray hit 0 should intersect"
            );
            assert!(
                ray_aabb_intersection(&parallel_ray_hit0, &aabb).is_some(),
                "parallel ray hit 0 should intersect"
            );
        }

        let parallel_ray_hit1 = Ray::new(
            Vec3::new(2.5, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0).normalize(),
        );

        assert!(
            ray_aabb_intersects(&parallel_ray_hit1, &aabb),
            "parallel ray hit 1 should intersect"
        );
        assert!(
            ray_aabb_intersection(&parallel_ray_hit1, &aabb).is_some(),
            "parallel ray hit 1 should intersect"
        );

        {
            let ray_inside = Ray::new(
                Vec3::new(2.5, 2.5, 0.0),
                Vec3::new(1.0, -1.0, 0.0).normalize(),
            );
            println!("ray inside: {:?}", ray_inside);
            assert!(
                ray_aabb_intersects(&ray_inside, &aabb),
                "ray inside should intersect"
            );
            let isect = ray_aabb_intersection(&ray_inside, &aabb);
            assert!(isect.is_some(), "ray inside should intersect");
            assert!(
                isect.unwrap().is_inside(),
                "ray inside should intersect inside"
            );
        }

        {
            let ray_inside_parallel0 = Ray::new(
                Vec3::new(2.5, 2.5, 0.0),
                Vec3::new(1.0, 0.0, 0.0).normalize(),
            );

            assert!(
                ray_aabb_intersects(&ray_inside_parallel0, &aabb),
                "ray inside parallel 0 should intersect"
            );
            let isect = ray_aabb_intersection(&ray_inside_parallel0, &aabb);
            assert!(isect.is_some(), "ray inside parallel 0 should intersect");
            assert!(
                isect.unwrap().is_inside(),
                "ray inside parallel 0 should intersect inside"
            );
        }

        {
            let ray_inside_parallel1 = Ray::new(
                Vec3::new(2.5, 2.5, 0.0),
                Vec3::new(0.0, 1.0, 0.0).normalize(),
            );

            assert!(
                ray_aabb_intersects(&ray_inside_parallel1, &aabb),
                "ray inside parallel 1 should intersect"
            );
            let isect = ray_aabb_intersection(&ray_inside_parallel1, &aabb);
            assert!(isect.is_some(), "ray inside parallel 1 should intersect");
            assert!(
                isect.unwrap().is_inside(),
                "ray inside parallel 1 should intersect inside"
            );
        }
    }
}

/// Ray tracing simulation result.
#[derive(Debug, Clone)]
pub struct RayTracingResult {
    /// The ray trajectory.
    pub trajectory: RayTrajectory,
    /// The hit result.
    pub hit: Hit,
}

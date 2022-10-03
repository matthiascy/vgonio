#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
use glam::Vec3;

use crate::acq::{util::gamma_f32, Ray};
use serde::{Deserialize, Serialize};

/// Axis-aligned bounding box.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Aabb {
    /// Minimum corner of the box.
    pub min: Vec3,

    /// Maximum corner of the box.
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }
}

impl Aabb {
    /// Creates a new [`Aabb`] from the given bounds.
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    /// Creates a new empty (zero-initialised) [`Aabb`].
    pub fn empty() -> Self { Self::default() }

    /// Construct from three points (e.g. triangle face)
    pub fn from(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        [v0, v1, v2].iter().for_each(|v| {
            for i in 0..3 {
                if v[i] < min[i] {
                    min[i] = v[i];
                }
                if v[i] > max[i] {
                    max[i] = v[i];
                }
            }
        });

        Self { min, max }
    }

    /// Tests intersection between AABB and Ray using slab method.
    /// The idea is to treat the AABB as the space inside of three pairs of
    /// parallel planes. The ray is clipped by each pair of parallel planes,
    /// and if any portion of the ray remains, it intersects the box.
    ///         // TODO: math library, component wise multiplication
    pub fn intersect_with_ray_p(&self, ray: Ray, t_min: f32, t_max: f32) -> bool {
        let mut t_enter = t_min;
        let mut t_exit = t_max;

        for i in 0..3 {
            let inv_d = 1.0 / ray.d[i];
            let (t_near, mut t_far) = {
                let near = (self.min[i] - ray.o[i]) * inv_d;
                let far = (self.max[i] - ray.o[i]) * inv_d;

                (near.min(far), near.max(far))
            };

            t_far *= 1.0 + 2.0 * gamma_f32(3.0);

            t_enter = t_near.max(t_enter);
            t_exit = t_far.min(t_exit);
        }

        t_exit > t_enter && t_exit >= 0.0
    }

    /// Find the intersection between a ray and a axis-aligned bounding box.
    pub fn intersect_with_ray(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<Vec3> {
        let mut t_enter = t_min;
        let mut t_exit = t_max;

        for i in 0..3 {
            let inv_d = 1.0 / ray.d[i];
            let (t_near, mut t_far) = {
                let near = (self.min[i] - ray.o[i]) * inv_d;
                let far = (self.max[i] - ray.o[i]) * inv_d;

                (near.min(far), near.max(far))
            };

            t_far *= 1.0 + 2.0 * gamma_f32(3.0);

            t_enter = t_near.max(t_enter);
            t_exit = t_far.min(t_exit);
        }

        (t_exit > t_enter && t_exit >= 0.0).then(|| {
            if t_enter < 0.0 {
                // ray origin is inside the box
                ray.o + ray.d * t_exit
            } else {
                ray.o + ray.d * t_enter
            }
        })
    }

    /// Extend the bounding box.
    pub fn extend(&mut self, other: &Aabb) {
        let mut min = [0.0_f32; 4];
        let mut max = [0.0_f32; 4];

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                _mm_storeu_ps(
                    min.as_mut_ptr(),
                    _mm_min_ps(
                        _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0),
                        _mm_setr_ps(other.min.x, other.min.y, other.min.z, 0.0),
                    ),
                );
                _mm_storeu_ps(
                    max.as_mut_ptr(),
                    _mm_max_ps(
                        _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0),
                        _mm_setr_ps(other.max.x, other.max.y, other.max.z, 0.0),
                    ),
                );
            }
            self.min = Vec3::new(min[0], min[1], min[2]);
            self.max = Vec3::new(max[0], max[1], max[2]);
        }

        #[cfg(target_arch = "wasm32")]
        {}
    }

    /// Maximum edge length.
    pub fn max_edge(&self) -> f32 { self.max.x.max(self.max.y).max(self.max.z) }

    // /// Union of two bounding boxes.
    // pub fn union(lhs: &Aabb, rhs: &Aabb) -> Aabb {
    //     let mut min = [0.0_f32; 4];
    //     let mut max = [0.0_f32; 4];
    //     unsafe {
    //         _mm_storeu_ps(
    //             min.as_mut_ptr(),
    //             _mm_min_ps(
    //                 _mm_setr_ps(lhs.min.x, lhs.min.y, lhs.min.z, 0.0),
    //                 _mm_setr_ps(rhs.min.x, rhs.min.y, rhs.min.z, 0.0),
    //             ),
    //         );
    //         _mm_storeu_ps(
    //             max.as_mut_ptr(),
    //             _mm_max_ps(
    //                 _mm_setr_ps(lhs.max.x, lhs.max.y, lhs.max.z, 0.0),
    //                 _mm_setr_ps(rhs.max.x, rhs.max.y, rhs.max.z, 0.0),
    //             ),
    //         );
    //     }
    //     Aabb {
    //         min: Vec3::new(min[0], min[1], min[2]),
    //         max: Vec3::new(max[0], max[1], max[2]),
    //     }
    // }

    // pub fn diagonal(&self) -> Vec3 {
    //     let mut out = [0.0_f32; 4];
    //     unsafe {
    //         _mm_storeu_ps(
    //             out.as_mut_ptr(),
    //             _mm_sub_ps(
    //                 _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0),
    //                 _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0),
    //             ),
    //         );
    //     }
    //     Vec3::new(out[0], out[1], out[2])
    // }

    // /// Compute the box center.
    // pub fn center(&self) -> Point3f {
    //     unsafe {
    //         let min_self = _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0);
    //         let max_self = _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0);
    //         let sum = _mm_add_ps(min_self, max_self);
    //         let result = _mm_mul_ps(sum, _mm_set1_ps(0.5));
    //         let mut c = [0.0f32; 4];
    //         _mm_storeu_ps(c.as_mut_ptr(), result);
    //
    //         Point3f {
    //             x: c[0],
    //             y: c[1],
    //             z: c[2],
    //         }
    //     }
    // }

    // /// Check if the box is fully contained in the box.
    // pub fn contains(&self, other: &Aabb) -> bool {
    //     unsafe {
    //         let min_self = _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0);
    //         let max_self = _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0);
    //         let min_other = _mm_setr_ps(other.min.x, other.min.y, other.min.z,
    // 0.0);         let max_other = _mm_setr_ps(other.max.x, other.max.y,
    // other.max.z, 0.0);
    //
    //         let result0 = _mm_movemask_ps(_mm_cmpge_ps(min_other, min_self));
    //         let result1 = _mm_movemask_ps(_mm_cmple_ps(max_other, max_self));
    //
    //         result0 & 0x07 == 0x07 && result1 & 0x07 == 0x07
    //     }
    // }

    // /// Check if the point is in the box.
    // pub fn contains_point(&self, point: &Point3f) -> bool {
    //     unsafe {
    //         let min_self = _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0);
    //         let max_self = _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0);
    //         let p = _mm_setr_ps(point.x, point.y, point.z, 0.0);
    //
    //         let result0 = _mm_movemask_ps(_mm_cmpge_ps(p, min_self));
    //         let result1 = _mm_movemask_ps(_mm_cmple_ps(p, max_self));
    //
    //         result0 & 0x07 == 0x07 && result1 & 0x07 == 0x07
    //     }
    // }

    // /// Enlarge the box by moving both min and max by `amount`.
    // pub fn enlarge(&mut self, amount: f32) {
    //     let amount = Vec3::new(amount, amount, amount);
    //     self.min -= amount;
    //     self.max += amount;
    // }

    // /// Compute the surface area of the box.
    // pub fn area(&self) -> f32 {
    //     let d = self.diagonal();
    //     2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    // }

    // pub fn half_area(&self) -> f32 {
    //     let d = self.diagonal();
    //     d.x * d.y + d.y * d.z + d.z * d.x
    // }
    //
    // pub fn volume(&self) -> f32 {
    //     let d = self.diagonal();
    //     d.x * d.y * d.z
    // }
    //
    // pub fn longest_axis(&self) -> Axis {
    //     let d = self.diagonal();
    //     if d.x > d.y && d.x > d.z {
    //         Axis::X
    //     } else if d.y > d.z {
    //         Axis::Y
    //     } else {
    //         Axis::Z
    //     }
    // }

    /// Compute the minimum Euclidean distance from a point on the surface of
    /// this [`Aabb`] to the point of interest.
    pub fn distance(&self, _point: &[f32; 3]) -> bool { todo!() }
}

#[test]
fn test_ray_aabb_intersection() {
    let ray = Ray::new(Vec3::new(-4.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

    println!(
        "{:?}",
        aabb.intersect_with_ray_p(ray, f32::NEG_INFINITY, f32::INFINITY)
    );
    println!(
        "{:?}",
        aabb.intersect_with_ray(ray, f32::NEG_INFINITY, f32::INFINITY)
    );

    let ray = Ray::new(Vec3::new(2.0, 2.0, 0.0), Vec3::new(-1.0, -1.0, 0.0));
    let aabb = Aabb::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0));

    println!(
        "{:?}",
        aabb.intersect_with_ray_p(ray, f32::NEG_INFINITY, f32::INFINITY)
    );
    println!(
        "{:?}",
        aabb.intersect_with_ray(ray, f32::NEG_INFINITY, f32::INFINITY)
    );
}

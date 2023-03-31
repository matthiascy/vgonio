#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
use cfg_if::cfg_if;
use glam::Vec3;

use crate::{common::gamma_f32, measure::rtc::Ray};
use serde::{Deserialize, Serialize};
use crate::common::ulp_eq;
use crate::measure::rtc::isect::Axis;

// TODO: test

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

    /// Creates a new empty (i.e. invalid) [`Aabb`].
    pub fn empty() -> Self { Self::default() }

    /// Constructs a box from three points (e.g. triangle face)
    pub fn from_points(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
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

    /// Returns the surface area of the box.
    pub fn area(&self) -> f32 {
        self.half_area() * 2.0
    }

    /// Returns half the surface area of the box.
    pub fn half_area(&self) -> f32 {
        let d = self.extent();
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    /// Returns the volume of the box.
    pub fn volume(&self) -> f32 {
        let d = self.extent();
        d.x * d.y * d.z
    }

    /// Computes the box center.
    pub fn center(&self) -> Vec3 {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let mut center = [0.0_f32; 4];
                unsafe {
                    _mm_storeu_ps(
                        center.as_mut_ptr(),
                        _mm_mul_ps(
                            _mm_add_ps(
                                _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0),
                                _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0),
                            ),
                            _mm_set1_ps(0.5), // broadcast 0.5
                        ),
                    );
                }
                Vec3::new(center[0], center[1], center[2])
            } else {
                (self.min + self.max) * 0.5
            }
        }
    }

    /// Computes the center of the box along the given axis.
    pub fn center_along_axis(&self, axis: Axis) -> f32 {
        match axis {
            Axis::X => (self.min.x + self.max.x) * 0.5,
            Axis::Y => (self.min.y + self.max.y) * 0.5,
            Axis::Z => (self.min.z + self.max.z) * 0.5,
        }
    }

    /// Checks if the box is valid.
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }

    /// Invalidates the box.
    pub fn invalidate(&mut self) {
        self.min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        self.max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    }

    /// Checks if the bounding box contains another bounding box.
    pub fn contains(&self, other: &Aabb) -> bool {
        cfg_if!{
            if #[cfg(target_arch = "x86_64")] {
                unsafe {
                    let a_min = _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0);
                    let a_max = _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0);
                    let b_min = _mm_setr_ps(other.min.x, other.min.y, other.min.z, 0.0);
                    let b_max = _mm_setr_ps(other.max.x, other.max.y, other.max.z, 0.0);
                    let cmp_min = _mm_cmple_ps(a_min, b_min);
                    let cmp_max = _mm_cmpge_ps(a_max, b_max);

                    // Combine the comparison results into a bitmask.
                    let cmp_mask = _mm_movemask_ps(_mm_and_ps(cmp_min, cmp_max));

                    // Check if the first AABB completely contains the second AABB.
                    cmp_mask == 0b111
                }
            } else {
                self.min.x <= other.min.x && self.min.y <= other.min.y && self.min.z <= other.min.z
                    && self.max.x >= other.max.x && self.max.y >= other.max.y && self.max.z >= other.max.z
            }
        }
    }

    /// Checks if the bounding box contains a point.
    pub fn contains_point(&self, point: &Vec3) -> bool {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                unsafe {
                    let min = _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0);
                    let max = _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0);
                    let p = _mm_setr_ps(point.x, point.y, point.z, 0.0);

                    let cmp_min = _mm_cmpge_ps(p, min);
                    let cmp_max = _mm_cmple_ps(p, max);

                    let cmp_mask = _mm_movemask_ps(_mm_and_ps(cmp_min, cmp_max));

                    cmp_mask == 0b111
                }
            } else {
                self.min.x <= point.x && self.min.y <= point.y && self.min.z <= point.z
                    && self.max.x >= point.x && self.max.y >= point.y && self.max.z >= point.z
            }
        }
    }

    /// Enlarges the box by moving both min and max by `amount`.
    pub fn enlarge(&mut self, amount: f32) {
        let amount = Vec3::new(amount, amount, amount);
        self.min -= amount;
        self.max += amount;
    }

    /// Computes the box diagonal.
    pub fn extent(&self) -> Vec3 {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let mut diag = [0.0_f32; 4];
                unsafe {
                    _mm_storeu_ps(
                        diag.as_mut_ptr(),
                        _mm_sub_ps(
                            _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0),
                            _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0),
                        ),
                    );
                }
                Vec3::new(diag[0], diag[1], diag[2])
            } else {
                self.max - self.min
            }
        }
    }

    /// Returns the extent of the box along the longest axis.
    pub fn max_extent(&self) -> f32 {
        let extent = self.extent();
        extent.x.max(extent.y).max(extent.z)
    }

    /// Computes the extent of the box along the given axis.
    pub fn extent_along_axis(&self, axis: Axis) -> f32 {
        match axis {
            Axis::X => self.max.x - self.min.x,
            Axis::Y => self.max.y - self.min.y,
            Axis::Z => self.max.z - self.min.z,
        }
    }

    /// Extends the box to contain another box.
    pub fn extend(&mut self, other: &Aabb) {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let mut min = [0.0_f32; 4];
                let mut max = [0.0_f32; 4];
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
            } else {
                self.min = self.min.min(other.min);
                self.max = self.max.max(other.max);
            }
        }
    }

    /// Extend the bounding box with a point.
    pub fn extend_point(&mut self, point: Vec3) {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let mut min = [0.0_f32; 4];
                let mut max = [0.0_f32; 4];
                unsafe {
                    _mm_storeu_ps(
                        min.as_mut_ptr(),
                        _mm_min_ps(
                            _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0),
                            _mm_setr_ps(point.x, point.y, point.z, 0.0),
                        ),
                    );
                    _mm_storeu_ps(
                        max.as_mut_ptr(),
                        _mm_max_ps(
                            _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0),
                            _mm_setr_ps(point.x, point.y, point.z, 0.0),
                        ),
                    );
                }
                self.min = Vec3::new(min[0], min[1], min[2]);
                self.max = Vec3::new(max[0], max[1], max[2]);
            } else {
                self.min = self.min.min(point);
                self.max = self.max.max(point);
            }
        }
    }

    /// Checks if the box is intersecting with another box.
    pub fn intersects(&self, other: &Aabb) -> bool {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let mut min = [0.0_f32; 4];
                let mut max = [0.0_f32; 4];
                unsafe {
                    _mm_storeu_ps(
                        min.as_mut_ptr(),
                        _mm_max_ps(
                            _mm_setr_ps(self.min.x, self.min.y, self.min.z, 0.0),
                            _mm_setr_ps(other.min.x, other.min.y, other.min.z, 0.0),
                        ),
                    );
                    _mm_storeu_ps(
                        max.as_mut_ptr(),
                        _mm_min_ps(
                            _mm_setr_ps(self.max.x, self.max.y, self.max.z, 0.0),
                            _mm_setr_ps(other.max.x, other.max.y, other.max.z, 0.0),
                        ),
                    );
                }
                let min = Vec3::new(min[0], min[1], min[2]);
                let max = Vec3::new(max[0], max[1], max[2]);
                min.x <= max.x && min.y <= max.y && min.z <= max.z
            } else {
                let min = self.min.max(other.min);
                let max = self.max.min(other.max);
                min.x <= max.x && min.y <= max.y && min.z <= max.z
            }
        }
    }

    /// Computes the intersection of two boxes.
    pub fn intersection(&self, other: &Aabb) -> Option<Aabb> {
        if self.intersects(other) {
            Some(Aabb {
                min: self.min.max(other.min),
                max: self.max.min(other.max),
            })
        } else {
            None
        }
    }

    /// Checks if the box is flat in any direction.
    pub fn is_flat(&self) -> bool {
        ulp_eq(self.min.x, self.max.x) || ulp_eq(self.min.y, self.max.y) || ulp_eq(self.min.z, self.max.z)
    }

    /// Returns the longest axis of the box.
    pub fn longest_axis(&self) -> Axis {
        Axis::max_axis(self.extent())
    }

    /// Unions two boxes.
    pub fn union(lhs: &Aabb, rhs: &Aabb) -> Aabb {
        let mut aabb = *lhs;
        aabb.extend(rhs);
        aabb
    }
}

impl PartialEq for Aabb {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..3 {
            if !ulp_eq(self.min[i], other.min[i]) || !ulp_eq(self.max[i], other.max[i]) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::io::empty;
    use proptest::prelude::*;
    use super::*;

    #[test]
    fn aabb_creation() {
        let aabb0 = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(aabb0.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(aabb0.max, Vec3::new(1.0, 1.0, 1.0));

        let aabb1 = Aabb::empty();
        assert_eq!(aabb1.min, Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY));
        assert_eq!(aabb1.max, Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY));
        assert!(!aabb1.is_valid());

        let aabb2 = Aabb::default();
        assert_eq!(aabb1, aabb2);
        assert!(!aabb2.is_valid());

        let mut aabb3 = Aabb::from_points(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.8), Vec3::new(1.5, 1.0, 1.0));
        assert_eq!(aabb3.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(aabb3.max, Vec3::new(1.5, 1.0, 1.8));
        assert!(aabb3.is_valid());

        aabb3.invalidate();
        assert!(!aabb3.is_valid());
    }

    const MIN: f32 = f32::MIN;
    const HALF: f32 = f32::MAX * 0.5;
    const MAX: f32 = f32::MAX;

    #[test]
    fn aabb_contains0() {
        let large_aabb = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(4.0, 4.0, 4.0));
        let small_aabb = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
        assert!(large_aabb.contains(&small_aabb));
    }

    proptest! {
        #[test]
        fn aabb_area_volume(a in MIN..HALF, b in MIN..HALF, c in MIN..HALF,
            d in HALF..MAX, e in HALF..MAX, f in HALF..MAX)
        {
            let aabb = Aabb::new(Vec3::new(a, b, c), Vec3::new(d, e, f));
            let area = aabb.area();
            let half_area = aabb.half_area();
            let d0 = d - a;
            let d1 = e - b;
            let d2 = f - c;
            let expected = 2.0 * (d0 * d1 + d0 * d2 + d1 * d2);
            prop_assert_eq!(area, expected);
            prop_assert_eq!(half_area, expected * 0.5);

            let volume = aabb.volume();
            let expected = d0 * d1 * d2;
            prop_assert_eq!(volume, expected);
        }

        #[test]
        fn aabb_center(a in MIN..HALF, b in MIN..HALF, c in MIN..HALF,
            d in HALF..MAX, e in HALF..MAX, f in HALF..MAX)
        {
            let min = Vec3::new(a, b, c);
            let max = Vec3::new(d, e, f);
            let aabb = Aabb::new(min, max);
            let center = aabb.center();
            let expected = (min + max) * 0.5;
            prop_assert_eq!(center, expected);

            let mid_x = (a + d) * 0.5;
            let mid_y = (b + e) * 0.5;
            let mid_z = (c + f) * 0.5;
            let center_x = aabb.center_along_axis(Axis::X);
            let center_y = aabb.center_along_axis(Axis::Y);
            let center_z = aabb.center_along_axis(Axis::Z);
            prop_assert_eq!(center_x, mid_x);
            prop_assert_eq!(center_y, mid_y);
            prop_assert_eq!(center_z, mid_z);
        }

        #[test]
        fn aabb_contains(a in MIN..HALF, b in MIN..HALF, c in MIN..HALF,
            d in HALF..MAX, e in HALF..MAX, f in HALF..MAX) {

            let large_aabb = Aabb::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(4.0, 4.0, 4.0));
            let small_aabb = Aabb::new(Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0));
            prop_assert!(large_aabb.contains(&small_aabb));

            // let min = Vec3::new(a, b, c);
            // let max = Vec3::new(d, e, f);
            // let large_aabb = Aabb::new(min, max);
            // let small_aabb =Aabb::new(min * 0.5, max * 0.5);
            // prop_assert!(large_aabb.contains(&small_aabb));
            // prop_assert!(!small_aabb.contains(&large_aabb));
            //
            // let point = max * 0.8;
            // prop_assert!(large_aabb.contains_point(&point));
            // prop_assert!(!small_aabb.contains_point(&point));
        }
    }
}

// /// Tests intersection between AABB and Ray using slab method.
// /// The idea is to treat the AABB as the space inside of three pairs of
// /// parallel planes. The ray is clipped by each pair of parallel planes,
// /// and if any portion of the ray remains, it intersects the box.
// ///         // TODO: math library, component wise multiplication
// pub fn ray_intersects_p(&self, ray: Ray, t_min: f32, t_max: f32) -> bool {
//     let mut t_enter = t_min;
//     let mut t_exit = t_max;
//
//     for i in 0..3 {
//         let inv_d = 1.0 / ray.d[i];
//         let (t_near, mut t_far) = {
//             let near = (self.min[i] - ray.o[i]) * inv_d;
//             let far = (self.max[i] - ray.o[i]) * inv_d;
//
//             (near.min(far), near.max(far))
//         };
//
//         t_far *= 1.0 + 2.0 * gamma_f32(3.0);
//
//         t_enter = t_near.max(t_enter);
//         t_exit = t_far.min(t_exit);
//     }
//
//     t_exit > t_enter && t_exit >= 0.0
// }
//
// /// Find the intersection between a ray and a axis-aligned bounding box.
// pub fn ray_intersects(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<Vec3> {
//     let mut t_enter = t_min;
//     let mut t_exit = t_max;
//
//     for i in 0..3 {
//         let inv_d = 1.0 / ray.d[i];
//         let (t_near, mut t_far) = {
//             let near = (self.min[i] - ray.o[i]) * inv_d;
//             let far = (self.max[i] - ray.o[i]) * inv_d;
//
//             (near.min(far), near.max(far))
//         };
//
//         t_far *= 1.0 + 2.0 * gamma_f32(3.0);
//
//         t_enter = t_near.max(t_enter);
//         t_exit = t_far.min(t_exit);
//     }
//
//     (t_exit > t_enter && t_exit >= 0.0).then(|| {
//         if t_enter < 0.0 {
//             // ray origin is inside the box
//             ray.o + ray.d * t_exit
//         } else {
//             ray.o + ray.d * t_enter
//         }
//     })
// }
//
// #[test]
// fn test_ray_aabb_intersection() {
//     let ray = Ray::new(Vec3::new(-4.0, 1.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
//     let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
//
//     println!(
//         "{:?}",
//         aabb.ray_intersects_p(ray, f32::NEG_INFINITY, f32::INFINITY)
//     );
//     println!(
//         "{:?}",
//         aabb.ray_intersects(ray, f32::NEG_INFINITY, f32::INFINITY)
//     );
//
//     let ray = Ray::new(Vec3::new(2.0, 2.0, 0.0), Vec3::new(-1.0, -1.0, 0.0));
//     let aabb = Aabb::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0));
//
//     println!(
//         "{:?}",
//         aabb.ray_intersects_p(ray, f32::NEG_INFINITY, f32::INFINITY)
//     );
//     println!(
//         "{:?}",
//         aabb.ray_intersects(ray, f32::NEG_INFINITY, f32::INFINITY)
//     );
// }

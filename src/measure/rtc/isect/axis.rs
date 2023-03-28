//! Axis enum for indexing 3D structures.
use crate::math::Vec3;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// An helper struct used to indexing element related to 3D axis.
///
/// # Examples
/// ```
/// use vgonio::isect::Axis;
///
/// let mut pos = [0.1, 0.4, 0.6];
/// pos[Axis::X] -= 0.1;
///
/// assert_eq!(pos[Axis::X], 0.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum Axis {
    /// X-axis.
    X = 0,

    /// Y-axis.
    Y = 1,

    /// Z-axis.
    Z = 2,
}

impl Axis {
    /// Given a vector, returns the axis that has the maximum component.
    pub fn max_axis(v: Vec3) -> Axis {
        if v.x > v.y && v.x > v.z {
            Axis::X
        } else if v.y > v.z {
            Axis::Y
        } else {
            Axis::Z
        }
    }

    /// Returns the next axis following the x, y, z order.
    pub fn next_axis(&self) -> Axis { Axis::from((*self as u32 + 1) % 3) }
}

impl From<i32> for Axis {
    fn from(i: i32) -> Self {
        assert!((0..3).contains(&i));
        match i {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            _ => unreachable!(),
        }
    }
}

impl From<u32> for Axis {
    fn from(i: u32) -> Self {
        assert!(i < 3);
        match i {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            _ => unreachable!(),
        }
    }
}

impl<T> Index<Axis> for [T] {
    type Output = T;

    fn index(&self, index: Axis) -> &Self::Output { &self[index as usize] }
}

impl Index<Axis> for Vec3 {
    type Output = f32;

    fn index(&self, index: Axis) -> &Self::Output {
        match index {
            Axis::X => &self.x,
            Axis::Y => &self.y,
            Axis::Z => &self.z,
        }
    }
}

impl<T> IndexMut<Axis> for [T] {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output { &mut self[index as usize] }
}

impl IndexMut<Axis> for Vec3 {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        match index {
            Axis::X => &mut self.x,
            Axis::Y => &mut self.y,
            Axis::Z => &mut self.z,
        }
    }
}

// TODO: replace with proptest
#[cfg(test)]
mod test {
    use crate::measure::rtc::isect::Axis;
    use quickcheck::quickcheck;

    quickcheck! {
        fn immutable_test(a: u32, b: u32, c: u32) -> bool {
            let v = [a, b, c];

            v[0] == v[Axis::X] && v[1] == v[Axis::Y] && v[2] == v[Axis::Z]
        }
    }

    quickcheck! {
        fn mutable_test(a: f32, b: f32, c: f32) -> bool {
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return true;
            }

            let mut v = [a, b, c];
            v[Axis::X] *= 2.0;
            v[Axis::Y] *= 1.1;
            v[Axis::Z] *= 0.6;

            v[0] == a * 2.0 && v[1] == b * 1.1 && v[2] == c * 0.6
        }
    }
}

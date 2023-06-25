//! Axis enum for indexing 3D structures.
use crate::math::Vec3;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

/// An helper struct used to indexing element related to 3D axis.
///
/// # Examples
/// ```
/// # use vgonio::measure::rtc::Axis;
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

macro impl_axis_from_traits($($t:ty)+) {
    $(
        impl From<$t> for Axis {
            fn from(i: $t) -> Self {
                debug_assert!((0..3).contains(&i));
                match i {
                    0 => Axis::X,
                    1 => Axis::Y,
                    2 => Axis::Z,
                    _ => unreachable!("Axis::from($t) is only defined for $t in [0, 3)"),
                }
            }
        }
    )+
}

impl_axis_from_traits!(u8 u16 u32 u64 usize i8 i16 i32 i64 isize);

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

#[cfg(test)]
mod test {
    use crate::measure::rtc::Axis;
    use proptest::prelude::*;

    proptest! {
        fn immutable_test(a: u32, b: u32, c: u32){
            let v = [a, b, c];

            prop_assert_eq!(v[Axis::X], a);
            prop_assert_eq!(v[Axis::Y], b);
            prop_assert_eq!(v[Axis::Z], c);
        }
    }

    proptest! {
        fn mutable_test(a: f32, b: f32, c: f32) {
            let mut v = [a, b, c];
            v[Axis::X] *= 2.0;
            v[Axis::Y] *= 1.1;
            v[Axis::Z] *= 0.6;

            prop_assert_eq!(v[Axis::X], a * 2.0);
            prop_assert_eq!(v[Axis::Y], b * 1.1);
            prop_assert_eq!(v[Axis::Z], c * 0.6);
        }
    }
}

use crate::array::{core::ArrCore, mem::heap::DynSized, MemLayout};
use std::{mem::MaybeUninit, ops::Index};

/// A dynamically sized array with a known number of dimensions at compile-time.
///
/// The dimension is set at compilation time and can't be changed, but the size
/// of each dimension can be changed at runtime.
///
/// By default, the layout of the array is row-major, and the number of
/// dimensions is 1.
pub struct DyArr<T, const N: usize = 1, const L: MemLayout = { MemLayout::RowMajor }>(
    pub(crate) ArrCore<DynSized<T>, [usize; N], L>,
);

impl<T, const N: usize, const L: MemLayout> DyArr<T, N, L> {
    forward_array_core_common_methods!();

    /// Creates a new array with the given data and shape.
    pub fn empty(shape: [usize; N]) -> DyArr<MaybeUninit<T>, N, L> {
        DyArr(ArrCore::new(
            shape,
            DynSized::new_uninit(shape.iter().product()),
        ))
    }

    /// Creates a new array with the given data and shape.
    pub fn with_data(shape: [usize; N], data: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self(ArrCore::new(shape, DynSized::from(data)))
    }

    /// Reshapes the array to the given shape.
    ///
    /// The new shape must have the same number of elements and the same number
    /// of dimensions as the original shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the array. Only one dimension size can be
    ///   -1, which means that the size of that dimension is inferred.
    pub fn reshape(&self, shape: [i32; N]) -> DyArr<T, N, L> {
        assert!(
            shape.iter().filter(|&&x| x == -1).count() <= 1,
            "Only one dimension size can be -1"
        );
        todo!()
    }
}

impl<T, const L: MemLayout, const N: usize> Index<[usize; N]> for DyArr<T, N, L> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output { &self.0[index] }
}

impl<T, const L: MemLayout, const N: usize> Clone for DyArr<T, N, L>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyarr_creation() {
        let arr = DyArr::<u32, 3>::empty([2, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
    }

    #[test]
    fn test_dyarr_reshape() {
        let arr: DyArr<MaybeUninit<i32>, 3> = DyArr::empty([2, 3, 2]);
        let arr = arr.reshape([-1, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
    }
}

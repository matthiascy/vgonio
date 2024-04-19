use crate::array::{core::ArrCore, mem::heap::DynSized, MemLayout};
use std::ops::Index;

/// A dynamically-sized array with a known number of dimensions at compile-time.
///
/// The dimension is set at compilation time and cannot be changed, but the size
/// of each dimension can be changed at runtime (non-growable).
pub struct DyArr<T, const N: usize, const L: MemLayout = { MemLayout::RowMajor }>(
    pub(crate) ArrCore<DynSized<T>, [usize; N], L>,
);

impl<T, const N: usize, const L: MemLayout> DyArr<T, N, L> {
    super::forward_core_array_methods!(@const
        shape -> &[usize], #[doc = "Returns the shape of the array."];
        strides -> &[usize], #[doc = "Returns the strides of the array."];
        order -> MemLayout, #[doc = "Returns the layout of the array."];
        dimension -> usize, #[doc = "Returns the number of dimensions of the array."];
    );

    /// Creates a new array with the given data and shape.
    pub fn new(shape: [usize; N]) -> Self {
        Self(ArrCore::new(
            shape,
            DynSized::with_capacity(shape.iter().product()),
        ))
    }

    /// Creates a new array with the given data and shape.
    pub fn with_data(shape: [usize; N], data: Vec<T>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        Self(ArrCore::new(shape, DynSized::from_vec(data)))
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
        let arr: DyArr<i32, 3> = DyArr::new([2, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
    }
}

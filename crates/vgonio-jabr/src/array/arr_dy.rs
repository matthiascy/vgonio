use crate::array::{core::ArrCore, mem::heap::DynSized, shape::compute_n_elems, MemLayout};
use num_traits::{One, Zero};
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
};

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

    /// Creates a new array from a vector.
    pub fn from_vec(shape: [usize; N], vec: Vec<T>) -> Self {
        assert_eq!(compute_n_elems(&shape), vec.len());
        Self(ArrCore::new(shape, DynSized::from_vec(vec)))
    }

    /// Creates a new array from a boxed slice.
    pub fn from_boxed_slice(shape: [usize; N], slice: Box<[T]>) -> Self {
        assert_eq!(compute_n_elems(&shape), slice.len());
        Self(ArrCore::new(shape, DynSized::from_boxed_slice(slice)))
    }

    /// Creates a new array from a slice.
    pub fn from_slice(shape: [usize; N], slice: &[T]) -> Self
    where
        T: Clone,
    {
        assert_eq!(compute_n_elems(&shape), slice.len());
        Self(ArrCore::new(shape, DynSized::from_slice(slice)))
    }

    /// Returns the array underlying data as a slice.
    pub fn as_slice(&self) -> &[T] { self.0.data.as_slice() }

    /// Returns the array underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] { self.0.data.as_mut_slice() }

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

    /// Creates a new array with all elements set to zero.
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Zero + Clone,
    {
        let n = compute_n_elems(&shape);
        Self(ArrCore::new(shape, DynSized::splat(T::zero(), n)))
    }

    /// Creates a new array with all elements set to one.
    pub fn ones(shape: [usize; N]) -> Self
    where
        T: One + Clone,
    {
        let n = compute_n_elems(&shape);
        Self(ArrCore::new(shape, DynSized::splat(T::one(), n)))
    }

    pub fn splat(value: T, shape: [usize; N]) -> Self
    where
        T: Clone,
    {
        let n = compute_n_elems(&shape);
        Self(ArrCore::new(shape, DynSized::splat(value, n)))
    }
}

impl<T, const N: usize, const L: MemLayout> Index<[usize; N]> for DyArr<T, N, L> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output { &self.0[index] }
}

impl<T, const N: usize, const L: MemLayout> IndexMut<[usize; N]> for DyArr<T, N, L> {
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output { &mut self.0[index] }
}

impl<T, const N: usize, const L: MemLayout> Clone for DyArr<T, N, L>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T, const N: usize, const L: MemLayout> Debug for DyArr<T, N, L>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DyArr({:?})", &self.0.data))
    }
}

impl<T, const N: usize, const L: MemLayout> PartialEq for DyArr<T, N, L>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T, const N: usize, const L: MemLayout> Eq for DyArr<T, N, L> where T: Eq {}

impl<T, const N: usize, const L: MemLayout> AsRef<[T]> for DyArr<T, N, L> {
    fn as_ref(&self) -> &[T] { self.as_slice() }
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

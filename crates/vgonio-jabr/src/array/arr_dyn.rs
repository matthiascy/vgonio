use num_traits::{One, Zero};

use crate::array::{
    core::ArrCore,
    mem::{heap::DynSized, MemLayout},
    shape::compute_n_elems,
};
use std::{
    fmt::{Debug, Formatter, Result},
    mem::MaybeUninit,
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

/// A dynamically sized array with dynamical number of dimensions
/// and size of each dimension at runtime (growable).
pub struct DynArr<T, const L: MemLayout = { MemLayout::RowMajor }>(
    pub(crate) ArrCore<DynSized<T>, Vec<usize>, L>,
);

impl<T, const L: MemLayout> DynArr<T, L> {
    forward_array_core_common_methods!();

    /// Creates a new array with the given data and shape.
    pub fn empty(shape: &[usize]) -> DynArr<MaybeUninit<T>, L> {
        let n_elems = shape.iter().product();
        let shape = shape.to_vec();
        DynArr(ArrCore::new(shape, DynSized::new_uninit(n_elems)))
    }

    pub fn from_vec(shape: &[usize], vec: Vec<T>) -> Self {
        assert_eq!(
            compute_n_elems(shape),
            vec.len(),
            "Shape and vector length mismatch"
        );
        Self(ArrCore::new(shape.to_vec(), DynSized::from_vec(vec)))
    }

    pub fn into_vec(self) -> Vec<T> { self.0.data.into_vec() }

    pub fn from_boxed_slice(shape: &[usize], slice: Box<[T]>) -> Self {
        assert_eq!(compute_n_elems(shape), slice.len());
        Self(ArrCore::new(
            shape.to_vec(),
            DynSized::from_boxed_slice(slice),
        ))
    }

    pub fn into_boxed_slice(self) -> Box<[T]> { self.0.data.into_boxed_slice() }

    pub fn from_slice(shape: &[usize], slice: &[T]) -> Self
    where
        T: Clone,
    {
        assert_eq!(compute_n_elems(shape), slice.len());
        Self(ArrCore::new(shape.to_vec(), DynSized::from_slice(slice)))
    }

    pub fn as_slice(&self) -> &[T] { self.0.data.as_slice() }

    pub fn as_mut_slice(&mut self) -> &mut [T] { self.0.data.as_mut_slice() }

    pub fn reshape(&self, shape: &[usize]) -> Self
    where
        T: Clone,
    {
        assert_eq!(compute_n_elems(shape), self.0.data.len());
        Self(ArrCore::new(shape.to_vec(), self.0.data.clone()))
    }

    pub fn zeros(shape: &[usize]) -> Self
    where
        T: Zero + Clone,
    {
        let n_elems = compute_n_elems(shape);
        Self(ArrCore::new(
            shape.to_vec(),
            DynSized::splat(T::zero(), n_elems),
        ))
    }

    pub fn ones(shape: &[usize]) -> Self
    where
        T: One + Clone,
    {
        let n_elems = compute_n_elems(shape);
        Self(ArrCore::new(
            shape.to_vec(),
            DynSized::splat(T::one(), n_elems),
        ))
    }

    pub fn splat(value: T, shape: &[usize]) -> Self
    where
        T: Clone,
    {
        let n_elems = compute_n_elems(shape);
        Self(ArrCore::new(
            shape.to_vec(),
            DynSized::splat(value, n_elems),
        ))
    }

    pub fn iter(&self) -> Iter<T> { self.0.data.as_slice().iter() }

    pub fn iter_mut(&mut self) -> IterMut<T> { self.0.data.as_mut_slice().iter_mut() }

    pub fn as_ptr(&self) -> *const T { self.0.data.as_ptr() }

    pub fn as_mut_ptr(&mut self) -> *mut T { self.0.data.as_mut_ptr() }
}

impl<T, const N: usize, const L: MemLayout> Index<[usize; N]> for DynArr<T, L> {
    type Output = T;

    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        assert!(
            compute_n_elems(&index) <= self.0.data.len(),
            "Index out of bounds"
        );
        &self.0[index]
    }
}

impl<T, const N: usize, const L: MemLayout> IndexMut<[usize; N]> for DynArr<T, L> {
    #[track_caller]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        assert!(
            compute_n_elems(&index) <= self.0.data.len(),
            "Index out of bounds"
        );
        &mut self.0[index]
    }
}

impl<T, const L: MemLayout> Index<usize> for DynArr<T, L> {
    type Output = T;

    #[track_caller]
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}

impl<T, const L: MemLayout> IndexMut<usize> for DynArr<T, L> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
}

impl<T, const L: MemLayout> Clone for DynArr<T, L>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T, const L: MemLayout> Debug for DynArr<T, L>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DynArr({:?})", &self.0.data))
    }
}

impl<T, const L: MemLayout> PartialEq for DynArr<T, L>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T, const L: MemLayout> Eq for DynArr<T, L> where T: Eq {}

impl<T, const L: MemLayout> AsRef<[T]> for DynArr<T, L> {
    fn as_ref(&self) -> &[T] { self.as_slice() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynarr_creation() {
        let arr = DynArr::<f32>::empty(&[2, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
        assert_eq!(arr.order(), MemLayout::RowMajor);
    }
}

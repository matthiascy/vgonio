use crate::array::{core::ArrCore, mem::heap::DynSized, shape::compute_n_elems, MemLayout};
use num_traits::{One, Zero};
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    ops::{Index, IndexMut},
    slice::Iter,
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

    /// Consumes the array and returns the underlying data as a vector.
    pub fn into_vec(self) -> Vec<T> { self.0.data.into_vec() }

    /// Creates a new array from a boxed slice.
    pub fn from_boxed_slice(shape: [usize; N], slice: Box<[T]>) -> Self {
        assert_eq!(compute_n_elems(&shape), slice.len());
        Self(ArrCore::new(shape, DynSized::from_boxed_slice(slice)))
    }

    /// Creates a new array from an iterator.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the array. Only one dimension size can be -1,
    ///   which means that the size of that dimension is inferred.
    /// * `iter` - The iterator to create the array from. The number of elements
    ///  in the iterator must be equal to the number of elements in the shape.
    pub fn from_iterator(shape: [isize; N], iter: impl IntoIterator<Item = T>) -> Self {
        let num_minuses = shape.iter().filter(|&&x| x == -1).count();
        assert!(
            num_minuses <= 1,
            "Only one dimension size can be -1, found {}",
            num_minuses
        );

        if num_minuses == 0 {
            let shape = shape.map(|x| x as usize);
            return Self::from_vec(shape, iter.into_iter().collect());
        }

        let vec: Vec<T> = iter.into_iter().collect();
        let known_size = shape.iter().filter(|&&x| x != -1).product::<isize>() as usize;
        let inferred_size = vec.len() / known_size;
        let mut inferred_shape = [0; N];
        inferred_shape
            .iter_mut()
            .zip(shape.iter())
            .for_each(|(s, &x)| {
                if x == -1 {
                    *s = inferred_size;
                } else {
                    *s = x as usize
                }
            });
        Self::from_vec(inferred_shape, vec)
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

    /// Creates a new array with all elements set to the given value.
    pub fn splat(value: T, shape: [usize; N]) -> Self
    where
        T: Clone,
    {
        let n = compute_n_elems(&shape);
        Self(ArrCore::new(shape, DynSized::splat(value, n)))
    }

    /// Returns an iterator over the array elements.
    pub fn iter(&self) -> Iter<T> { self.0.data.as_slice().iter() }
}

impl<T, const N: usize, const L: MemLayout> DyArr<MaybeUninit<T>, N, L> {
    /// Assumes that the array is fully initialised and returns a new array with
    /// the same shape but with the data assumed to be initialised.
    pub unsafe fn assume_init(self) -> DyArr<T, N, L> {
        let mut shape = [0; N];
        shape.copy_from_slice(self.0.shape());
        let ArrCore { data, .. } = self.0;
        DyArr(ArrCore::new(shape, data.assume_init()))
    }
}

impl<T, const L: MemLayout> DyArr<T, 1, L> {
    /// Creates a new 1D array from a vector.
    pub fn from_vec_1d(vec: Vec<T>) -> Self {
        Self(ArrCore::new([vec.len()], DynSized::from_vec(vec)))
    }

    /// Creates a new 1D array from a boxed slice.
    pub fn from_boxed_slice_1d(slice: Box<[T]>) -> Self {
        Self(ArrCore::new(
            [slice.len()],
            DynSized::from_boxed_slice(slice),
        ))
    }

    /// Creates a new 1D array from a slice.
    pub fn from_slice_1d(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self(ArrCore::new([slice.len()], DynSized::from_slice(slice)))
    }
}

impl<T, const N: usize, const L: MemLayout> Index<[usize; N]> for DyArr<T, N, L> {
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output { &self.0[index] }
}

impl<T, const N: usize, const L: MemLayout> IndexMut<[usize; N]> for DyArr<T, N, L> {
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output { &mut self.0[index] }
}

// impl<T, const L: MemLayout> Index<(usize,)> for DyArr<T, 1, L> {
//     type Output = T;
//
//     #[inline]
//     fn index(&self, index: (usize,)) -> &Self::Output { &self.0[index.0] }
// }
//
// impl<T, const L: MemLayout> IndexMut<(usize,)> for DyArr<T, 1, L> {
//     #[inline]
//     fn index_mut(&mut self, index: (usize,)) -> &mut Self::Output { &mut
// self.0[index.0] } }
//
// impl<T, const L: MemLayout> Index<(usize, usize)> for DyArr<T, 2, L> {
//     type Output = T;
//
//     #[inline]
//     fn index(&self, index: (usize, usize)) -> &Self::Output {
// &self.0[[index.0, index.1]] } }
//
// impl<T, const L: MemLayout> IndexMut<(usize, usize)> for DyArr<T, 2, L> {
//     #[inline]
//     fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
//         &mut self.0[[index.0, index.1]]
//     }
// }

macro_rules! tuple_len {
    () => { 0 };
    ($x:tt $($xs:tt)*) => { 1 + tuple_len!($($xs)*) };
}

macro_rules! impl_tuple_indexing_inner {
    ($($idx:tt, $idx_ty:ty),+) => {
        impl<T, const L: MemLayout> Index<($($idx_ty,)*)> for DyArr<T, {tuple_len!($($idx_ty)*)}, L> {
            type Output = T;

            #[inline]
            fn index(&self, index: ($($idx_ty,)*)) -> &Self::Output {
                &self.0[[$(index.$idx),*]]
            }
        }

        impl<T, const L: MemLayout> IndexMut<($($idx_ty,)*)> for DyArr<T, {tuple_len!($($idx_ty)*)}, L> {
            #[inline]
            fn index_mut(&mut self, index: ($($idx_ty,)*)) -> &mut Self::Output {
                &mut self.0[[$(index.$idx),*]]
            }
        }
    };
}

macro_rules! impl_tuple_indexing {
    ($($idx:tt),+) => {
        impl_tuple_indexing_inner!($($idx, usize),+);
    };
}

impl_tuple_indexing!(0);
impl_tuple_indexing!(0, 1);
impl_tuple_indexing!(0, 1, 2);
impl_tuple_indexing!(0, 1, 2, 3);
impl_tuple_indexing!(0, 1, 2, 3, 4);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_tuple_indexing!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

impl<T, const N: usize, const L: MemLayout> Index<usize> for DyArr<T, N, L> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}

impl<T, const N: usize, const L: MemLayout> IndexMut<usize> for DyArr<T, N, L> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
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

    #[test]
    fn test_dyarr_from_iterator() {
        let arr = DyArr::<i32, 3, { MemLayout::RowMajor }>::from_iterator([2, 3, 2], 0..12);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
        assert_eq!(arr.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

        let arr = DyArr::<i32, 3, { MemLayout::RowMajor }>::from_iterator([-1, 3, 2], 0..12);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
        assert_eq!(arr.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);

        let arr = DyArr::<i32, 1, { MemLayout::RowMajor }>::from_iterator([-1], 0..12);
        assert_eq!(arr.shape(), &[12]);
        assert_eq!(arr.strides(), &[1]);
        assert_eq!(arr.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn test_dyarr_indexing() {
        let arr = DyArr::<i32, 3>::from_iterator([2, 3, 2], 0..12);
        assert_eq!(arr[[0, 0, 0]], 0);
        assert_eq!(arr[[0, 1, 1]], 3);
        assert_eq!(arr[[1, 2, 1]], 11);

        assert_eq!(arr[[0, 0, 0]], arr[(0, 0, 0)]);
        assert_eq!(arr[[0, 1, 1]], arr[(0, 1, 1)]);
        assert_eq!(arr[[1, 2, 1]], arr[(1, 2, 1)]);
    }
}

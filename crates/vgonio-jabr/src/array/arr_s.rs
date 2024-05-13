use crate::{
    array::{
        core::ArrCore,
        mem::{stack::FixedSized, MemLayout, C},
        shape::ConstShape,
    },
    utils::{Assert, IsTrue},
};
use num_traits::{One, Zero};
use std::ops::{Index, IndexMut};

/// A fixed-size multidimensional array on the stack with type level fixed
/// shape (dimensions and size of each dimension).
/// There is no extra metadata associated with the array.
pub struct Array<T, S, const L: MemLayout = C>(
    pub(crate) ArrCore<FixedSized<T, { S::N_ELEMS }>, S, L>,
)
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:;

/// One-dimensional array on the stack.
type Arr<T, const N: usize, const L: MemLayout = C> = Array<T, s![N], L>;

impl<T, S, const L: MemLayout> Array<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    forward_array_core_common_methods!();

    /// Creates a new array with the given data and shape.
    pub fn new(data: [T; S::N_ELEMS]) -> Self { Self(ArrCore::new(S::SHAPE, FixedSized(data))) }

    pub fn reshape<S0>(self) -> Array<T, S0, L>
    where
        S0: ConstShape<Underlying = [usize; S0::N_DIMS]>,
        [(); S0::N_ELEMS]:,
        Assert<{ S::N_ELEMS == S0::N_ELEMS }>: IsTrue,
    {
        todo!("Implement reshape for Arr")
    }

    /// Creates a new array with all elements set to one.
    pub fn ones() -> Self
    where
        T: One + Copy,
    {
        Self(ArrCore::new(S::SHAPE, FixedSized([T::one(); S::N_ELEMS])))
    }

    /// Creates a new array with all elements set to zero.
    pub fn zeros() -> Self
    where
        T: Zero + Copy,
    {
        Self(ArrCore::new(S::SHAPE, FixedSized([T::zero(); S::N_ELEMS])))
    }

    /// Creates a new array with all elements set to the given value.
    pub fn full(value: T) -> Self
    where
        T: Copy,
    {
        Self(ArrCore::new(S::SHAPE, FixedSized([value; S::N_ELEMS])))
    }

    /// Creates a new array with all elements set to the given value.
    pub fn splat(value: T) -> Self
    where
        T: Copy,
    {
        Self(ArrCore::new(S::SHAPE, FixedSized([value; S::N_ELEMS])))
    }
}

impl<T, S, const L: MemLayout> Clone for Array<T, S, L>
where
    T: Clone,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T, S, const L: MemLayout> Copy for Array<T, S, L>
where
    T: Copy,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
}

impl<T, S, const L: MemLayout, const N: usize> Index<[usize; N]> for Array<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output { self.0.index(index) }
}

impl<T, S, const L: MemLayout, const N: usize> IndexMut<[usize; N]> for Array<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output { self.0.index_mut(index) }
}

impl<T, S, const L: MemLayout> Index<usize> for Array<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output { self.0.index(index) }
}

impl<T, S, const L: MemLayout> IndexMut<usize> for Array<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { self.0.index_mut(index) }
}

use crate::array::shape::s;

/// A macro for creating a fixed-size array with the given elements on the
/// stack.
pub macro arr($($n:expr),+ $(,)*; [$($x:expr),* $(,)*]) {{
    crate::array::Array::<_, s![$($n),*], { MemLayout::RowMajor }>::new([$($x),*])
}}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::s;

    #[test]
    fn test_arr_type_size() {
        let arr: Array<i32, s![2, 3], { MemLayout::RowMajor }> = Array::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(std::mem::size_of_val(&arr), 6 * std::mem::size_of::<i32>());
    }

    #[test]
    fn test_arr_1d() {
        let arr = Arr::<i32, 6>::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[6]);
        assert_eq!(arr.strides(), &[1]);
        assert_eq!(arr.order(), MemLayout::RowMajor);
    }

    #[test]
    fn test_arr_creation() {
        let arr: Array<i32, s![2, 3], { MemLayout::RowMajor }> = Array::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[3, 1]);
        assert_eq!(arr.order(), MemLayout::RowMajor);
    }

    #[test]
    fn test_arr_indexing() {
        let arr: Array<i32, s![2, 3], { MemLayout::RowMajor }> = Array::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr[[0, 0]], 1);
        assert_eq!(arr[[0, 1]], 2);
        assert_eq!(arr[[0, 2]], 3);
        assert_eq!(arr[[1, 0]], 4);
        assert_eq!(arr[[1, 1]], 5);
        assert_eq!(arr[[1, 2]], 6);

        for i in 0..6 {
            assert_eq!(arr[i], i as i32 + 1);
        }
    }

    #[test]
    fn test_ones() {
        let arr: Array<i32, s![2, 3, 5], { MemLayout::RowMajor }> = Array::ones();
        assert_eq!(arr.shape(), &[2, 3, 5]);
        assert_eq!(arr.strides(), &[15, 5, 1]);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    assert_eq!(arr[[i, j, k]], 1);
                }
            }
        }
    }

    #[test]
    fn test_arr_macro_dim1() {
        let a = arr!(6; [1, 2, 3, 4, 5, 6]);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.strides(), &[1]);
        let b = arr!(2, 3; [1, 2, 3, 4, 5, 6]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.strides(), &[3, 1]);
        assert_eq!(b[[1, 2]], 6);
    }
}

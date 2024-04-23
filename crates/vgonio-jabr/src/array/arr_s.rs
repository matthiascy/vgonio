use crate::{
    array::{
        core::ArrCore,
        mem::{stack::FixedSized, MemLayout},
        shape::ConstShape,
    },
    utils::{Assert, IsTrue},
};
use std::ops::Index;

/// A fixed-size multidimensional array on the stack with type level fixed
/// shape (dimensions and size of each dimension).
/// There is no extra metadata associated with the array.
pub struct Arr<T, S, const L: MemLayout = { MemLayout::RowMajor }>(
    pub(crate) ArrCore<FixedSized<T, { S::N_ELEMS }>, S, L>,
)
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:;

impl<T, S, const L: MemLayout> Arr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    super::forward_core_array_methods!(@const
        shape -> &[usize], #[doc = "Returns the shape of the array."];
        strides -> &[usize], #[doc = "Returns the strides of the array."];
        order -> MemLayout, #[doc = "Returns the layout of the array."];
        dimension -> usize, #[doc = "Returns the number of dimensions of the array."];
        len -> usize, #[doc = "Returns the total number of elements in the array."];
    );

    /// Creates a new array with the given data and shape.
    pub fn new(data: [T; S::N_ELEMS]) -> Self { Self(ArrCore::new(S::SHAPE, FixedSized(data))) }

    pub fn reshape<S0>(self) -> Arr<T, S0, L>
    where
        S0: ConstShape<Underlying = [usize; S0::N_DIMS]>,
        [(); S0::N_ELEMS]:,
        Assert<{ S::N_ELEMS == S0::N_ELEMS }>: IsTrue,
    {
        todo!("Implement reshape for Arr")
    }
}

impl<T, S, const L: MemLayout> Clone for Arr<T, S, L>
where
    T: Clone,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T, S, const L: MemLayout> Copy for Arr<T, S, L>
where
    T: Copy,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
}

impl<T, S, const L: MemLayout, const N: usize> Index<[usize; N]> for Arr<T, S, L>
where
    T: Copy,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output { &self.0[index] }
}

use crate::array::shape::s;

/// A macro for creating a fixed-size array with the given elements on the
/// stack.
pub macro arr([$($n:expr),+ $(,)*]; [$($x:expr),* $(,)*]) {{
    crate::array::Arr::<_, s![$($n),*], { MemLayout::RowMajor }>::new([$($x),*])
}}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::s;

    #[test]
    fn test_arr_creation() {
        let arr: Arr<i32, s![2, 3], { MemLayout::RowMajor }> = Arr::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[3, 1]);
        assert_eq!(arr.order(), MemLayout::RowMajor);
    }

    #[test]
    fn test_arr_macro_dim1() {
        let a = arr!([6]; [1, 2, 3, 4, 5, 6]);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.strides(), &[1]);
        let b = arr!([2, 3]; [1, 2, 3, 4, 5, 6]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.strides(), &[3, 1]);
        assert_eq!(b[[1, 2]], 6);
    }

    #[test]
    fn test_deref() {
        let arr: Arr<i32, s![2, 3], { MemLayout::ColMajor }> = Arr::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[1, 2]);
    }
}

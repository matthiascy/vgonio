use crate::array::{
    core::ArrCore,
    forward_core_array_methods,
    mem::{stack::FixedSized, MemLayout},
    shape::ConstShape,
};
use std::ops::Deref;

/// A fixed-size multidimensional array on the stack with type level fixed
/// shape (dimensions and size of each dimension).
/// There is no extra metadata associated with the array.
pub struct Arr<T, S, const L: MemLayout = { MemLayout::ColMajor }>(
    ArrCore<FixedSized<T, { S::N_ELEMS }>, S, L>,
)
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:;

impl<T, S, const L: MemLayout> Arr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    super::forward_const_core_array_methods!();

    /// Creates a new array with the given data and shape.
    pub fn new(data: [T; S::N_ELEMS]) -> Self { Self(ArrCore::new(S::SHAPE, FixedSized(data))) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::s;

    #[test]
    fn test_arr_creation() {
        let arr: Arr<i32, s![2, 3], { MemLayout::ColMajor }> = Arr::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[1, 2]);
        assert_eq!(arr.order(), MemLayout::ColMajor);
    }

    #[test]
    fn test_deref() {
        let arr: Arr<i32, s![2, 3], { MemLayout::ColMajor }> = Arr::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[1, 2]);
    }
}

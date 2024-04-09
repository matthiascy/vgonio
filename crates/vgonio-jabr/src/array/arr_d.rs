use crate::array::{
    core::ArrCore,
    mem::{heap::DynFixSized, MemLayout},
    shape::ConstShape,
};

/// A fixed-size multidimensional array on the heap with type level fixed shape
/// (dimensions and size of each dimension).
///
/// The number of dimensions and the size of each dimension are set at
/// compilation time and cannot be changed.
pub struct DArr<T, S, const L: MemLayout = { MemLayout::ColMajor }>(
    pub(crate) ArrCore<DynFixSized<T, { S::N_ELEMS }>, S, L>,
)
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:;

impl<T, S, const L: MemLayout> DArr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    super::forward_core_array_methods!(@const
        shape -> &[usize], #[doc = "Returns the shape of the array."];
        strides -> &[usize], #[doc = "Returns the strides of the array."];
        order -> MemLayout, #[doc = "Returns the layout of the array."];
        dimension -> usize, #[doc = "Returns the number of dimensions of the array."];
    );

    /// Creates a new array with the given data and shape.
    pub fn new(data: [T; S::N_ELEMS]) -> Self
    where
        T: Clone,
    {
        Self(ArrCore::new(S::SHAPE, DynFixSized::from_slice(&data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::s;

    #[test]
    fn test_darr_creation() {
        let arr: DArr<i32, s![2, 3, 2, 2], { MemLayout::RowMajor }> = DArr::new([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
        assert_eq!(arr.shape(), &[2, 3, 2, 2]);
        assert_eq!(arr.strides(), &[12, 4, 2, 1]);
    }
}

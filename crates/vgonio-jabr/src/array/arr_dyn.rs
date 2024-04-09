use crate::array::{
    core::ArrCore,
    mem::{heap::DynSized, MemLayout},
};

/// A dynamically-sized array with dynamical number of dimensions
/// and size of each dimension at runtime (growable).
pub struct DynArr<T, const L: MemLayout = { MemLayout::ColMajor }>(
    pub(crate) ArrCore<DynSized<T>, Vec<usize>, L>,
);

impl<T, const L: MemLayout> DynArr<T, L> {
    super::forward_core_array_methods!(@const
        shape -> &[usize], #[doc = "Returns the shape of the array."];
        strides -> &[usize], #[doc = "Returns the strides of the array."];
        order -> MemLayout, #[doc = "Returns the layout of the array."];
        dimension -> usize, #[doc = "Returns the number of dimensions of the array."];
    );

    /// Creates a new array with the given data and shape.
    pub fn new(shape: &[usize]) -> Self {
        let n_elems = shape.iter().product();
        let shape = shape.to_vec();
        Self(ArrCore::new(shape, DynSized::with_capacity(n_elems)))
    }

    // pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> DyArr<T, M, L> {
    // todo!() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynarr_creation() {
        let arr: DynArr<i32, { MemLayout::RowMajor }> = DynArr::new(&[2, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
    }
}

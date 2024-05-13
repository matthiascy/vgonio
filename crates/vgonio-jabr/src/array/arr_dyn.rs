use crate::array::{
    core::ArrCore,
    mem::{heap::DynSized, MemLayout},
};
use std::mem::MaybeUninit;

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

    // pub fn reshape<const M: usize>(&self, shape: [usize; M]) -> DyArr<T, M, L> {
    // todo!() }
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

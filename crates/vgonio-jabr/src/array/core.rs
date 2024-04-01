use crate::array::{
    dim::DimSeq,
    mem::{Data, MemLayout, MemLayout::ColMajor},
    shape::{compute_n_elems, ConstShape, Shape},
};

/// Base struct for all arrays.
pub struct ArrCore<D, S, const L: MemLayout = ColMajor>
where
    D: Data,
    S: Shape,
{
    /// The data of the array.
    data: D,
    /// The extra shape info of the array.
    meta: S::Metadata,
    marker: core::marker::PhantomData<(D, S)>,
}

impl<D, S, const L: MemLayout> ArrCore<D, S, L>
where
    D: Data,
    S: Shape,
{
    /// Creates a new array with the given data, shape, strides, and layout.
    ///
    /// Rarely used directly. Only used inside the crate.
    pub fn new(shape: S::Underlying, data: D) -> Self {
        // Make sure the data is the right size.
        debug_assert_eq!(
            data.as_slice().len(),
            compute_n_elems(shape.as_slice()),
            "data size doesn't match shape"
        );
        Self {
            data,
            meta: S::new_metadata(&shape, L),
            marker: core::marker::PhantomData,
        }
    }
}

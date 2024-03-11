use crate::array::{
    mem::{Data, MemLayout},
    shape::Shape,
};

pub struct ArrCore<D, S, const L: MemLayout>
where
    D: Data,
    S: Shape,
{
    /// The data of the array.
    data: D,
    /// The shape of the array including the number of dimensions and the size
    /// of each dimension.
    shape: S::Underlying,
    /// The number of elements needed to skip to get to the next element along
    /// each dimension. Its interpretation depends on the layout of the array.
    strides: S::Underlying,
    marker: core::marker::PhantomData<(D, S)>,
}

impl<D, S, const L: MemLayout> ArrCore<D, S, L> {
    /// Creates a new array with the given data, shape, strides, and layout.
    ///
    /// Rarely used directly. Only used inside the crate.
    pub fn new(shape: S::Underlying, data: D) -> Self {
        // Make sure the data is the right size.
        debug_assert_eq!(
            data.as_slice().len(),
            compute_num_elems(shape.as_slice()),
            "data size doesn't match shape"
        );

        let mut strides = shape.clone();
        compute_strides(shape.as_slice(), strides.as_slice_mut(), layout);
        Self {
            data,
            shape,
            strides,
            marker: core::marker::PhantomData,
        }
    }
}

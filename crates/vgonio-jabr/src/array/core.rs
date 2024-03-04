use crate::array::{mem::MemLayout, shape::Shape};

pub struct ArrCore<D, S, const L: MemLayout>
where
    D: DataRaw,
    S: Shape,
{
    /// The data of the array.
    data: D,

    /// The shape of the array including the number of dimensions and the size
    /// of each dimension.
    shape: S::UnderlyingType,

    /// The number of elements needed to skip to get to the next element along
    /// each dimension. Its interpretation depends on the layout of the array.
    strides: S::UnderlyingType,

    marker: core::marker::PhantomData<(D, S)>,
}

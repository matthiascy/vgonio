use crate::array::{
    core::ArrCore,
    mem::{
        heap::{DynFixSized, DynSized},
        stack::FixedSized,
        MemLayout,
        MemLayout::ColMajor,
    },
    shape::ConstShape,
};

mod core;
mod dim;
mod mem;
mod shape;

/// A fixed-size multidimensional array on the stack with type level fixed
/// shape (dimensions and size of each dimension).
/// There is no extra metadata associated with the array.
pub struct Arr<A, S, const L: MemLayout = ColMajor>
where
    S: ConstShape,
    [(); S::N_ELEMS]:,
{
    inner: ArrCore<FixedSized<A, { S::N_ELEMS }>, S, L>,
}

/// A fixed-size multidimensional array on the heap with type level fixed shape
/// (dimensions and size of each dimension).
///
/// The number of dimensions and the size of each dimension are set at
/// compilation time and cannot be changed.
pub struct DArr<A, S, const L: MemLayout = ColMajor>
where
    S: ConstShape,
    [(); S::N_ELEMS]:,
{
    inner: ArrCore<DynFixSized<A, { S::N_ELEMS }>, S, L>,
}

/// A dynamically-sized array with known number of dimensions at compile-time.
///
/// The dimension is set at compilation time and cannot be changed, but the size
/// of each dimension can be changed at runtime.
pub struct DyArr<A, const N: usize, const L: MemLayout = ColMajor> {
    inner: ArrCore<DynSized<A>, [usize; N], L>,
}

/// A dynamically-sized array with dynamical number of dimensions
/// and size of each dimension at runtime.
pub struct DynArr<A, const L: MemLayout = ColMajor> {
    inner: ArrCore<DynSized<A>, Vec<usize>, L>,
}

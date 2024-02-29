mod shape;

/// Memory layout of a multidimensional array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemLayout {
    /// Row-major layout (or C layout). The data is stored row by row in memory;
    /// the strides grow from right to left; the last dimension varies the
    /// fastest.
    RowMajor,
    /// Column-major layout (or Fortran layout). The data is stored column by
    /// column in memory; the strides grow from left to right; the first
    /// dimension varies the fastest.
    ColMajor,
}

// /// Base type for a multidimensional array.
// pub struct Array<D, S, const L: MemLayout> {
//     data: D,
//     shape: S::UnderlyingType,
//     strides: S::UnderlyingType,
//     marker: core::marker::PhantomData<(D, S)>,
// }

// /// A fixed-size array with `N` elements of type `A` on the stack.
// ///
// /// Can be nested to create multidimensional arrays.
// pub struct SArr<A, const N: usize> {
//     data: [A; N],
// }
//
// /// A fixed-size array with elements of type `A` on the stack.
// pub struct DArr<A, S, L> {}
//
// /// A dynamically-sized array with elements of type `A` on the heap.
// pub struct DynArr<A, L> {}

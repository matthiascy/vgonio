mod core;
mod dim;
mod mem;
mod shape;

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

pub mod heap;
pub mod stack;

use core::{fmt::Write, marker::ConstParamTy};

/// Memory layout of a multidimensional array.
///
/// The memory layout determines how the elements of the array are stored in
/// memory. The elements of the array are stored in a contiguous block of
/// memory, and the memory layout determines how the elements are ordered in
/// this block.
#[derive(ConstParamTy, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Trait providing raw access to the elements of the storage.
pub unsafe trait Data: Sized {
    /// The type of the elements stored in the array.
    type Elem;
    type Uninit;

    /// Returns a pointer to the first element of the array.
    fn as_ptr(&self) -> *const Self::Elem;

    /// Returns a mutable pointer to the first element of the array.
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;

    /// Returns a slice of the data.
    fn as_slice(&self) -> &[Self::Elem];

    /// Returns a mutable slice of the data.
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];

    fn uninit(size: usize) -> Self::Uninit;
}

pub trait DataClone: Data + Clone {}

impl<A: Clone> DataClone for A where A: Data {}

pub trait DataCopy: DataClone + Copy {}

impl<A: Copy> DataCopy for A where A: DataClone {}

pub(crate) fn print_slice<A>(f: &mut ::core::fmt::Formatter<'_>, seq: &[A]) -> ::core::fmt::Result
where
    A: ::core::fmt::Display,
{
    f.write_char('[')?;
    for (i, x) in seq.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", x)?;
    }
    f.write_char(']')
}

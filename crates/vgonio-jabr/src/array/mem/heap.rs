use crate::array::mem::{self, Data};
use std::{
    alloc::Allocator,
    fmt::{Debug, Display},
    mem::{ManuallyDrop, MaybeUninit},
    ops::Deref,
    ptr::NonNull,
};

// TODO: rewrite avoiding the use of `Vec` to enable slicing.
/// A fixed-size array allocated on the heap, with a dynamically determined
/// size.
#[repr(C)]
#[derive(Clone)]
pub struct DynSized<T>(Vec<T>);

impl<T> DynSized<T> {
    pub(crate) fn new() -> Self { Self(Vec::new()) }

    /// Creates a new `DynSized` from a vector without copying the data.
    pub(crate) fn from_vec(vec: Vec<T>) -> Self {
        let mut v = vec;
        // Make sure there is no extra memory allocated.
        v.shrink_to_fit();
        Self(v)
    }

    pub(crate) fn from_boxed_slice(slice: Box<[T]>) -> Self { Self(slice.into_vec()) }

    pub(crate) fn as_slice(&self) -> &[T] { self.0.as_slice() }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] { self.0.as_mut_slice() }

    pub(crate) fn as_ptr(&self) -> *const T { self.0.as_ptr() }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T { self.0.as_mut_ptr() }
}

impl<T> PartialEq for DynSized<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .all(|(a, b)| a == b)
    }
}

impl<T: Eq> Eq for DynSized<T> {}

unsafe impl<T> Data for DynSized<T> {
    type Elem = T;
    type Uninit = DynSized<MaybeUninit<T>>;

    fn as_ptr(&self) -> *const T { self.0.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.0.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.as_slice() }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { self.as_mut_slice() }

    fn uninit(size: usize) -> Self::Uninit { DynSized::new_uninit(size) }
}

impl<T> Debug for DynSized<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DynSized({:?})", &self.as_slice()))
    }
}

impl<T> Display for DynSized<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        mem::print_slice(f, &self.as_slice())
    }
}

impl<T> DynSized<T> {
    pub fn new_uninit(len: usize) -> DynSized<MaybeUninit<T>> {
        DynSized::from_boxed_slice(Box::new_uninit_slice(len))
    }

    pub fn splat(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        Self::from_vec(vec![value; len])
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self::from_boxed_slice(slice.to_vec().into_boxed_slice())
    }
}

impl<T> DynSized<MaybeUninit<T>> {
    pub unsafe fn assume_init(self) -> DynSized<T> {
        let mut vec = self.0;
        let ptr = vec.as_mut_ptr() as *mut T;
        let len = vec.len();
        std::mem::forget(vec);
        DynSized(Vec::from_raw_parts(ptr, len, len))
    }
}

impl<T, const N: usize> From<[T; N]> for DynSized<T>
where
    T: Clone,
{
    fn from(array: [T; N]) -> Self { Self::from_slice(&array) }
}

impl<T> From<Vec<T>> for DynSized<T> {
    fn from(vec: Vec<T>) -> Self { Self::from_vec(vec) }
}

unsafe impl<T: Send> Send for DynSized<T> {}
unsafe impl<T: Sync> Sync for DynSized<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyn_sized_display() {
        let a = DynSized::from(vec![1, 2, 3]);
        assert_eq!(format!("{:?}", a), "DynSized([1, 2, 3])");
        assert_eq!(format!("{}", a), "[1, 2, 3]");
        let b = DynSized::from_slice(&[1, 2, 3]);
        assert_eq!(format!("{:?}", b), "DynSized([1, 2, 3])");
    }

    #[test]
    fn test_fixed_sized() {
        let array: DynSized<u32> = DynSized::from([1, 2, 3]);
        assert_eq!(array.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn fixed_sized_display() {
        let a = DynSized::from([1u32, 2, 3]);
        assert_eq!(format!("{:?}", a), "FixSized([1, 2, 3])");
        assert_eq!(format!("{}", a), "[1, 2, 3]");
    }
}

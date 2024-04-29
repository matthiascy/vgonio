use crate::array::mem::{self, Data};
use std::{
    alloc::{Allocator, Global},
    fmt::{Debug, Display},
    mem::{ManuallyDrop, MaybeUninit},
    ops::Deref,
    ptr::NonNull,
};

/// A fixed-size array allocated on the heap, with a dynamically determined
/// size.
#[repr(C)]
pub struct DynSized<T> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T> DynSized<T> {
    pub(crate) fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
        }
    }

    pub(crate) fn len(&self) -> usize { self.len }

    /// Creates a new `DynSized` from a vector without copying the data.
    pub(crate) fn from_vec(vec: Vec<T>) -> Self {
        let mut vec = {
            let mut v = vec;
            v.shrink_to_fit();
            ManuallyDrop::new(v)
        };
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        Self {
            ptr: NonNull::new(ptr).unwrap(),
            len,
        }
    }

    pub(crate) fn from_boxed_slice(slice: Box<[T]>) -> Self {
        let ptr = slice.as_ptr();
        let len = slice.len();
        std::mem::forget(slice);
        Self {
            ptr: NonNull::new(ptr as *mut T).unwrap(),
            len,
        }
    }

    pub(crate) fn into_vec(self) -> Vec<T> {
        let ptr = self.ptr.as_ptr();
        let len = self.len;
        let cap = self.len;
        unsafe {
            let vec = Vec::from_raw_parts(ptr, len, cap);
            vec
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_ptr(&self) -> *const T { self.ptr.as_ptr() }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T { self.ptr.as_ptr() }
}

impl<T> Clone for DynSized<T>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self::from_vec(self.as_slice().to_owned()) }

    fn clone_from(&mut self, source: &Self) {
        let mut v = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.len) };
        if v.len() > source.len {
            v.truncate(source.len);
        }
        let other = source.as_slice();
        let (front, back) = other.split_at(v.len());
        v.clone_from_slice(front);
        v.extend_from_slice(back);
        *self = Self::from_vec(v);
    }
}

impl<T> Drop for DynSized<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let ptr = self.ptr.as_ptr();
            let len = self.len;
            let cap = self.len;
            unsafe {
                Vec::from_raw_parts(ptr, len, cap);
            }
        }
    }
}

impl<T> PartialEq for DynSized<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len() {
            return false;
        }
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

    fn as_ptr(&self) -> *const T { self.ptr.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.ptr.as_ptr() }

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
        let mut slice = Box::new_uninit_slice(len);
        DynSized {
            ptr: NonNull::new(slice.as_mut_ptr()).unwrap(),
            len,
        }
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
        let ptr = self.ptr.cast::<T>();
        let len = self.len;
        std::mem::forget(self);
        DynSized { ptr, len }
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

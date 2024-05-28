use crate::array::mem::{self, Data};
use std::{
    fmt::{Debug, Display},
    mem::{ManuallyDrop, MaybeUninit},
    ptr::NonNull,
};

/// A block of memory allocated on the heap, with a fixed size.
#[repr(C)]
pub struct DynSized<T> {
    data: NonNull<T>,
    len: usize,
}

impl<T> DynSized<T> {
    pub(crate) fn new_uninit(len: usize) -> DynSized<MaybeUninit<T>> {
        let mut data = ManuallyDrop::new(Box::new_uninit_slice(len));
        let data = unsafe { NonNull::new_unchecked(data.as_mut_ptr()) };
        DynSized { data, len }
    }

    pub(crate) fn splat(val: T, len: usize) -> Self
    where
        T: Clone,
    {
        let mut data = {
            let mut data = Box::new_uninit_slice(len);
            for elem in data.iter_mut() {
                elem.write(val.clone());
            }
            ManuallyDrop::new(unsafe { data.assume_init() })
        };
        Self {
            data: unsafe { NonNull::new_unchecked(data.as_mut_ptr()) },
            len,
        }
    }

    pub(crate) fn from_vec(vec: Vec<T>) -> Self {
        let mut data = ManuallyDrop::new(vec.into_boxed_slice());
        let len = data.len();
        Self {
            data: unsafe { NonNull::new_unchecked(data.as_mut_ptr()) },
            len,
        }
    }

    pub(crate) fn into_vec(self) -> Vec<T> {
        unsafe {
            let mut data = ManuallyDrop::new(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.data.as_ptr(),
                self.len,
            )));
            Vec::from_raw_parts(data.as_mut_ptr(), self.len, self.len)
        }
    }

    pub(crate) fn from_boxed_slice(slice: Box<[T]>) -> Self {
        let mut data = ManuallyDrop::new(slice);
        let len = data.len();
        Self {
            data: unsafe { NonNull::new_unchecked(data.as_mut_ptr()) },
            len,
        }
    }

    pub(crate) fn into_boxed_slice(self) -> Box<[T]> {
        unsafe {
            let mut data = ManuallyDrop::new(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.data.as_ptr(),
                self.len,
            )));
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                data.as_mut_ptr(),
                self.len,
            ))
        }
    }

    pub(crate) fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        let mut data = {
            let mut data = Box::new_uninit_slice(slice.len());
            for (elem, val) in data.iter_mut().zip(slice.iter()) {
                elem.write(val.clone());
            }
            ManuallyDrop::new(unsafe { data.assume_init() })
        };
        Self {
            data: unsafe { NonNull::new_unchecked(data.as_mut_ptr()) },
            len: slice.len(),
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    pub(crate) fn as_ptr(&self) -> *const T { self.data.as_ptr() }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T { self.data.as_ptr() }

    pub(crate) fn len(&self) -> usize { self.len }
}

impl<T> DynSized<MaybeUninit<T>> {
    pub unsafe fn assume_init(self) -> DynSized<T> {
        let len = self.len;
        let myself = ManuallyDrop::new(self);
        DynSized {
            data: NonNull::new_unchecked(myself.data.as_ptr() as *mut T),
            len,
        }
    }
}

impl<T> Clone for DynSized<T>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self::from_vec(self.as_slice().to_vec()) }

    fn clone_from(&mut self, source: &Self) {
        if self.len == source.len {
            self.as_mut_slice().clone_from_slice(source.as_slice());
        } else {
            *self = source.clone();
        }
    }
}

impl<T> Drop for DynSized<T> {
    fn drop(&mut self) {
        let _ = unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.data.as_ptr(),
                self.len,
            ))
        };
    }
}

unsafe impl<T> Sync for DynSized<T> where T: Sync {}
unsafe impl<T> Send for DynSized<T> where T: Send {}

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

impl<T, const N: usize> From<[T; N]> for DynSized<T>
where
    T: Clone,
{
    fn from(array: [T; N]) -> Self { Self::from_slice(&array) }
}

impl<T> From<Vec<T>> for DynSized<T> {
    fn from(vec: Vec<T>) -> Self { Self::from_vec(vec) }
}

impl<T> From<Box<[T]>> for DynSized<T> {
    fn from(boxed: Box<[T]>) -> Self { Self::from_boxed_slice(boxed) }
}

impl<T> From<&[T]> for DynSized<T>
where
    T: Clone,
{
    fn from(slice: &[T]) -> Self { Self::from_slice(slice) }
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

    fn as_ptr(&self) -> *const T { self.data.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.data.as_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.as_slice() }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { self.as_mut_slice() }

    fn uninit(size: usize) -> Self::Uninit { DynSized::new_uninit(size) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyn_sized_display() {
        let a = DynSized::from(vec![1, 2, 3]);
        assert_eq!(format!("{:?}", a), "DynSized([1, 2, 3])");
        assert_eq!(format!("{}", a), "[1, 2, 3]");
        let b = DynSized::from([1, 2, 3].as_slice());
        assert_eq!(format!("{:?}", b), "DynSized([1, 2, 3])");
        let c = DynSized::from([1, 2, 3]);
        assert_eq!(format!("{:?}", c), "DynSized([1, 2, 3])");
    }
}

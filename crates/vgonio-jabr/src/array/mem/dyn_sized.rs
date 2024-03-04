use crate::array::mem::{Data, DataRaw, Sealed};
use std::fmt::Debug;

pub struct DynSized<T>(pub(crate) Vec<T>);

impl<T> Sealed for DynSized<T> {}
impl<'a, T> Sealed for &'a DynSized<T> {}
impl<'a, T> Sealed for &'a mut DynSized<T> {}

impl<T: Clone> Clone for DynSized<T> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T: PartialEq> PartialEq for DynSized<T> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T: Eq> Eq for DynSized<T> {}

unsafe impl<T> Data for DynSized<T> {
    type Elem = T;

    fn as_ptr(&self) -> *const T { self.0.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.0.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.0 }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.0 }

    unsafe fn alloc_uninit(n: usize) -> Self { Self(Vec::with_capacity(n)) }
}

impl<T: Debug> Debug for DynSized<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

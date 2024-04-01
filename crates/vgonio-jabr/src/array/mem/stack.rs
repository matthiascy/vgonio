use crate::array::mem::{Data, Sealed};
use std::ops::Deref;

/// A fixed-sized array that is allocated on the stack.
pub struct FixedSized<T, const N: usize> {
    inner: [T; N],
}

impl<T, const N: usize> Sealed for FixedSized<T, N> {}

unsafe impl<T, const N: usize> Data for FixedSized<T, N> {
    type Elem = T;

    fn as_ptr(&self) -> *const Self::Elem { self.inner.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.inner.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.inner }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.inner }
}

impl<T, const N: usize> Clone for FixedSized<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T, const N: usize> Copy for FixedSized<T, N> where T: Copy {}

impl<T, const N: usize> PartialEq for FixedSized<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool { self.inner.eq(&other.inner) }
}

impl<T, const N: usize> Deref for FixedSized<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target { &self.inner }
}

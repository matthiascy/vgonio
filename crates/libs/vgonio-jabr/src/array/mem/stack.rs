use crate::array::mem::Data;
use std::{mem::MaybeUninit, ops::Deref};

/// A fixed-sized array that is allocated on the stack.
pub struct FixedSized<T, const N: usize>(pub(crate) [T; N]);

unsafe impl<T, const N: usize> Data for FixedSized<T, N> {
    type Elem = T;
    type Uninit = FixedSized<MaybeUninit<T>, N>;

    fn as_ptr(&self) -> *const Self::Elem { self.0.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.0.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.0 }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.0 }

    fn uninit(_: usize) -> Self::Uninit {
        FixedSized(unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() })
    }
}

impl<T, const N: usize> Clone for FixedSized<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T, const N: usize> Copy for FixedSized<T, N> where T: Copy {}

impl<T, const N: usize> PartialEq for FixedSized<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool { self.0.eq(&other.0) }
}

impl<T, const N: usize> Deref for FixedSized<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target { &self.0 }
}

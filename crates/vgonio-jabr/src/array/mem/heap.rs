use crate::array::mem::{self, Data, Sealed};
use std::{
    alloc::{Allocator, Global},
    fmt::{Debug, Display},
    mem::MaybeUninit,
    ops::Deref,
};

// TODO: avoid using Vec here as it will over allocate memory
/// Dynamic-sized array on the heap.
pub struct DynSized<T, A = Global>
where
    A: Allocator,
{
    inner: Vec<T, A>,
}

impl<T, A> DynSized<T, A>
where
    A: Allocator,
{
    /// Creates a new empty array with a specific allocator.
    pub fn new_in(alloc: A) -> Self {
        Self {
            inner: Vec::new_in(alloc),
        }
    }

    /// Creates a new empty array with the global allocator.
    pub fn from_vec(vec: Vec<T, A>) -> Self { Self { inner: vec } }

    /// Creates a new empty array with the global allocator.
    pub fn from_slice_in(slice: &[T], alloc: A) -> Self
    where
        T: Clone,
    {
        Self {
            inner: slice.to_vec_in(alloc),
        }
    }

    pub fn with_capacity_in(cap: usize, alloc: A) -> Self {
        Self {
            inner: Vec::with_capacity_in(cap, alloc),
        }
    }
}

impl<T> DynSized<T, Global> {
    /// Creates a new empty array with the global allocator.
    pub fn new() -> Self { Self { inner: Vec::new() } }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap),
        }
    }

    /// Creates a new empty array with the global allocator.
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        Self {
            inner: slice.to_vec(),
        }
    }
}

impl<T, A: Allocator> Sealed for DynSized<T, A> {}
impl<'a, T, A: Allocator> Sealed for &'a DynSized<T, A> {}
impl<'a, T, A: Allocator> Sealed for &'a mut DynSized<T, A> {}

impl<T, A> Clone for DynSized<T, A>
where
    T: Clone,
    A: Allocator + Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: PartialEq, A: Allocator> PartialEq for DynSized<T, A> {
    fn eq(&self, other: &Self) -> bool { self.inner == other.inner }
}

impl<T: Eq, A: Allocator> Eq for DynSized<T, A> {}

unsafe impl<T, A: Allocator> Data for DynSized<T, A> {
    type Elem = T;

    fn as_ptr(&self) -> *const T { self.inner.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.inner.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.inner }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.inner }

    // unsafe fn alloc_uninit(n: usize) -> Self { Self(Vec::with_capacity(n)) }
}

impl<T: Debug, A: Allocator> Debug for DynSized<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DynSized({:?})", &self.inner))
    }
}

impl<T: Display, A: Allocator> Display for DynSized<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        mem::print_slice(f, &self.inner)
    }
}

impl<T, A: Allocator> Deref for DynSized<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl<T, A: Allocator> From<Vec<T, A>> for DynSized<T, A> {
    fn from(vec: Vec<T, A>) -> Self { Self { inner: vec } }
}

/// A fixed-size array allocated on the heap.
#[repr(transparent)]
pub struct DynFixSized<T, const N: usize, A = Global>
where
    A: Allocator,
{
    inner: Box<[T], A>,
    marker: std::marker::PhantomData<[T; N]>,
}

impl<T, const N: usize, A: Allocator> Sealed for DynFixSized<T, N, A> {}
impl<'a, T, const N: usize, A: Allocator> Sealed for &'a DynFixSized<T, N, A> {}
impl<'a, T, const N: usize, A: Allocator> Sealed for &'a mut DynFixSized<T, N, A> {}

impl<T, const N: usize, A> Clone for DynFixSized<T, N, A>
where
    T: Clone,
    A: Allocator + Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, const N: usize, A> PartialEq for DynFixSized<T, N, A>
where
    A: Allocator,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        let mut i = 0;
        while i < N {
            if self.inner[i].eq(&other.inner[i]) {
                return false;
            }
            i += 1;
        }
        true
    }
}

impl<T: Eq, const N: usize, A: Allocator> Eq for DynFixSized<T, N, A> {}

unsafe impl<T, const N: usize, A> Data for DynFixSized<T, N, A>
where
    A: Allocator,
{
    type Elem = T;

    fn as_ptr(&self) -> *const T { self.inner.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.inner.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.inner }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.inner }

    // unsafe fn alloc_uninit(n: usize) -> Self {
    // std::mem::MaybeUninit::uninit().assume_init() }
}

impl<T, const N: usize, A> Debug for DynFixSized<T, N, A>
where
    T: Debug,
    A: Allocator,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("FixSized({:?})", &self.inner))
    }
}

impl<T, const N: usize, A> Display for DynFixSized<T, N, A>
where
    T: Display,
    A: Allocator,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        mem::print_slice(f, &self.inner)
    }
}

impl<T, const N: usize, A> Deref for DynFixSized<T, N, A>
where
    A: Allocator,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl<T, const N: usize> DynFixSized<T, N, Global> {
    pub fn new_uninit() -> DynFixSized<MaybeUninit<T>, N> {
        DynFixSized {
            inner: Box::new_uninit_slice(N),
            marker: std::marker::PhantomData,
        }
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        assert!(
            slice.len() >= N,
            "slice length must greater than array size"
        );
        Self {
            inner: slice.to_vec().into_boxed_slice(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, const N: usize, A> DynFixSized<T, N, A>
where
    A: Allocator,
{
    pub fn new_uninit_in(alloc: A) -> DynFixSized<MaybeUninit<T>, N, A> {
        DynFixSized {
            inner: Box::new_uninit_slice_in(N, alloc),
            marker: std::marker::PhantomData,
        }
    }

    pub fn from_slice_in(slice: &[T], alloc: A) -> Self
    where
        T: Clone,
    {
        assert!(slice.len() > N, "slice length must greater than array size");
        Self {
            inner: slice.to_vec_in(alloc).into_boxed_slice(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, const N: usize, A> DynFixSized<MaybeUninit<T>, N, A>
where
    A: Allocator,
{
    pub unsafe fn assume_init(self) -> DynFixSized<T, N, A> {
        let inner = self.inner.assume_init();
        DynFixSized {
            inner,
            marker: std::marker::PhantomData,
        }
    }
}

impl<T, const N: usize> From<[T; N]> for DynFixSized<T, N> {
    fn from(array: [T; N]) -> Self {
        Self {
            inner: array.into(),
            marker: std::marker::PhantomData,
        }
    }
}

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
        let array: DynFixSized<u32, 3> = DynFixSized::from([1, 2, 3]);
        assert_eq!(array.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn fixed_sized_display() {
        let a = DynFixSized::from([1u32, 2, 3]);
        assert_eq!(format!("{:?}", a), "FixSized([1, 2, 3])");
        assert_eq!(format!("{}", a), "[1, 2, 3]");
    }
}

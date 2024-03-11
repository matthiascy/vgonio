use crate::array::mem::{print_slice, Data, Sealed};
use std::{
    fmt::{Debug, Display},
    ops::Deref,
};

pub struct DynSized<A>(pub(crate) Vec<A>);

impl<A> Sealed for DynSized<A> {}
impl<'a, A> Sealed for &'a DynSized<A> {}
impl<'a, A> Sealed for &'a mut DynSized<A> {}

impl<A: Clone> Clone for DynSized<A> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<A: PartialEq> PartialEq for DynSized<A> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<A: Eq> Eq for DynSized<A> {}

unsafe impl<A> Data for DynSized<A> {
    type Elem = A;

    fn as_ptr(&self) -> *const A { self.0.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.0.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.0 }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.0 }

    unsafe fn alloc_uninit(n: usize) -> Self { Self(Vec::with_capacity(n)) }
}

impl<A: Debug> Debug for DynSized<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DynSized({:?})", &self.0))
    }
}

impl<A: Display> Display for DynSized<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { print_slice(f, &self.0) }
}

impl<A> Deref for DynSized<A> {
    type Target = [A];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<A> From<&[A]> for DynSized<A>
where
    A: Clone,
{
    fn from(slice: &[A]) -> Self { Self(slice.to_vec()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyn_sized_display() {
        let a = DynSized(vec![1, 2, 3]);
        assert_eq!(format!("{:?}", a), "DynSized([1, 2, 3])");
        assert_eq!(format!("{}", a), "[1, 2, 3]");
    }
}

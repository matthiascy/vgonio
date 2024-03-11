use crate::array::mem::{Data, Sealed};

#[repr(transparent)]
pub struct FixedSized<A, const N: usize>(pub(crate) [A; N]);

impl<A, const N: usize> Sealed for FixedSized<A, N> {}
impl<'a, A, const N: usize> Sealed for &'a FixedSized<A, N> {}
impl<'a, A, const N: usize> Sealed for &'a mut FixedSized<A, N> {}

impl<A: Clone, const N: usize> Clone for FixedSized<A, N> {
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<A: Copy, const N: usize> Copy for FixedSized<A, N> {}

impl<A: PartialEq, const N: usize> PartialEq for FixedSized<A, N> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<A: Eq, const N: usize> Eq for FixedSized<A, N> {}

unsafe impl<A, const N: usize> Data for FixedSized<A, N> {
    type Elem = A;

    fn as_ptr(&self) -> *const A { self.0.as_ptr() }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem { self.0.as_mut_ptr() }

    fn as_slice(&self) -> &[Self::Elem] { &self.0 }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] { &mut self.0 }

    unsafe fn alloc_uninit(n: usize) -> Self { std::mem::MaybeUninit::uninit().assume_init() }
}

impl<A: std::fmt::Debug, const N: usize> std::fmt::Debug for FixedSized<A, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("FixedSized({:?})", &self.0))
    }
}

impl<A: std::fmt::Display, const N: usize> std::fmt::Display for FixedSized<A, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::array::mem::print_slice(f, &self.0)
    }
}

impl<A, const N: usize> std::ops::Deref for FixedSized<A, N> {
    type Target = [A];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<A, const N: usize> From<[A; N]> for FixedSized<A, N> {
    fn from(array: [A; N]) -> Self { Self(array) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_sized() {
        let array = FixedSized([1, 2, 3]);
        assert_eq!(array.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn fixed_sized_display() {
        let a = FixedSized([1, 2, 3]);
        assert_eq!(format!("{:?}", a), "FixedSized([1, 2, 3])");
        assert_eq!(format!("{}", a), "[1, 2, 3]");
    }
}

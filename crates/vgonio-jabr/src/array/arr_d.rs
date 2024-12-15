use crate::array::{
    core::ArrCore,
    mem::{heap::DynSized, MemLayout},
    shape::ConstShape,
    DyArr,
};
use num_traits::{One, Zero};
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

/// A fixed-size multidimensional array on the heap with type level fixed shape,
/// in other words with fixed dimensions and size of each dimension.
///
/// The number of dimensions and the size of each dimension are set at
/// compilation time and can't be changed.
pub struct DArr<T, S, const L: MemLayout = { MemLayout::RowMajor }>(
    pub(crate) ArrCore<DynSized<T>, S, L>,
)
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:;

impl<T, S, const L: MemLayout> DArr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    forward_array_core_common_methods!();

    /// Creates a new array with the given data and shape.
    pub fn new(data: [T; S::N_ELEMS]) -> Self
    where
        T: Clone,
    {
        Self(ArrCore::new(S::SHAPE, DynSized::from_slice(&data)))
    }

    pub fn ones() -> Self
    where
        T: One + Clone,
    {
        Self(ArrCore::new(
            S::SHAPE,
            DynSized::splat(T::one(), S::N_ELEMS),
        ))
    }

    pub fn zeros() -> Self
    where
        T: Zero + Clone,
    {
        Self(ArrCore::new(
            S::SHAPE,
            DynSized::splat(T::zero(), S::N_ELEMS),
        ))
    }

    pub fn full(value: T) -> Self
    where
        T: Clone,
    {
        Self(ArrCore::new(
            S::SHAPE,
            DynSized::splat(value.clone(), S::N_ELEMS),
        ))
    }

    pub fn splat(value: T) -> Self
    where
        T: Clone,
    {
        Self(ArrCore::new(
            S::SHAPE,
            DynSized::splat(value.clone(), S::N_ELEMS),
        ))
    }

    pub fn as_slice(&self) -> &[T] { self.0.data.as_slice() }

    pub fn as_mut_slice(&mut self) -> &mut [T] { self.0.data.as_mut_slice() }
}

impl<T, S, const L: MemLayout, const N: usize> Index<[usize; N]> for DArr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; N]) -> &Self::Output { &self.0[index] }
}

impl<T, S, const L: MemLayout, const N: usize> IndexMut<[usize; N]> for DArr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    #[inline]
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output { &mut self.0[index] }
}

impl<T, S, const L: MemLayout> Clone for DArr<T, S, L>
where
    T: Clone,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T, S, const L: MemLayout> Debug for DArr<T, S, L>
where
    T: Debug,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("DArr({:?})", &self.0.data))
    }
}

impl<T, S, const L: MemLayout> PartialEq for DArr<T, S, L>
where
    T: PartialEq,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

impl<T, S, const L: MemLayout> Eq for DArr<T, S, L>
where
    T: Eq,
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
}

impl<T, S, const L: MemLayout> AsRef<[T]> for DArr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn as_ref(&self) -> &[T] { self.as_ref() }
}

impl<T, S, const L: MemLayout> AsMut<[T]> for DArr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    fn as_mut(&mut self) -> &mut [T] { self.0.data.as_mut_slice() }
}

/// A macro to create a fixed-size array on the heap with type level fixed
/// shape.
pub macro darr($($n:expr),+ $(,)*; [$($x:expr),* $(,)*]) {{
    crate::array::DArr::<_, s![$($n),*], { MemLayout::RowMajor }>::new([$($x),*])
}}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::s;

    #[test]
    fn test_darr_creation() {
        let arr: DArr<i32, s![2, 3, 2, 2], { MemLayout::RowMajor }> = DArr::new([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]);
        assert_eq!(arr.shape(), &[2, 3, 2, 2]);
        assert_eq!(arr.strides(), &[12, 4, 2, 1]);
    }

    #[test]
    fn test_darr_macro() {
        let arr = darr!(2, 2, 2, 2; [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(arr.shape(), &[2, 2, 2, 2]);
        assert_eq!(arr.strides(), &[8, 4, 2, 1]);
    }

    #[test]
    fn test_darr_ones() {
        let arr = DArr::<i32, s![2, 2], { MemLayout::RowMajor }>::ones();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.strides(), &[2, 1]);
        assert_eq!(arr[[0, 0]], 1);
        assert_eq!(arr[[1, 1]], 1);
    }

    #[test]
    fn test_darr_zeros() {
        let arr = DArr::<i32, s![2, 2], { MemLayout::RowMajor }>::zeros();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.strides(), &[2, 1]);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[1, 1]], 0);
    }
}

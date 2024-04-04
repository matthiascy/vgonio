use crate::array::{
    core::ArrCore,
    mem::{heap::DynSized, MemLayout},
};

/// A dynamically-sized array with a known number of dimensions at compile-time.
///
/// The dimension is set at compilation time and cannot be changed, but the size
/// of each dimension can be changed at runtime (non-growable).
pub struct DyArr<T, const N: usize, const L: MemLayout = { MemLayout::ColMajor }>(
    ArrCore<DynSized<T>, [usize; N], L>,
);

/// A dynamically-sized array with dynamical number of dimensions
/// and size of each dimension at runtime (growable).
pub struct DynArr<T, const L: MemLayout = { MemLayout::ColMajor }>(
    ArrCore<DynSized<T>, Vec<usize>, L>,
);

impl<T, const N: usize, const L: MemLayout> DyArr<T, N, L> {
    /// Creates a new array with the given data and shape.
    pub fn new(shape: [usize; N]) -> Self {
        Self(ArrCore::new(
            shape,
            DynSized::with_capacity(shape.iter().product()),
        ))
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] { self.0.shape() }

    /// Returns the strides of the array.
    pub fn strides(&self) -> &[usize] { self.0.strides() }
}

impl<T, const L: MemLayout> DynArr<T, L> {
    /// Creates a new array with the given data and shape.
    pub fn new(shape: &[usize]) -> Self {
        let n_elems = shape.iter().product();
        let shape = shape.to_vec();
        Self(ArrCore::new(shape, DynSized::with_capacity(n_elems)))
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] { self.0.shape() }

    /// Returns the strides of the array.
    pub fn strides(&self) -> &[usize] { self.0.strides() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyarr_creation() {
        let arr: DyArr<i32, 3, { MemLayout::RowMajor }> = DyArr::new([2, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
    }

    #[test]
    fn test_dynarr_creation() {
        let arr: DynArr<i32, { MemLayout::RowMajor }> = DynArr::new(&[2, 3, 2]);
        assert_eq!(arr.shape(), &[2, 3, 2]);
        assert_eq!(arr.strides(), &[6, 2, 1]);
    }
}

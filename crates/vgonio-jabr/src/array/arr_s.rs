use crate::array::{
    core::ArrCore,
    mem::{stack::FixedSized, MemLayout},
    shape::ConstShape,
};

/// A fixed-size multidimensional array on the stack with type level fixed
/// shape (dimensions and size of each dimension).
/// There is no extra metadata associated with the array.
pub struct Arr<T, S, const L: MemLayout = { MemLayout::ColMajor }>(
    ArrCore<FixedSized<T, { S::N_ELEMS }>, S, L>,
)
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:;

impl<T, S, const L: MemLayout> Arr<T, S, L>
where
    S: ConstShape<Underlying = [usize; S::N_DIMS]>,
    [(); S::N_ELEMS]:,
{
    /// Creates a new array with the given data and shape.
    pub fn new(data: [T; S::N_ELEMS]) -> Self { Self(ArrCore::new(S::SHAPE, FixedSized(data))) }

    /// Returns the shape of the array.
    #[inline]
    pub fn shape(&self) -> &[usize] { self.0.shape() }

    /// Returns the strides of the array.
    #[inline]
    pub fn strides(&self) -> &[usize] { self.0.strides() }

    /// Returns the layout of the array.
    #[inline]
    pub fn order(&self) -> MemLayout { self.0.order() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::s;

    #[test]
    fn test_arr_creation() {
        let arr: Arr<i32, s![2, 3], { MemLayout::ColMajor }> = Arr::new([1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.strides(), &[1, 2]);
        assert_eq!(arr.order(), MemLayout::ColMajor);
    }
}

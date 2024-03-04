use crate::array::dim::DimSeq;

/// Common trait for types that can be used to represent the shape of an array.
pub trait Shape {
    /// The underlying type used to store the shape.
    type Underlying: DimSeq;
}

/// Trait for fixed-size multidimensional shapes with a known number of
/// dimensions and size of each dimension at compile-time.
///
/// This trait is a helper to construct a concrete array shape from
/// the recursively defined const shape type `ConstShape`.
pub trait CShape {
    /// Underlying storage type for the shape.
    type Underlying: DimSeq;
    /// The number of dimensions of the array.
    const N_DIMS: usize;
    /// The number of elements in the array.
    const N_ELEMS: usize;
    /// The shape of the array. For a fixed-size shape, this is a const array.
    const SHAPE: Self::Underlying;
    /// Pre-computed array strides for row-major layout: the number of elements
    /// needed to move one step in each dimension.
    const ROW_MAJOR_STRIDES: Self::Underlying;
    /// Pre-computed array strides for column-major layout: the number of
    /// elements needed to move one step in each dimension.
    const COL_MAJOR_STRIDES: Self::Underlying;
}

impl<const N: usize> Shape for [usize; N] {
    type Underlying = [usize; N];
}

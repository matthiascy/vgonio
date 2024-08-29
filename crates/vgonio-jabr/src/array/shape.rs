use crate::array::{dim::DimSeq, mem::MemLayout};

// TODO: #[const_trait], blocked by effects feature and new const traits
// implementation See: https://github.com/rust-lang/rust/issues/110395
/// Common trait for types that can be used to represent the shape of an array.
pub trait Shape {
    /// The underlying type used to store the shape.
    type Underlying: DimSeq;
    type Metadata: ShapeMetadata;
    /// Creates a new metadata object for the shape.
    fn new_metadata(dims: &Self::Underlying, layout: MemLayout) -> Self::Metadata;
}

pub trait ShapeMetadata: Clone + PartialEq {
    fn shape(&self) -> &[usize];
    fn strides<const L: MemLayout>(&self) -> &[usize];
    fn dimension(&self) -> usize;
}

// TODO: constify
/// Computes the index of an element in a multidimensional array given its
/// coordinates and the layout of the array.
pub fn compute_index<M, const L: MemLayout>(meta: &M, index: &[usize]) -> usize
where
    M: ShapeMetadata,
{
    let strides = meta.strides::<L>();
    let mut i = 0;
    let mut idx = 0;
    while i < meta.dimension() {
        idx += index[i] * strides[i];
        i += 1;
    }
    idx
}

/// Metadata for dynamically-sized shapes.
#[derive(Clone, Debug)]
pub struct DynShapeMetadata<T>
where
    T: DimSeq,
{
    /// The shape of the array including the number of dimensions and the size
    /// of each dimension.
    pub shape: T,
    /// The number of elements needed to skip to get to the next element along
    /// each dimension. Its interpretation depends on the layout of the array.
    pub strides: T,
}

impl<T> ShapeMetadata for DynShapeMetadata<T>
where
    T: DimSeq,
{
    fn shape(&self) -> &[usize] { self.shape.as_slice() }

    fn strides<const L: MemLayout>(&self) -> &[usize] { self.strides.as_slice() }

    fn dimension(&self) -> usize { self.shape.len() }
}

impl<T> PartialEq for DynShapeMetadata<T>
where
    T: DimSeq,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape.as_slice() == other.shape.as_slice()
            && self.strides.as_slice() == other.strides.as_slice()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FixedShapeMetadata<T: ConstShape>(::core::marker::PhantomData<T>);

impl<T, const N: usize> ShapeMetadata for FixedShapeMetadata<T>
where
    T: ConstShape<Underlying = [usize; N]>,
{
    #[inline]
    fn shape(&self) -> &[usize] { &T::SHAPE }

    #[inline]
    fn strides<const L: MemLayout>(&self) -> &[usize] {
        if L == MemLayout::RowMajor {
            &T::ROW_MAJOR_STRIDES
        } else {
            &T::COL_MAJOR_STRIDES
        }
    }

    fn dimension(&self) -> usize { T::N_DIMS }
}

/// Shape for fixed-size dimension sequences.
impl<const N: usize> Shape for [usize; N] {
    type Underlying = [usize; N];
    type Metadata = DynShapeMetadata<[usize; N]>;

    fn new_metadata(dims: &Self::Underlying, layout: MemLayout) -> Self::Metadata {
        let shape = dims.clone();
        let mut strides = [0usize; N];
        compute_strides(dims.as_slice(), &mut strides, layout);
        DynShapeMetadata { shape, strides }
    }
}

/// Shape for dynamically-sized dimension sequences.
impl Shape for Vec<usize> {
    type Underlying = Vec<usize>;
    type Metadata = DynShapeMetadata<Vec<usize>>;

    fn new_metadata(dims: &Self::Underlying, layout: MemLayout) -> Self::Metadata {
        let shape = dims.clone();
        let mut strides = dims.to_vec();
        compute_strides(shape.as_slice(), &mut strides, layout);
        DynShapeMetadata { shape, strides }
    }
}

/// Trait for fixed-size multidimensional shapes with a known number of
/// dimensions and size of each dimension at compile-time.
///
/// This trait is a helper to construct a concrete array shape from
/// type-level constants.
#[const_trait]
pub trait ConstShape: Sized + Clone + Copy + PartialEq {
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

impl<T, const N: usize> Shape for T
where
    T: ConstShape<Underlying = [usize; N]>,
{
    type Underlying = <T as ConstShape>::Underlying;
    type Metadata = FixedShapeMetadata<T>;

    fn new_metadata(_: &Self::Underlying, _: MemLayout) -> Self::Metadata {
        FixedShapeMetadata(::core::marker::PhantomData)
    }
}

mod const_shape {
    use crate::array::{
        mem::MemLayout,
        shape::{compute_strides, ConstShape},
    };

    impl ConstShape for () {
        type Underlying = [usize; 0];
        const N_DIMS: usize = 0;
        const N_ELEMS: usize = 0;
        const SHAPE: Self::Underlying = [];
        const ROW_MAJOR_STRIDES: Self::Underlying = [];
        const COL_MAJOR_STRIDES: Self::Underlying = [];
    }

    /// A type-level representation of a shape with a single dimension.
    ///
    /// This shape type embeds the number of dimensions and the size of each
    /// dimension at compile-time as type-level constants. The shape will be
    /// constructed recursively by adding dimensions to the shape. This allows
    /// the creation of multidimensional arrays with a fixed shape at
    /// compile-time. Macro `s!` is provided to simplify the creation of shapes.
    ///
    /// # Note
    ///
    /// This is a workaround to the lack of variadic generics in Rust. Once
    /// variadic generics are stabilized, this type will be replaced by a
    /// variadic generic type.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CShape<A: ConstShape, const N: usize>(::core::marker::PhantomData<[A; N]>);

    /// Recursive macro generating type signature for `CShape` from a sequence
    /// of integers representing the size of each dimension.
    macro generate_const_shape {
        ($n:ident) => {
            CShape<(), $n>
        },
        ($n:ident, $($ns:ident),+) => {
            CShape<generate_const_shape!($($ns),+), $n>
        }
    }

    /// Macro counting the number of elements in a list of arguments.
    pub macro count {
        ($x:tt) => { 1usize },
        ($x:tt, $($xs:tt),*) => { 1usize + count!($($xs),*) }
    }

    /// Macro calculating the product of the elements in a list of arguments.
    macro product {
        ($x:tt) => { $x },
        ($x:tt, $($xs:tt),+) => { $x * product!($($xs),+) }
    }

    const fn calc_row_major_strides<const N: usize>(shape: &[usize; N]) -> [usize; N] {
        let mut strides = [1usize; N];
        compute_strides(shape, &mut strides, MemLayout::RowMajor);
        strides
    }

    const fn calc_col_major_strides<const N: usize>(shape: &[usize; N]) -> [usize; N] {
        let mut strides = [1usize; N];
        compute_strides(shape, &mut strides, MemLayout::ColMajor);
        strides
    }

    /// Macro generating the implementation of `ConstShape` for a given shape.
    macro impl_const_shape($($n:ident),+) {
        impl<$(const $n: usize),+> const ConstShape for generate_const_shape!($($n),+) {
            type Underlying = [usize; count!($($n),+)];
            const N_DIMS: usize = count!($($n),+);
            const N_ELEMS: usize = product!($($n),+);
            const SHAPE: Self::Underlying = [$($n),+];
            const ROW_MAJOR_STRIDES: Self::Underlying = calc_row_major_strides(&[$($n),+]);
            const COL_MAJOR_STRIDES: Self::Underlying = calc_col_major_strides(&[$($n),+]);
        }
    }

    /// Recursive macro generating the implementation of `ConstShape` for a
    /// sequence of shapes.
    macro impl_const_shapes {
        (@inner $n:tt) => {  },
        (@inner $n:tt, $($ns:tt),*) => {
            impl_const_shape!($($ns),*);
            impl_const_shapes!(@inner $($ns),*);
        },
        ($($ns:tt),*) => {
            impl_const_shape!($($ns),*);
            impl_const_shapes!(@inner $($ns),*);
        }
    }

    impl_const_shapes!(N0, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15);

    /// Macro generating a type alias for a shape with a given number of
    /// dimensions.
    pub macro s {
        ($n:expr) => { CShape<(), $n> },
        ($n:expr, $($ns:expr),*) => { CShape<s!($($ns),*), $n> }
    }
}

pub use const_shape::s;

/// Computes the number of elements in an array with the given shape.
pub(crate) const fn compute_n_elems(shape: &[usize]) -> usize {
    let mut n_elems = 1;
    let mut i = 0;
    let n = shape.len();
    while i < n {
        n_elems *= shape[i];
        i += 1;
    }
    n_elems
}

/// Computes the strides of an array with the given shape and layout.
pub(crate) const fn compute_strides(shape: &[usize], strides: &mut [usize], layout: MemLayout) {
    let n = shape.len();
    let mut i = 0;
    let mut stride = 1;
    match layout {
        MemLayout::RowMajor => {
            while i < n {
                strides[n - i - 1] = stride;
                stride *= shape[n - i - 1];
                i += 1;
            }
        }
        MemLayout::ColMajor => {
            while i < n {
                strides[i] = stride;
                stride *= shape[i];
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_const_shape_macro() {
        type S3 = s![2, 3, 4];
        assert_eq!(S3::N_DIMS, 3);
        assert_eq!(S3::N_ELEMS, 24);
        assert_eq!(S3::SHAPE, [2, 3, 4]);
        assert_eq!(S3::ROW_MAJOR_STRIDES, [12, 4, 1]);
        assert_eq!(S3::COL_MAJOR_STRIDES, [1, 2, 6]);
    }

    #[test]
    fn test_num_elems() {
        assert_eq!(compute_n_elems(&[2, 3, 4]), 24);
        assert_eq!(compute_n_elems(&[5, 6, 7]), 210);
    }

    #[test]
    fn test_strides() {
        let mut strides = [0; 4];
        compute_strides(&[2, 3, 4, 5], &mut strides, MemLayout::RowMajor);
        assert_eq!(strides, [60, 20, 5, 1]);

        let mut strides = [0; 4];
        compute_strides(&[2, 3, 4, 5], &mut strides, MemLayout::ColMajor);
        assert_eq!(strides, [1, 2, 6, 24]);
    }
}

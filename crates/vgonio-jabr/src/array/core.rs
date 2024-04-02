use crate::array::{
    dim::DimSeq,
    mem::{Data, MemLayout},
    shape::{compute_n_elems, Shape, ShapeMetadata},
};

/// Base struct for all arrays.
pub(crate) struct ArrCore<D, S, const L: MemLayout = { MemLayout::ColMajor }>
where
    D: Data,
    S: Shape,
{
    /// The data of the array.
    data: D,
    /// The extra shape info of the array.
    meta: S::Metadata,
    marker: core::marker::PhantomData<(D, S)>,
}

impl<D, S, const L: MemLayout> ArrCore<D, S, L>
where
    D: Data,
    S: Shape,
{
    /// Creates a new array with the given data, shape, strides, and layout.
    ///
    /// Rarely used directly. Only used inside the crate.
    pub fn new(shape: S::Underlying, data: D) -> Self {
        Self {
            data,
            meta: S::new_metadata(&shape, L),
            marker: core::marker::PhantomData,
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize] { self.meta.shape() }

    /// Returns the strides of the array.
    pub fn strides(&self) -> &[usize] { self.meta.strides::<L>() }

    /// Returns the layout of the array.
    pub fn order(&self) -> MemLayout { L }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{
        mem::{
            heap::{DynFixSized, DynSized},
            stack::FixedSized,
        },
        shape::s,
    };

    #[test]
    fn test_arr_core_shape() {
        let arr: ArrCore<FixedSized<f32, 24>, s![3, 2, 4]> =
            ArrCore::new([3, 2, 4], FixedSized([0.0f32; 24]));
        let darr: ArrCore<DynFixSized<f32, 9>, s![3, 3]> =
            ArrCore::new([3, 3], DynFixSized::from_slice(&[3.0f32; 9]));
        let dyarr: ArrCore<DynSized<f32>, [usize; 3], { MemLayout::RowMajor }> =
            ArrCore::new([4, 6, 2], DynSized::from(vec![3.0f32; 48]));
        let dynarr: ArrCore<DynSized<f32>, Vec<usize>, { MemLayout::RowMajor }> =
            ArrCore::new(vec![4, 6, 2], DynSized::from(vec![3.0f32; 48]));
        assert_eq!(arr.shape(), [3, 2, 4]);
        assert_eq!(arr.strides(), [1, 3, 6]);

        assert_eq!(darr.shape(), [3, 3]);
        assert_eq!(darr.strides(), [1, 3]);

        assert_eq!(dyarr.shape(), [4, 6, 2]);
        assert_eq!(dyarr.strides(), [12, 2, 1]);

        assert_eq!(dynarr.shape(), dynarr.shape());
        assert_eq!(dynarr.strides(), dynarr.strides());
    }

    #[test]
    fn test_arr_core_size() {
        let arr: ArrCore<FixedSized<f32, 24>, s![3, 2, 4]> =
            ArrCore::new([3, 2, 4], FixedSized([0.0f32; 24]));
        let darr: ArrCore<DynFixSized<f32, 9>, s![3, 3]> =
            ArrCore::new([3, 3], DynFixSized::from_slice(&[3.0f32; 9]));
        let dyarr: ArrCore<DynSized<f32>, [usize; 3], { MemLayout::ColMajor }> =
            ArrCore::new([4, 6, 2], DynSized::from(vec![3.0f32; 48]));
        let dynarr: ArrCore<DynSized<f32>, Vec<usize>, { MemLayout::RowMajor }> =
            ArrCore::new(vec![4, 6, 2], DynSized::from(vec![3.0f32; 48]));
        assert_eq!(std::mem::size_of_val(&arr), 24 * std::mem::size_of::<f32>());
        assert_eq!(
            std::mem::size_of_val(&darr),
            std::mem::size_of::<DynFixSized<f32, 9>>()
        );
        assert_eq!(
            std::mem::size_of_val(&dyarr),
            std::mem::size_of::<DynSized<f32>>() + 3 * std::mem::size_of::<usize>() * 2
        );
        assert_eq!(
            std::mem::size_of_val(&dynarr),
            std::mem::size_of::<DynSized<f32>>() + std::mem::size_of::<Vec<usize>>() * 2
        );
    }
}

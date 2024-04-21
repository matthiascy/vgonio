use crate::array::{
    mem::{Data, DataClone, DataCopy, MemLayout},
    shape::{self, Shape, ShapeMetadata},
};
use std::ops::Index;

/// Base struct for all arrays.
pub(crate) struct ArrCore<D, S, const L: MemLayout = { MemLayout::RowMajor }>
where
    D: Data,
    S: Shape,
{
    /// The data of the array.
    pub data: D,
    /// The extra shape info of the array.
    pub meta: S::Metadata,
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
    #[inline]
    pub const fn shape(&self) -> &[usize] { self.meta.shape() }

    /// Returns the strides of the array.
    #[inline]
    pub const fn strides(&self) -> &[usize] { self.meta.strides::<L>() }

    /// Returns the number of dimensions of the array.
    #[inline]
    pub const fn dimension(&self) -> usize { self.meta.dimension() }

    /// Returns the layout of the array.
    #[inline]
    pub const fn order(&self) -> MemLayout { L }

    /// Returns the total number of elements in the array.
    #[inline]
    pub const fn len(&self) -> usize { shape::compute_n_elems(self.meta.shape()) }
}

impl<D, S, const L: MemLayout> Clone for ArrCore<D, S, L>
where
    D: DataClone,
    S: Shape,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            meta: self.meta.clone(),
            marker: core::marker::PhantomData,
        }
    }
}

impl<D, S, const L: MemLayout> Copy for ArrCore<D, S, L>
where
    D: DataCopy,
    S: Shape,
    S::Metadata: Copy,
{
}

impl<D, S, const L: MemLayout, const N: usize> Index<[usize; N]> for ArrCore<D, S, L>
where
    D: Data,
    S: Shape,
{
    type Output = D::Elem;

    #[track_caller]
    fn index(&self, index: [usize; N]) -> &Self::Output {
        debug_assert!(
            N <= self.meta.dimension()
                && index
                    .iter()
                    .zip(self.meta.shape().iter())
                    .all(|(&i, &s)| i < s),
            "Index out of bounds"
        );
        let idx = shape::compute_index::<_, L>(&self.meta, &index);
        &self.data.as_slice()[idx]
    }
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
        let dyarr: ArrCore<DynSized<f32>, [usize; 3], { MemLayout::ColMajor }> =
            ArrCore::new([4, 6, 2], DynSized::from(vec![3.0f32; 48]));
        assert_eq!(arr.shape(), [3, 2, 4]);
        assert_eq!(arr.strides(), [8, 4, 1]);

        assert_eq!(darr.shape(), [3, 3]);
        assert_eq!(darr.strides(), [3, 1]);

        assert_eq!(dyarr.shape(), [4, 6, 2]);
        assert_eq!(dyarr.strides(), [1, 4, 24]);
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

    #[test]
    fn test_arr_core_index() {
        let data = (0..48).into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let dyarr: ArrCore<DynSized<f32>, [usize; 3], { MemLayout::ColMajor }> =
            ArrCore::new([4, 6, 2], DynSized::from(data));
        assert_eq!(dyarr[[0, 0, 0]], 0.0);
        assert_eq!(dyarr[[1, 2, 1]], 33.0);
        assert_eq!(dyarr[[3, 5, 1]], 47.0);
        // TODO: test with other layouts
    }

    #[test]
    #[should_panic]
    fn test_arr_core_index_panic_dimension_mismatch() {
        let data = (0..48).into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let dynarr: ArrCore<DynSized<f32>, Vec<usize>, { MemLayout::RowMajor }> =
            ArrCore::new(vec![2, 3, 4, 2], DynSized::from(data));
        assert_eq!(dynarr[[0, 0, 0, 0, 0]], 0.0);
    }

    #[test]
    #[should_panic]
    fn test_arr_core_index_panic_size_mismatch() {
        let data = (0..48).into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let dynarr: ArrCore<DynSized<f32>, Vec<usize>> =
            ArrCore::new(vec![2, 3, 4, 2], DynSized::from(data));
        assert_eq!(dynarr[[1, 2, 4, 0]], 0.0);
    }

    #[test]
    fn test_arr_num_elems() {
        let arr: ArrCore<FixedSized<f32, 24>, s![3, 2, 4]> =
            ArrCore::new([3, 2, 4], FixedSized([0.0f32; 24]));
        let darr: ArrCore<DynFixSized<f32, 9>, s![3, 3]> =
            ArrCore::new([3, 3], DynFixSized::from_slice(&[3.0f32; 9]));
        let dyarr: ArrCore<DynSized<f32>, [usize; 3], { MemLayout::ColMajor }> =
            ArrCore::new([4, 6, 2], DynSized::from(vec![3.0f32; 48]));
        assert_eq!(arr.len(), 24);
        assert_eq!(darr.len(), 9);
        assert_eq!(dyarr.len(), 48);
    }
}

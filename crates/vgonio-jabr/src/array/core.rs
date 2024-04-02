use crate::array::{
    dim::DimSeq,
    mem::{Data, MemLayout, MemLayout::ColMajor},
    shape::{compute_n_elems, Shape, ShapeMetadata},
};

/// Base struct for all arrays.
pub struct ArrCore<D, S, const L: MemLayout = ColMajor>
where
    D: Data,
    S: Shape<L>,
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
    S: Shape<L>,
{
    /// Creates a new array with the given data, shape, strides, and layout.
    ///
    /// Rarely used directly. Only used inside the crate.
    pub fn new(shape: S::Underlying, data: D) -> Self {
        // Make sure the data is the right size.
        debug_assert_eq!(
            data.as_slice().len(),
            compute_n_elems(shape.as_slice()),
            "data size doesn't match shape"
        );
        Self {
            data,
            meta: S::new_metadata(&shape),
            marker: core::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[usize] { self.meta.shape() }

    pub fn strides(&self) -> &[usize] { self.meta.strides() }
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
        let arr: ArrCore<FixedSized<f32, 9>, s![3, 3]> =
            ArrCore::new([3, 3], FixedSized([3.0f32; 9]));
        let darr: ArrCore<DynFixSized<f32, 9>, s![3, 3]> =
            ArrCore::new([3, 3], DynFixSized::from_slice(&[3.0f32; 9]));
        let dyarr: ArrCore<DynSized<f32>, [usize; 2]> =
            ArrCore::new([3, 3], DynSized::from(vec![3.0f32; 9]));
        println!("{:?}", arr.shape());
        println!("{:?}", darr.shape());
        println!("{:?}", dyarr.shape());
    }
}

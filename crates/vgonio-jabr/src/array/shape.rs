/// Dimension sequence.
///
/// Trait for types that can be used to represent the shape of an array.
pub trait DimSeq: Clone + PartialEq + Eq {
    fn n_dims(&self) -> usize;
    fn as_slice(&self) -> &[usize];
    fn as_mut_slice(&mut self) -> &mut [usize];
}

/// A fixed-size dimension sequence with `N` elements.
impl<const N: usize> DimSeq for [usize; N] {
    fn n_dims(&self) -> usize { N }

    fn as_slice(&self) -> &[usize] { &self[..] }

    fn as_mut_slice(&mut self) -> &mut [usize] { &mut self[..] }
}

/// A dynamically-sized dimension sequence.
impl DimSeq for Vec<usize> {
    fn n_dims(&self) -> usize { self.len() }

    fn as_slice(&self) -> &[usize] { &self[..] }

    fn as_mut_slice(&mut self) -> &mut [usize] { &mut self[..] }
}

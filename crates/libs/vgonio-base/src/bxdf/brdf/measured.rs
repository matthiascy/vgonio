//! Measured BRDF models.
use crate::{math::Sph2, units::Nanometres, utils::medium::Medium};
use jabr::array::DyArr;
use std::{fmt::Debug, ops::Index};

pub mod clausen;
pub mod merl;
pub mod rgl;
pub mod utia;
pub mod vgonio;
pub mod yan;

/// The kind of the measured BRDF.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasuredBrdfKind {
    #[cfg_attr(feature = "cli", clap(name = "clausen"))]
    /// The measured BRDF by Clausen.
    Clausen,
    #[cfg_attr(feature = "cli", clap(name = "merl"))]
    /// The MERL BRDF dataset.
    Merl,
    #[cfg_attr(feature = "cli", clap(name = "utia"))]
    /// The measured BRDF by UTIA at Czech Technical University.
    Utia,
    #[cfg_attr(feature = "cli", clap(name = "rgl"))]
    /// The measured BRDF by Dupuy and Jakob in RGL at EPFL.
    Rgl,
    #[cfg_attr(feature = "cli", clap(name = "vgonio"))]
    /// The simulated BRDF by vgonio.
    Vgonio,
    #[cfg_attr(feature = "cli", clap(name = "yan2018"))]
    /// The BRDF model by Yan et al. 2018.
    Yan2018,
    #[cfg_attr(feature = "cli", clap(name = "unknown"))]
    /// Unknown.
    Unknown,
}

/// The origin of the measured BRDF.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Origin {
    /// Measured in the real world.
    RealWorld,
    /// Measured in a simulation.
    Simulated,
    /// Analytically defined, the data is generated from a mathematical model.
    /// In this case, the measured BRDF is just a collection of samples of
    /// the analytical model.
    Analytical,
}

/// The parameterisation kind of the measured BRDF.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum BrdfParamKind {
    /// The BRDF is parameterised in the half-vector domain.
    HalfVector,
    /// The BRDF is parameterised in the incident and outgoing directions.
    InOutDirs,
}

/// Measured BRDF parameterisation.
pub trait BrdfParam: PartialEq {
    /// Returns the kind of the parameterisation.
    fn kind() -> BrdfParamKind;
}

pub use clausen::*;
pub use merl::*;
pub use utia::*;
pub use vgonio::*;
pub use yan::*;

/// A BRDF measured either in the real world or in a simulation.
///
/// The BRDF is parameterised by a specific parameterisation. Extra data can be
/// associated with the measured BRDF.
#[derive(Clone, Debug)]
pub struct MeasuredBrdf<P, const N: usize>
where
    P: Clone + Send + Sync + BrdfParam + 'static,
{
    /// The origin of the measured BRDF.
    pub origin: Origin,
    /// Incident medium.
    pub incident_medium: Medium,
    /// Transmitted medium.
    pub transmitted_medium: Medium,
    /// The parameterisation of the measured BRDF.
    pub params: Box<P>,
    /// Wavelengths at which the BRDF is measured.
    pub spectrum: DyArr<Nanometres>,
    /// Sampled BRDF data stored in a multidimensional array.
    /// Its shape depends on the parameterisation of the BRDF.
    pub samples: DyArr<f32, N>,
    /// The kind of the measured BRDF.
    pub kind: MeasuredBrdfKind,
}

impl<P, const N: usize> MeasuredBrdf<P, N>
where
    P: Clone + Send + Sync + BrdfParam + PartialEq + 'static,
{
    /// Returns the number of wavelengths of the measured BRDF.
    pub fn n_spectrum(&self) -> usize { self.spectrum.len() }

    /// Returns the number of samples of the measured BRDF.
    pub fn n_samples(&self) -> usize { self.samples.len() }
}

impl<P, const N: usize> PartialEq for MeasuredBrdf<P, N>
where
    P: Clone + Send + Sync + BrdfParam + PartialEq + 'static,
{
    fn eq(&self, other: &Self) -> bool {
        self.origin == other.origin
            && self.incident_medium == other.incident_medium
            && self.transmitted_medium == other.transmitted_medium
            && self.params == other.params
            && self.spectrum == other.spectrum
            && self.samples == other.samples
            && self.kind == other.kind
    }
}

/// An iterator over the snapshots of a measured BRDF.
pub struct BrdfSnapshotIterator<'a, P, const N: usize>
where
    P: Clone + Send + Sync + BrdfParam + PartialEq + 'static,
{
    /// The measured BRDF.
    brdf: &'a MeasuredBrdf<P, N>,
    /// Number of wavelengths of the measured BRDF.
    n_spectrum: usize,
    /// Number of incident directions of the measured BRDF.
    n_incoming: usize,
    /// Number of outgoing directions of the measured BRDF.
    n_outgoing: usize,
    /// The current index (snapshot) of the iterator.
    index: usize,
}

/// Snapshot of a measured BRDF at a specific incident direction.
pub struct BrdfSnapshot<'a, S> {
    /// The incident direction of the snapshot.
    pub wi: Sph2,
    // TODO: use NdArray subslice which carries the shape information.
    /// Number of wavelengths of the measured BRDF.
    pub n_spectrum: usize,
    // TODO: use NdArray subslice which carries the shape information.
    /// Samples of the snapshot stored in a flat row-major array with
    /// dimensions ωο, λ.
    pub samples: &'a [S],
}

// TODO: once the NdArray subslice is implemented, no need to implement Index
impl<S: 'static> Index<[usize; 2]> for BrdfSnapshot<'_, S> {
    type Output = S;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        // The first index is the outgoing direction and the second index is the
        // wavelength.
        &self.samples[index[0] * self.n_spectrum + index[1]]
    }
}

impl<'a, P, S: 'static, const N: usize> ExactSizeIterator for BrdfSnapshotIterator<'a, P, N>
where
    BrdfSnapshotIterator<'a, P, N>: Iterator<Item = BrdfSnapshot<'a, S>>,
    P: Clone + Send + Sync + BrdfParam + PartialEq + 'static,
{
    fn len(&self) -> usize { self.n_incoming - self.index }
}

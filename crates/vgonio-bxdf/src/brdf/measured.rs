use base::{math::Sph2, medium::Medium, units::Nanometres, MeasuredBrdfKind};
#[cfg(feature = "fitting")]
use base::{
    optics::ior::RefractiveIndexRegistry, units::Radians, ErrorMetric, ResidualErrorMetric,
};
use jabr::array::DyArr;
use std::{fmt::Debug, ops::Index};

#[cfg(feature = "fitting")]
macro_rules! impl_analytical_fit_trait {
    ($self:ident) => {
        #[inline]
        fn spectrum(&$self) -> &[Nanometres] { $self.spectrum.as_ref() }

        #[inline]
        fn samples(&$self) -> &[f32] { $self.samples.as_ref() }

        #[inline]
        fn params(&$self) -> &Self::Params { &$self.params }

        #[inline]
        fn incident_medium(&$self) -> Medium {
            $self.incident_medium
        }

        #[inline]
        fn transmitted_medium(&$self) -> Medium {
            $self.transmitted_medium
        }

        #[inline]
        fn as_any(&$self) -> &dyn std::any::Any {
            $self
        }

        #[inline]
        fn as_any_mut(&mut $self) -> &mut dyn std::any::Any {
            $self
        }
    };
}

pub mod clausen;
pub mod merl;
pub mod utia;
pub mod vgonio;
pub mod yan;

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
pub enum ParametrisationKind {
    /// The BRDF is parameterised in the half-vector domain.
    HalfVector,
    /// The BRDF is parameterised in the incident direction domain.
    IncidentDirection,
}

pub trait BrdfParameterisation: PartialEq {
    /// Returns the kind of the parameterisation.
    fn kind() -> ParametrisationKind;
}

#[cfg(feature = "fitting")]
use crate::brdf::Bxdf;
pub use clausen::*;
pub use merl::*;
pub use utia::*;
pub use vgonio::*;
pub use yan::*;

#[cfg(feature = "fitting")]
/// A BRDF that can be fitted analytically.
pub trait AnalyticalFit {
    type Params: BrdfParameterisation;

    fn kind(&self) -> MeasuredBrdfKind;

    /// Creates a new analytical BRDF with the same parameterisation as the
    /// given measured BRDF.
    fn new_analytical(
        medium_i: Medium,
        medium_t: Medium,
        spectrum: &[Nanometres],
        params: &Self::Params,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &RefractiveIndexRegistry,
    ) -> Self
    where
        Self: Sized;

    /// Creates a new analytical BRDF based on the parameterisation of the
    /// measured BRDF (self).
    fn new_analytical_from_self(
        &self,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &RefractiveIndexRegistry,
    ) -> Self
    where
        Self: Sized,
    {
        Self::new_analytical(
            self.incident_medium(),
            self.transmitted_medium(),
            self.spectrum(),
            self.params(),
            model,
            iors,
        )
    }

    /// Returns the distance between two BRDFs.
    /// The final distance is calculated based on the given error metric.
    /// The residual error metric is used to calculate the distance between
    /// two single values.
    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: ResidualErrorMetric) -> f64
    where
        Self: Sized;

    /// Returns the distance between two BRDFs with a filter applied to exclude
    /// certain polar angles.
    fn filtered_distance(
        &self,
        other: &Self,
        metric: ErrorMetric,
        rmetric: ResidualErrorMetric,
        limit: Radians,
    ) -> f64
    where
        Self: Sized;

    /// Returns the wavelengths at which the BRDF is measured.
    fn spectrum(&self) -> &[Nanometres];

    /// Returns the samples of the measured BRDF as a one-dimensional array.
    fn samples(&self) -> &[f32];

    /// Returns the parameterisation of the measured BRDF.
    fn params(&self) -> &Self::Params;

    fn incident_medium(&self) -> Medium;

    fn transmitted_medium(&self) -> Medium;

    fn as_any(&self) -> &dyn std::any::Any;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// A BRDF measured either in the real world or in a simulation.
///
/// The BRDF is parameterised by a specific parameterisation. Extra data can be
/// associated with the measured BRDF.
#[derive(Clone, Debug)]
pub struct MeasuredBrdf<P, const N: usize>
where
    P: Clone + Send + Sync + BrdfParameterisation + 'static,
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
}

impl<P, const N: usize> MeasuredBrdf<P, N>
where
    P: Clone + Send + Sync + BrdfParameterisation + PartialEq + 'static,
{
    /// Returns the number of wavelengths of the measured BRDF.
    pub fn n_spectrum(&self) -> usize { self.spectrum.len() }

    /// Returns the number of samples of the measured BRDF.
    pub fn n_samples(&self) -> usize { self.samples.len() }
}

impl<P, const N: usize> PartialEq for MeasuredBrdf<P, N>
where
    P: Clone + Send + Sync + BrdfParameterisation + PartialEq + 'static,
{
    fn eq(&self, other: &Self) -> bool {
        self.origin == other.origin
            && self.incident_medium == other.incident_medium
            && self.transmitted_medium == other.transmitted_medium
            && self.params == other.params
            && self.spectrum == other.spectrum
            && self.samples == other.samples
    }
}

/// An iterator over the snapshots of a measured BRDF.
pub struct BrdfSnapshotIterator<'a, P, const N: usize>
where
    P: Clone + Send + Sync + BrdfParameterisation + PartialEq + 'static,
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
    P: Clone + Send + Sync + BrdfParameterisation + PartialEq + 'static,
{
    fn len(&self) -> usize { self.n_incoming - self.index }
}

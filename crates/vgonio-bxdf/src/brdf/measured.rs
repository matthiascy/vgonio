use base::{medium::Medium, units::Nanometres, MeasuredData};
use jabr::array::DyArr;
use std::fmt::Debug;

#[cfg(feature = "fitting")]
use base::{math, optics::ior::RefractiveIndexRegistry, ErrorMetric};

#[cfg(feature = "fitting")]
macro_rules! impl_analytical_fit_trait {
    ($self:ident) => {
        fn spectrum(&$self) -> &[Nanometres] { $self.spectrum.as_ref() }

        fn samples(&$self) -> &[f32] { $self.samples.as_ref() }

        fn params(&$self) -> &Self::Params { &$self.params }

        fn incident_medium(&$self) -> Medium {
            $self.incident_medium
        }

        fn transmitted_medium(&$self) -> Medium {
            $self.transmitted_medium
        }
    };
}

pub mod clausen;
pub mod merl;
pub mod utia;
pub mod vgonio;

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

use crate::brdf::Bxdf;
pub use clausen::*;
pub use merl::*;
pub use utia::*;
pub use vgonio::*;

/// The kind of the measured BRDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasuredBrdfKind {
    Clausen,
    Merl,
    Utia,
    Vgonio,
    Unknown,
}

#[cfg(feature = "fitting")]
/// A BRDF that can be fitted analytically.
pub trait AnalyticalFit {
    type Params: BrdfParameterisation;

    #[inline]
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
    ) -> Self;

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
    fn distance(&self, other: &Self, metric: ErrorMetric) -> f64 {
        assert_eq!(self.spectrum(), other.spectrum(), "Spectra must be equal!");
        if self.params() != other.params() {
            panic!("Parameterisations must be the same!");
        }
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(self.samples().len() as f64),
            ErrorMetric::Nlls => 0.5,
        };
        self.samples()
            .iter()
            .zip(other.samples().iter())
            .fold(0.0f64, |acc, (a, b)| {
                let diff = *a as f64 - *b as f64;
                acc + math::sqr(diff) * factor
            })
    }

    /// Returns the wavelengths at which the BRDF is measured.
    fn spectrum(&self) -> &[Nanometres];

    /// Returns the samples of the measured BRDF as a one-dimensional array.
    fn samples(&self) -> &[f32];

    /// Returns the parameterisation of the measured BRDF.
    fn params(&self) -> &Self::Params;

    fn incident_medium(&self) -> Medium;

    fn transmitted_medium(&self) -> Medium;
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

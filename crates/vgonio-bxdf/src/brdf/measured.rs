use base::{
    math,
    medium::Medium,
    units::{Nanometres, Rads},
    ErrorMetric,
};
use jabr::array::DyArr;
use std::fmt::Debug;

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

pub trait BrdfParameterisation {
    /// Returns the kind of the parameterisation.
    fn kind() -> ParametrisationKind;
}

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

/// A BRDF measured either in the real world or in a simulation.
///
/// The BRDF is parameterised by a specific parameterisation. Extra data can be
/// associated with the measured BRDF.
#[derive(Clone, Debug)]
pub struct MeasuredBrdf<P, const N: usize>
where
    P: Clone + Send + Sync + BrdfParameterisation + PartialEq + 'static,
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

    /// Computes the distance between two BRDFs.
    pub fn distance(&self, other: &Self, metric: ErrorMetric) -> f64 {
        assert_eq!(self.spectrum, other.spectrum, "Spectra must be equal!");
        if self.params != other.params {
            panic!("Parameterisations must be the same!");
        }
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(self.n_samples() as f64),
            ErrorMetric::Nlls => 0.5,
        };
        self.samples
            .as_slice()
            .iter()
            .zip(other.samples.as_slice())
            .fold(0.0f64, |acc, (a, b)| {
                let diff = *a as f64 - *b as f64;
                acc + math::sqr(diff) * factor
            })
    }
}

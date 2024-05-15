use base::units::Nanometres;
use jabr::array::DyArr;

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

/// A BRDF measured either in the real world or in a simulation.
#[derive(Clone)]
pub struct MeasuredBrdf<P, const N: usize>
where
    P: Clone + Send + Sync + BrdfParameterisation + 'static,
{
    /// The origin of the measured BRDF.
    origin: Origin,
    /// The parameterisation of the measured BRDF.
    params: Box<P>,
    /// Wavelengths at which the BRDF is measured.
    spectrum: DyArr<Nanometres>,
    /// Sampled BRDF data stored in a multi-dimensional array.
    samples: DyArr<f32, N>,
}

impl<P, const N: usize> MeasuredBrdf<P, N>
where
    P: Clone + Send + Sync + BrdfParameterisation + 'static,
{
    /// Returns the number of wavelengths of the measured BRDF.
    pub fn n_spectrum(&self) -> usize { self.spectrum.len() }

    /// Returns the number of samples of the measured BRDF.
    pub fn n_samples(&self) -> usize { self.samples.len() }
}

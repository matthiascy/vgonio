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
pub struct MeasuredBrdf<P: BrdfParameterisation, const N: usize> {
    /// The origin of the measured BRDF.
    origin: Origin,
    /// The parameterisation of the measured BRDF.
    param: Box<P>,
    /// Wavelengths of the measured BRDF.
    wavelengths: DyArr<Nanometres>,
    /// Sampled BRDF data stored in a multi-dimensional array.
    samples: DyArr<f32, N>,
    /// Maximum values of the BRDF for each incident direction and wavelength.
    max_values: DyArr<f32, 2>,
    /// Whether the BRDF is normalised (per incident direction) or not.
    normalised: bool,
}

impl<P: BrdfParameterisation, const N: usize> MeasuredBrdf<P, N> {
    /// Return the origin of the measured BRDF.
    pub fn n_wavelengths(&self) -> usize { self.wavelengths.len() }

    /// Return the total number of samples in the measured BRDF.
    pub fn n_samples(&self) -> usize { self.samples.len() }
}

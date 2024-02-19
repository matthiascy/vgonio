//! Index of refraction.
use crate::{
    medium::MaterialKind,
    units::{nanometres, Nanometres},
};
use std::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter},
};

/// Material's complex refractive index which varies with wavelength of the
/// light. Wavelengths are in *nanometres*; 0.0 means that the refractive index
/// is constant over all the wavelengths.
#[derive(Copy, Clone, PartialEq)]
pub struct RefractiveIndex {
    /// corresponding wavelength in nanometres.
    pub wavelength: Nanometres,

    /// Index of refraction.
    pub eta: f32,

    /// Extinction coefficient.
    pub k: f32,
}

impl Debug for RefractiveIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IOR({}, η={}, κ={})", self.wavelength, self.eta, self.k)
    }
}

impl Display for RefractiveIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) }
}

impl PartialOrd for RefractiveIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.wavelength.partial_cmp(&other.wavelength)
    }
}

impl RefractiveIndex {
    /// Refractive index of vacuum.
    pub const VACUUM: Self = Self {
        wavelength: nanometres!(0.0),
        eta: 1.0,
        k: 0.0,
    };

    /// Creates a new refractive index.
    pub fn new(wavelength: Nanometres, eta: f32, k: f32) -> RefractiveIndex {
        RefractiveIndex { wavelength, eta, k }
    }

    /// Returns the kind of material (insulator or conductor).
    pub fn material_kind(&self) -> MaterialKind {
        if self.k == 0.0 {
            MaterialKind::Insulator
        } else {
            MaterialKind::Conductor
        }
    }

    /// Whether the refractive index represents insulator material.
    pub fn is_dielectric(&self) -> bool { (self.k - 0.0).abs() < f32::EPSILON }

    /// Whether the refractive index represents conductor material.
    pub fn is_conductor(&self) -> bool { !self.is_dielectric() }
}

//! Index of refraction.
use crate::{
    medium::MaterialKind,
    units::{nanometres, Nanometres},
};
use std::{
    cmp::Ordering,
    fmt::{Debug, Display, Formatter},
    ops::Deref,
};

/// Material's complex refractive index which varies with wavelength of the
/// light, mainly used in [`RefractiveIndexRegistry`].
/// Wavelengths are in *nanometres*; 0.0 means that the refractive index
/// is constant over all the wavelengths.
#[derive(Copy, Clone, PartialEq)]
pub struct RefractiveIndexRecord {
    /// corresponding wavelength in nanometres.
    pub wavelength: Nanometres,
    /// Index of refraction data.
    pub ior: Ior,
}

impl Debug for RefractiveIndexRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IOR({}, η={}, κ={})", self.wavelength, self.eta, self.k)
    }
}

impl Display for RefractiveIndexRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) }
}

impl PartialOrd for RefractiveIndexRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.wavelength.partial_cmp(&other.wavelength)
    }
}

impl RefractiveIndexRecord {
    /// Refractive index of vacuum.
    pub const VACUUM: Self = Self {
        wavelength: nanometres!(0.0),
        ior: Ior { eta: 1.0, k: 0.0 },
    };

    /// Creates a new refractive index.
    pub fn new(wavelength: Nanometres, eta: f32, k: f32) -> RefractiveIndexRecord {
        RefractiveIndexRecord {
            wavelength,
            ior: Ior { eta, k },
        }
    }

    /// Returns the kind of material (insulator or conductor).
    pub fn material_kind(&self) -> MaterialKind {
        if self.k == 0.0 {
            MaterialKind::Insulator
        } else {
            MaterialKind::Conductor
        }
    }
}

impl Deref for RefractiveIndexRecord {
    type Target = Ior;

    fn deref(&self) -> &Self::Target { &self.ior }
}

/// Complex index of refraction without wavelength information.
#[derive(Copy, Clone, PartialEq)]
pub struct Ior {
    /// Index of refraction.
    pub eta: f32,
    /// Extinction coefficient.
    pub k: f32,
}

impl Debug for Ior {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IOR(η={}, κ={})", self.eta, self.k)
    }
}

impl Display for Ior {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "η={}, κ={}", self.eta, self.k)
    }
}

impl Ior {
    /// Checks whether the refractive index represents insulator material.
    pub fn is_dielectric(&self) -> bool { (self.k - 0.0).abs() < f32::EPSILON }

    /// Checks whether the refractive index represents conductor material.
    pub fn is_conductor(&self) -> bool { !self.is_dielectric() }
}

//! Index of refraction.

use crate::{MaterialKind, Medium};
use base::{
    math,
    units::{nanometres, Length, LengthMeasurement, Nanometres},
};
use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    ops::Deref,
    path::Path,
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

/// Refractive index database.
#[derive(Debug)]
pub struct RefractiveIndexDatabase(pub(crate) HashMap<Medium, Vec<RefractiveIndex>>);

impl Default for RefractiveIndexDatabase {
    fn default() -> Self { Self::new() }
}

impl Deref for RefractiveIndexDatabase {
    type Target = HashMap<Medium, Vec<RefractiveIndex>>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl RefractiveIndexDatabase {
    /// Create an empty database.
    pub fn new() -> RefractiveIndexDatabase { RefractiveIndexDatabase(HashMap::new()) }

    /// Returns the refractive index of the given medium at the given wavelength
    /// (in nanometres).
    pub fn ior_of(&self, medium: Medium, wavelength: Nanometres) -> Option<RefractiveIndex> {
        let refractive_indices = self
            .get(&medium)
            .unwrap_or_else(|| panic!("unknown medium {:?}", medium));
        // Search for the position of the first wavelength equal or greater than the
        // given one in refractive indices.
        let i = refractive_indices
            .iter()
            .position(|ior| ior.wavelength >= wavelength)
            .unwrap();
        let ior_after = refractive_indices[i];
        // If the first wavelength is equal to the given one, return it.
        if math::ulp_eq(ior_after.wavelength.value(), wavelength.value()) {
            Some(ior_after)
        } else {
            // Otherwise, interpolate between the two closest refractive indices.
            let ior_before = if i == 0 {
                refractive_indices[0]
            } else {
                refractive_indices[i - 1]
            };
            let diff_eta = ior_after.eta - ior_before.eta;
            let diff_k = ior_after.k - ior_before.k;
            let t = (wavelength - ior_before.wavelength)
                / (ior_after.wavelength - ior_before.wavelength);
            Some(RefractiveIndex {
                wavelength,
                eta: ior_before.eta + t * diff_eta,
                k: ior_before.k + t * diff_k,
            })
        }
    }

    /// Returns the refractive index of the given medium at the given spectrum
    /// (in nanometers).
    pub fn ior_of_spectrum<A: LengthMeasurement>(
        &self,
        medium: Medium,
        wavelengths: &[Length<A>],
    ) -> Option<Vec<RefractiveIndex>> {
        wavelengths
            .iter()
            .map(|wavelength| self.ior_of(medium, wavelength.in_nanometres()))
            .collect()
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

    /// Read a csv file and return a vector of refractive indices.
    /// File format: "wavelength, µm", "eta", "k"
    pub(crate) fn read_iors_from_file(path: &Path) -> Option<Vec<RefractiveIndex>> {
        std::fs::File::open(path)
            .map(|f| {
                let mut rdr = csv::Reader::from_reader(f);

                // Read the header (the first line of the file) to get the unit of the
                // wavelength.
                let mut coefficient = 1.0f32;

                let mut is_conductor = false;

                if let Ok(header) = rdr.headers() {
                    is_conductor = header.len() == 3;
                    match header.get(0).unwrap().split(' ').last().unwrap() {
                        "nm" => coefficient = 1.0,
                        "µm" => coefficient = 1e3,
                        &_ => coefficient = 1.0,
                    }
                }

                if is_conductor {
                    rdr.records()
                        .filter_map(|ior_record| match ior_record {
                            Ok(record) => {
                                let wavelength = record[0].parse::<f32>().unwrap() * coefficient;
                                let eta = record[1].parse::<f32>().unwrap();
                                let k = record[2].parse::<f32>().unwrap();
                                Some(RefractiveIndex::new(wavelength.into(), eta, k))
                            }
                            Err(_) => None,
                        })
                        .collect()
                } else {
                    rdr.records()
                        .filter_map(|ior_record| match ior_record {
                            Ok(record) => {
                                let wavelength = record[0].parse::<f32>().unwrap() * coefficient;
                                let eta = record[1].parse::<f32>().unwrap();
                                Some(RefractiveIndex::new(wavelength.into(), eta, 0.0))
                            }
                            Err(_) => None,
                        })
                        .collect()
                }
            })
            .ok()
    }
}

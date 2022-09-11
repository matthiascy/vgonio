use crate::acq::Medium;
use std::{cmp::Ordering, collections::HashMap, path::Path};

// todo: merge ior db into vgonio db

/// Material's complex refractive index which varies with wavelength of the
/// light. Wavelengths are in *nanometres*; 0.0 means that the refractive index
/// is constant over all the wavelengths.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Ior {
    /// corresponding wavelength in nanometres.
    pub wavelength: f32,

    /// Index of refraction.
    pub eta: f32,

    /// Extinction coefficient.
    pub k: f32,
}

impl PartialOrd for Ior {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.wavelength.partial_cmp(&other.wavelength)
    }
}

/// Refractive index database.
#[derive(Debug)]
pub struct IorDb(pub(crate) HashMap<Medium, Vec<Ior>>);

impl Default for IorDb {
    fn default() -> Self { Self::new() }
}

impl IorDb {
    /// Create an empty database.
    pub fn new() -> IorDb { IorDb(HashMap::new()) }

    /// Returns the refractive index of the given medium at the given wavelength
    /// (in nanometres).
    pub fn ior_of(&self, medium: Medium, wavelength: f32) -> Option<Ior> {
        let refractive_indices = self.0.get(&medium).expect("unknown medium");
        // Search for the position of the first wavelength equal or greater than the
        // given one in refractive indices.
        let i = refractive_indices
            .iter()
            .position(|ior| ior.wavelength >= wavelength)
            .unwrap();
        let ior_after = refractive_indices[i];

        // If the first wavelength is equal to the given one, return it.
        if (ior_after.wavelength - wavelength).abs() < f32::EPSILON {
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
            Some(Ior {
                wavelength,
                eta: ior_before.eta + t * diff_eta,
                k: ior_before.k + t * diff_k,
            })
        }
    }

    /// Returns the refractive index of the given medium at the given spectrum
    /// (in nanometers).
    pub fn ior_of_spectrum(&self, medium: Medium, wavelengths: &[f32]) -> Option<Vec<Ior>> {
        wavelengths
            .iter()
            .map(|wavelength| self.ior_of(medium, *wavelength))
            .collect()
    }
}

impl Ior {
    /// Refractive index of vacuum.
    pub const VACUUM: Self = Self {
        wavelength: 0.0,
        eta: 1.0,
        k: 0.0,
    };

    /// Refractive index of air.
    pub const AIR: Self = Self {
        wavelength: 0.0,
        eta: 1.00029,
        k: 0.0,
    };

    /// Creates a new refractive index.
    pub fn new(wavelength: f32, eta: f32, k: f32) -> Ior { Ior { wavelength, eta, k } }

    /// Whether the refractive index represents dielectric material.
    pub fn is_dielectric(&self) -> bool { (self.k - 0.0).abs() < f32::EPSILON }

    /// Read a csv file and return a vector of refractive indices.
    /// File format: "wavelength, µm", "eta", "k"
    pub(crate) fn read_iors_from_file(path: &Path) -> Option<Vec<Ior>> {
        std::fs::File::open(path)
            .map(|f| {
                let mut rdr = csv::Reader::from_reader(f);

                // Read the header (the first line of the file) to get the unit of the
                // wavelength.
                let mut coefficient = 1.0f32;

                if let Ok(header) = rdr.headers() {
                    match header.get(0).unwrap().split(' ').last().unwrap() {
                        "nm" => coefficient = 1.0,
                        "µm" => coefficient = 1e3,
                        &_ => coefficient = 1.0,
                    }
                }

                rdr.records()
                    .into_iter()
                    .filter_map(|ior_record| match ior_record {
                        Ok(record) => {
                            let wavelength = record[0].parse::<f32>().unwrap() * coefficient;
                            let eta = record[1].parse::<f32>().unwrap();
                            let k = record[2].parse::<f32>().unwrap();
                            Some(Ior::new(wavelength, eta, k))
                        }
                        Err(_) => None,
                    })
                    .collect()
            })
            .ok()
    }
}

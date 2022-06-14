use crate::acq::Medium;
use crate::app::VgonioConfig;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

/// Material's complex refractive index which varies with wavelength of the
/// light. Wavelengths are in *nanometres*; 0.0 means that the refractive index
/// is constant over all the wavelengths.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RefractiveIndex {
    /// corresponding wavelength in nanometres.
    pub wavelength: f32,

    /// Index of refraction.
    pub eta: f32,

    /// Extinction coefficient.
    pub k: f32,
}

impl PartialOrd for RefractiveIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.wavelength.partial_cmp(&other.wavelength)
    }
}

pub struct RefractiveIndexDatabase(HashMap<Medium, Vec<RefractiveIndex>>);

impl Default for RefractiveIndexDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl RefractiveIndexDatabase {
    pub fn new() -> RefractiveIndexDatabase {
        RefractiveIndexDatabase(HashMap::new())
    }

    /// Returns the refractive index of the given medium at the given wavelength
    /// (in nanometres).
    pub fn refractive_index_of(&self, medium: Medium, wavelength: f32) -> Option<RefractiveIndex> {
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
            let t = (wavelength - ior_before.wavelength) / (ior_after.wavelength - ior_before.wavelength);
            Some(RefractiveIndex {
                wavelength,
                eta: ior_before.eta + t * diff_eta,
                k: ior_before.k + t * diff_k,
            })
        }
    }

    pub fn ior_of_spectrum(&self, medium: Medium, wavelengths: &[f32]) -> Option<Vec<RefractiveIndex>> {
        wavelengths
            .iter()
            .map(|wavelength| self.refractive_index_of(medium, *wavelength))
            .collect()
    }

    /// Load the refractive index database from the paths specified in the
    /// config.
    pub fn load_from_config_dirs(config: &VgonioConfig) -> Self {
        let mut database = RefractiveIndexDatabase::new();
        let default_path = config.data_files_dir.join("ior");
        let user_path = config.user_config.data_files_dir.join("ior");

        // First load refractive indices from `VgonioConfig::data_files_dir`.
        if default_path.exists() {
            // Load one by one the files in the directory.
            let n = Self::load_refractive_indices(&mut database, &default_path);
            log::debug!("Loaded {} ior files from {:?}", n, default_path);
        }

        // Second load refractive indices from
        // `VgonioConfig::user_config::data_files_dir`.
        if user_path.exists() {
            let n = Self::load_refractive_indices(&mut database, &user_path);
            log::debug!("Loaded {} ior files from {:?}", n, user_path);
        }

        database
    }

    /// Load the refractive index database from the given path.
    /// Returns the number of files loaded.
    fn load_refractive_indices(ior_db: &mut RefractiveIndexDatabase, path: &Path) -> u32 {
        let mut n_files = 0;
        if path.is_file() {
            log::debug!("Loading refractive index database from {:?}", path);
            let medium =
                Medium::from_str(path.file_name().unwrap().to_str().unwrap().split('_').next().unwrap()).unwrap();
            let refractive_indices = RefractiveIndex::read_iors_from_file(path).unwrap();
            let iors = ior_db.0.entry(medium).or_insert(Vec::new());
            for ior in refractive_indices {
                if !iors.contains(&ior) {
                    iors.push(ior);
                }
            }
            iors.sort_by(|a, b| a.wavelength.partial_cmp(&b.wavelength).unwrap());
            n_files += 1;
        } else if path.is_dir() {
            for entry in path.read_dir().unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                n_files += RefractiveIndexDatabase::load_refractive_indices(ior_db, &path);
            }
        }

        n_files
    }
}

impl RefractiveIndex {
    pub const VACUUM: Self = Self {
        wavelength: 0.0,
        eta: 1.0,
        k: 0.0,
    };

    pub const AIR: Self = Self {
        wavelength: 0.0,
        eta: 1.00029,
        k: 0.0,
    };

    pub fn new(wavelength: f32, eta: f32, k: f32) -> RefractiveIndex {
        RefractiveIndex { wavelength, eta, k }
    }

    pub fn is_dielectric(&self) -> bool {
        (self.k - 0.0).abs() < f32::EPSILON
    }

    /// Read a csv file and return a vector of refractive indices.
    /// File format: "wavelength, µm", "eta", "k"
    pub(crate) fn read_iors_from_file(path: &Path) -> Option<Vec<RefractiveIndex>> {
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
                            Some(RefractiveIndex::new(wavelength, eta, k))
                        }
                        Err(_) => None,
                    })
                    .collect()
            })
            .ok()
    }
}

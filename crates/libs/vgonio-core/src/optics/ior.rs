//! Index of refraction.
use crate::{
    asset, math,
    res::{Asset, AssetLoader, AssetTypeId, AssetTypeRegistry, Error},
    units::{nanometres, Length, LengthMeasurement, Nanometres},
    utils::medium::{MaterialKind, Medium},
};
use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    ops::{Deref, DerefMut},
    path::PathBuf,
    str::FromStr,
};

use std::path::Path;

/// Refractive index database.
#[derive(Debug, Clone)]
pub struct IorReg(pub(crate) HashMap<Medium, Vec<IorRecord>>);

impl Default for IorReg {
    fn default() -> Self { Self::new() }
}

impl Deref for IorReg {
    type Target = HashMap<Medium, Vec<IorRecord>>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for IorReg {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

asset!(IorReg, "IorReg");

/// Loader for refractive index database.
pub struct IorRegLoader {
    /// System directory.
    sys_dir: Option<PathBuf>,
    /// User directory.
    usr_dir: Option<PathBuf>,
    /// Excluded files.
    excluded: Option<Vec<String>>,
}

impl IorRegLoader {
    /// Creates a new refractive index database loader.
    ///
    /// # Arguments
    ///
    /// * `sys_dir` - The base system directory used to resolve the path when
    ///   the path is in form `sys://`. The actual path is resolved to
    ///   `sys_dir/ior/path`.
    /// * `usr_dir` - The base user directory used to resolve the path when the
    ///   path is in form `user://`. The actual path is resolved to
    ///   `usr_dir/ior/path`.
    /// * `excluded` - The list of excluded files.
    pub fn new(
        sys_dir: Option<&Path>,
        usr_dir: Option<&Path>,
        excluded: Option<Vec<String>>,
    ) -> Self {
        Self {
            sys_dir: sys_dir.map(|path| path.to_path_buf().join("ior")),
            usr_dir: usr_dir.map(|path| path.to_path_buf().join("ior")),
            excluded,
        }
    }
}

impl AssetLoader for IorRegLoader {
    fn asset_type(&self) -> &'static str { IorReg::asset_type() }

    fn asset_type_id(&self) -> AssetTypeId { IorReg::asset_type_id() }

    fn load(&self, path: Option<&Path>) -> Result<Box<dyn Asset>, Error> {
        let own_excluded = self
            .excluded
            .as_ref()
            .and_then(|ss| Some(ss.iter().map(|s| s.as_str()).collect::<Vec<_>>()));
        let excluded = own_excluded.as_deref().unwrap_or(&[]);
        match path {
            Some(path) => {
                let mut ior_reg = IorReg::new();
                let n = ior_reg.load_from_path(path, excluded)?;
                log::debug!("  Loaded {} ior files from {:?}", n, path);
                Ok(Box::new(ior_reg))
            },
            None => {
                log::debug!("Loading refractive index database from default paths ...");
                log::debug!("  -- sys_dir: {:?}", self.sys_dir);
                log::debug!("  -- usr_dir: {:?}", self.usr_dir);

                if self.sys_dir.is_none() {
                    log::debug!(
                        "  Refractive index database not found at sys dir: {:?}",
                        self.sys_dir
                    );
                }

                if self.usr_dir.is_none() {
                    log::debug!(
                        "  Refractive index database not found at user dir: {:?}",
                        self.usr_dir
                    );
                }

                if self.usr_dir.is_none() && self.sys_dir.is_none() {
                    log::error!("No path specified to load refractive index database");
                    todo!("return an error");
                }

                let mut ior_reg = IorReg::new();
                let n = ior_reg.load_from_path(self.sys_dir.as_ref().unwrap(), excluded)?;
                log::debug!("  Loaded {} ior files from {:?}", n, self.sys_dir);
                let n = ior_reg.load_from_path(self.usr_dir.as_ref().unwrap(), excluded)?;
                log::debug!("  Loaded {} ior files from {:?}", n, self.usr_dir);
                Ok(Box::new(ior_reg))
            },
        }
    }
}

impl IorReg {
    /// Create an empty database.
    pub fn new() -> IorReg { IorReg(HashMap::new()) }

    /// Returns the refractive index of the given medium at the given wavelength
    /// (in nanometres).
    pub fn ior_of(&self, medium: Medium, wavelength: Nanometres) -> Option<IorRecord> {
        if medium == Medium::Vacuum {
            return Some(IorRecord::VACUUM);
        }
        let refractive_indices = self
            .get(&medium)
            .unwrap_or_else(|| panic!("unknown medium {:?}", medium));
        // Search for the position of the first wavelength equal or greater than the
        // given one in refractive indices.
        let i = refractive_indices
            .iter()
            .position(|ior| ior.wavelength >= wavelength)
            .expect(&format!(
                "Medium: {:?}, IOR {} not found in the database",
                medium, wavelength
            ));
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
            Some(IorRecord {
                wavelength,
                ior: Ior {
                    eta: ior_before.eta + t * diff_eta,
                    k: ior_before.k + t * diff_k,
                },
            })
        }
    }

    /// Returns the refractive index of the given medium at the given spectrum
    /// (in nanometers).
    pub fn ior_of_spectrum<A: LengthMeasurement>(
        &self,
        medium: Medium,
        wavelengths: &[Length<A>],
    ) -> Option<Box<[Ior]>> {
        wavelengths
            .iter()
            .map(|wavelength| {
                self.ior_of(medium, wavelength.in_nanometres())
                    .map(|ior| ior.ior)
            })
            .collect::<Option<Box<[_]>>>()
    }

    /// Loads refractive indices from the given path.
    pub fn load_from_path(&mut self, path: &Path, excluded: &[&str]) -> Result<u32, Error> {
        let mut n_loaded = 0;
        if path.is_file() {
            let filename = path.file_name().unwrap().to_str().unwrap();
            if excluded.contains(&filename) {
                log::debug!("  -- excluded: {}", filename);
                return Ok(0);
            }
            let medium = Medium::from_str(
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .split('_')
                    .next()
                    .unwrap(),
            )
            .unwrap();
            // TODO: print error if read_iors_from_file returns None as we don't want to
            // interrupt the loading process of other files.
            let loaded_iors = IorReg::read_iors_from_file(path).unwrap();
            let iors = self.0.entry(medium).or_default();
            for ior in loaded_iors.iter() {
                if !iors.contains(ior) {
                    iors.push(*ior);
                }
            }
            iors.sort_by(|a, b| a.wavelength.partial_cmp(&b.wavelength).unwrap());
            n_loaded += 1;
        } else if path.is_dir() {
            for entry in path.read_dir()? {
                let entry = entry?;
                let path = entry.path();
                n_loaded += self.load_from_path(&path, excluded)?;
            }
        } else {
            todo!("return an error");
        }
        Ok(n_loaded)
    }

    /// Read a csv file and return a vector of refractive indices.
    /// File format: "wavelength, µm", "eta", "k"
    pub fn read_iors_from_file(path: &Path) -> Option<Box<[IorRecord]>> {
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
                                Some(IorRecord::new(wavelength.into(), eta, k))
                            },
                            Err(_) => None,
                        })
                        .collect::<Box<_>>()
                } else {
                    rdr.records()
                        .filter_map(|ior_record| match ior_record {
                            Ok(record) => {
                                let wavelength = record[0].parse::<f32>().unwrap() * coefficient;
                                let eta = record[1].parse::<f32>().unwrap();
                                Some(IorRecord::new(wavelength.into(), eta, 0.0))
                            },
                            Err(_) => None,
                        })
                        .collect::<Box<_>>()
                }
            })
            .ok()
    }
}

/// Material's complex refractive index, which varies with the wavelength of the
/// light, mainly used in [`RefractiveIndexRegistry`]. Wavelengths are in
/// *nanometres*.
///
/// Wavelength 0.0 means that the refractive index is constant over all the
/// wavelengths.
#[derive(Copy, Clone, PartialEq)]
pub struct IorRecord {
    /// corresponding wavelength in nanometres.
    pub wavelength: Nanometres,
    /// Index of refraction data.
    pub ior: Ior,
}

impl Debug for IorRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IOR({}, η={}, κ={})", self.wavelength, self.eta, self.k)
    }
}

impl Display for IorRecord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) }
}

impl PartialOrd for IorRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.wavelength.partial_cmp(&other.wavelength)
    }
}

impl IorRecord {
    /// Refractive index of vacuum.
    pub const VACUUM: Self = Self {
        wavelength: nanometres!(0.0),
        ior: Ior { eta: 1.0, k: 0.0 },
    };

    /// Creates a new refractive index.
    pub fn new(wavelength: Nanometres, eta: f32, k: f32) -> IorRecord {
        IorRecord {
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

impl Deref for IorRecord {
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

//! Index of refraction: runtime data model and registry.
//!
//! On-disk datasets are vgonio-native `*.ior.ron` files (see [`dto`]); loading
//! is handled by [`loader::IorRegLoader`]. Wavelengths in the runtime model are
//! in *nanometres*; refractiveindex.info data (which uses micrometres) is converted
//! on the way in.

pub mod dto;
pub mod formula;
pub mod loader;

pub use dto::*;
pub use formula::DispersionFormula;
pub use loader::IorRegLoader;

use crate::{
    asset, math,
    res::AssetTypeId,
    units::{nanometres, Length, LengthMeasurement, Nanometres},
    utils::medium::MediumId,
};
use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    ops::{Deref, DerefMut},
};

use std::path::Path;

/// Complex index of refraction without wavelength information.
#[derive(Copy, Clone, PartialEq)]
pub struct Ior {
    /// Index of refraction (real part).
    pub eta: f32,
    /// Extinction coefficient (imaginary part).
    pub k: f32,
}

impl Ior {
    /// Creates a new index of refraction for insulator material.
    pub fn new_dielectric(eta: f32) -> Ior { Ior { eta, k: 0.0 } }

    /// Creates a new index of refraction for conductor material.
    pub fn new_conductor(eta: f32, k: f32) -> Ior { Ior { eta, k } }

    /// Checks whether the refractive index represents insulator material.
    pub fn is_dielectric(&self) -> bool { (self.k - 0.0).abs() < f32::EPSILON }

    /// Checks whether the refractive index represents conductor material.
    pub fn is_conductor(&self) -> bool { !self.is_dielectric() }
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

/// One (wavelength, η, κ) tabulated sample. Wavelength is in nanometres.
///
/// Wavelength 0.0 means that the refractive index is constant over the spectrum.
#[derive(Copy, Clone, PartialEq)]
pub struct IorRecord {
    /// corresponding wavelength in nanometres.
    pub wavelength: Nanometres,
    /// Index of refraction data.
    pub ior: Ior,
}

impl IorRecord {
    /// Refractive index of vacuum (constant over the spectrum).
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
}

impl Deref for IorRecord {
    type Target = Ior;

    fn deref(&self) -> &Self::Target { &self.ior }
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

/// The payload of an [`IorDataset`]: either tabulated samples or a dispersion formula.
#[derive(Debug, Clone, PartialEq)]
pub enum IorData {
    /// λ -> (η,κ) samples, sorted ascending by wavelength. κ may be all-zero.
    /// Lookup interpolates between neighbours; λ outside the sampled range => `None`.
    Tabulated(Box<[IorRecord]>),
    /// A dispersion formula for η over `range`, with an optional separately
    /// tabulated κ table (sorted ascending). κ is 0 when `k` is `None`. A lookup
    /// returns `None` when λ is outside `range`, or when `k` is `Some` and λ is
    /// outside the κ table's own range.
    Dispersion {
        /// The formula yielding η(λ).
        formula: DispersionFormula,
        /// Valid wavelength range [lo, hi] (nanometres).
        range: (Nanometres, Nanometres),
        /// Optional κ samples (wavelength_nm, κ), sorted ascending.
        k: Option<Box<[(Nanometres, f32)]>>,
    },
}

/// A complete refractive-index dataset for one medium, from one source.
#[derive(Debug, Clone, PartialEq)]
pub struct IorDataset {
    /// The medium this dataset describes.
    pub medium: MediumId,
    /// Short source label (e.g. `"McPeak2015"`), also the file-name suffix.
    pub name: String,
    /// Plain-text upstream citation.
    pub reference: String,
    /// Plain-text upstream comments.
    pub comments: String,
    /// The η/κ data.
    pub data: IorData,
}

impl IorDataset {
    /// Looks up η/κ at the given wavelength (nanometres). `None` if out of range.
    pub fn ior_at(&self, wavelength: Nanometres) -> Option<IorRecord> {
        match &self.data {
            IorData::Tabulated(samples) => interp_tabulated(samples, wavelength),
            IorData::Dispersion { formula, range, k } => {
                if wavelength < range.0 || wavelength > range.1 {
                    return None;
                }
                let eta = formula.eval_eta(wavelength.value() as f64 / 1000.0) as f32;
                let k = match k {
                    None => 0.0,
                    Some(table) => interp_k(table, wavelength)?,
                };
                Some(IorRecord {
                    wavelength,
                    ior: Ior { eta, k },
                })
            },
        }
    }

    /// Reads and validates a `*.ior.ron` file into a runtime dataset.
    ///
    /// Returns an error if the file cannot be read or parsed, or if the parsed data is invalid.
    pub fn read(path: &Path) -> Result<IorDataset, IorFileError> {
        let dto = IorDatasetDto::read(path)?;
        dto.into_runtime(&path.display().to_string())
    }

    /// Serialises a runtime dataset to a `*.ior.ron` file (pretty-printed).
    pub fn write(&self, path: &Path) -> Result<(), IorFileError> {
        let dto = IorDatasetDto::from_runtime(self);
        dto.write(path)
    }
}

/// Linear interpolation in a sorted `IorRecord` table; `None` if `w` is outside it.
fn interp_tabulated(samples: &[IorRecord], w: Nanometres) -> Option<IorRecord> {
    if samples.is_empty() || w < samples[0].wavelength || w > samples[samples.len() - 1].wavelength
    {
        return None;
    }
    let i = samples.iter().position(|r| r.wavelength >= w).unwrap();
    let after = samples[i];
    if math::ulp_eq(after.wavelength.value(), w.value()) || i == 0 {
        return Some(IorRecord {
            wavelength: w,
            ior: after.ior,
        });
    }
    let before = samples[i - 1];
    let t = (w - before.wavelength) / (after.wavelength - before.wavelength);
    Some(IorRecord {
        wavelength: w,
        ior: Ior {
            eta: before.eta + t * (after.eta - before.eta),
            k: before.k + t * (after.k - before.k),
        },
    })
}

/// Linear interpolation in a sorted `(λ_nm, κ)` table; `None` if `w` is outside it.
fn interp_k(table: &[(Nanometres, f32)], w: Nanometres) -> Option<f32> {
    if table.is_empty() || w < table[0].0 || w > table[table.len() - 1].0 {
        return None;
    }
    let i = table.iter().position(|&(lw, _)| lw >= w).unwrap();
    let (aw, ak) = table[i];
    if math::ulp_eq(aw.value(), w.value()) || i == 0 {
        return Some(ak);
    }
    let (bw, bk) = table[i - 1];
    let t = (w - bw) / (aw - bw);
    Some(bk + t * (ak - bk))
}

asset!(IorReg, "IorReg");

/// In-memory refractive-index registry: one chosen dataset per medium.
#[derive(Debug, Clone, Default)]
pub struct IorReg(pub(crate) HashMap<MediumId, IorDataset>);

impl Deref for IorReg {
    type Target = HashMap<MediumId, IorDataset>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for IorReg {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl IorReg {
    /// Create an empty database.
    pub fn new() -> IorReg { IorReg(HashMap::new()) }

    /// Refractive index of `medium` at `wavelength` (nanometres).
    ///
    /// Returns `None` if the medium is unknown to the registry or the wavelength
    /// is outside the dataset's valid range. `MediumId::Vacuum` always returns
    /// [`IorRecord::VACUUM`].
    pub fn ior_of(&self, medium: MediumId, wavelength: Nanometres) -> Option<IorRecord> {
        if medium == MediumId::VACUUM {
            return Some(IorRecord::VACUUM);
        }
        self.0
            .get(&medium)
            .and_then(|dataset| dataset.ior_at(wavelength))
    }

    /// Refractive index of `medium` over a spectrum. `None` if any wavelength is
    /// out of range or the medium is unknown.
    pub fn ior_of_spectrum<A: LengthMeasurement>(
        &self,
        medium: MediumId,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::nm;

    fn tab(samples: &[(f32, f32, f32)]) -> IorDataset {
        IorDataset {
            medium: MediumId::AL,
            name: "test".into(),
            reference: String::new(),
            comments: String::new(),
            data: IorData::Tabulated(
                samples
                    .iter()
                    .map(|&(w, e, k)| IorRecord::new(nm!(w), e, k))
                    .collect(),
            ),
        }
    }

    #[test]
    fn vacuum_is_special() {
        let reg = IorReg::new();
        assert_eq!(
            reg.ior_of(MediumId::VACUUM, nm!(500.0)),
            Some(IorRecord::VACUUM)
        );
    }

    #[test]
    fn unknown_medium_is_none() {
        let reg = IorReg::new();
        assert_eq!(reg.ior_of(MediumId::AL, nm!(500.0)), None);
    }

    #[test]
    fn tabulated_exact_and_interp_and_out_of_range() {
        let mut reg = IorReg::new();
        reg.insert(MediumId::AL, tab(&[(400.0, 1.0, 5.0), (500.0, 2.0, 7.0)]));
        // exact
        assert_eq!(reg.ior_of(MediumId::AL, nm!(400.0)).unwrap().eta, 1.0);
        // midpoint interpolation
        let mid = reg.ior_of(MediumId::AL, nm!(450.0)).unwrap();
        assert!((mid.eta - 1.5).abs() < 1e-6 && (mid.k - 6.0).abs() < 1e-6);
        // below / above range ⇒ None (no extrapolation)
        assert_eq!(reg.ior_of(MediumId::AL, nm!(399.0)), None);
        assert_eq!(reg.ior_of(MediumId::AL, nm!(501.0)), None);
    }

    #[test]
    fn dispersion_eta_and_k_rules() {
        let ds = IorDataset {
            medium: MediumId::PVC,
            name: "test".into(),
            reference: String::new(),
            comments: String::new(),
            data: IorData::Dispersion {
                formula: DispersionFormula::Cauchy {
                    c0: 1.5,
                    terms: vec![],
                },
                range: (nm!(400.0), nm!(800.0)),
                k: Some(Box::from([(nm!(400.0), 0.1f32), (nm!(800.0), 0.3f32)])),
            },
        };
        let mut reg = IorReg::new();
        reg.insert(MediumId::PVC, ds);
        // η from the formula, κ interpolated
        let m = reg.ior_of(MediumId::PVC, nm!(600.0)).unwrap();
        assert!((m.eta - 1.5).abs() < 1e-6 && (m.k - 0.2).abs() < 1e-6);
        // outside formula range ⇒ None
        assert_eq!(reg.ior_of(MediumId::PVC, nm!(900.0)), None);
        // κ absent ⇒ 0
        let mut reg2 = IorReg::new();
        reg2.insert(
            MediumId::PVC,
            IorDataset {
                medium: MediumId::PVC,
                name: "t".into(),
                reference: String::new(),
                comments: String::new(),
                data: IorData::Dispersion {
                    formula: DispersionFormula::Cauchy {
                        c0: 1.5,
                        terms: vec![],
                    },
                    range: (nm!(400.0), nm!(800.0)),
                    k: None,
                },
            },
        );
        assert_eq!(reg2.ior_of(MediumId::PVC, nm!(600.0)).unwrap().k, 0.0);
    }

    #[test]
    fn spectrum_returns_none_if_any_out_of_range() {
        let mut reg = IorReg::new();
        reg.insert(MediumId::AL, tab(&[(400.0, 1.0, 5.0), (500.0, 2.0, 7.0)]));
        assert!(reg
            .ior_of_spectrum(MediumId::AL, &[nm!(420.0), nm!(480.0)])
            .is_some());
        assert!(reg
            .ior_of_spectrum(MediumId::AL, &[nm!(420.0), nm!(900.0)])
            .is_none());
    }
}

//! On-disk representation of an IOR dataset: a versioned RON DTO (`*.ior.ron`),
//! its conversion to/from the runtime [`super::IorDataset`], the `sources.toml`
//! manifest model, and a reader for the legacy `*.csv` files.

use super::{DispersionFormula, IorData, IorDataset, IorRecord};
use crate::{units::nanometres, utils::medium::Medium};
use serde::{Deserialize, Serialize};
use std::{path::Path, str::FromStr};

/// Current `*.ior.ron` version.
pub const IOR_DATASET_SCHEMA_VERSION: u32 = 1;

/// Errors from reading/writing/converting IOR dataset files.
#[derive(Debug, thiserror::Error)]
pub enum IorFileError {
    /// I/O error.
    #[error("I/O error for {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// RON (de)serialization error.
    #[error("RON error for {path}: {source}")]
    Ron {
        path: String,
        #[source]
        source: ron::Error,
    },
    /// TOML (de)serialization error.
    #[error("TOML error: {0}")]
    Toml(String),
    /// CSV parsing error (legacy import).
    #[error("CSV error for {path}: {msg}")]
    Csv { path: String, msg: String },
    /// Unsupported schema version.
    #[error(
        "{path}: unsupported ior dataset schema_version {found} (this build expects {expected})"
    )]
    SchemaVersion {
        path: String,
        found: u32,
        expected: u32,
    },
    /// Unknown medium string.
    #[error("{0}: unknown medium {1:?}")]
    UnknownMedium(String, String),
    /// `medium`/`name` disagreement between the file body, its name, and the manifest.
    #[error("{0}: {1}")]
    Inconsistent(String, String),
}

/// On-disk dataset DTO. Wavelengths are in nanometres; numbers are `f64` on
/// disk for losslessness and narrowed to `f32` in the runtime model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IorDatasetDto {
    /// Schema version; must equal [`IOR_DATASET_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Lowercase medium name (see [`Medium::name`]).
    pub medium: String,
    /// Source label (e.g. `"McPeak2015"`).
    pub name: String,
    /// Plain-text citation.
    #[serde(default)]
    pub reference: String,
    /// Plain-text comments.
    #[serde(default)]
    pub comments: String,
    /// The η/κ payload.
    pub data: IorDataDto,
}

/// On-disk payload variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IorDataDto {
    /// Rows of `[λ_nm, η, κ]`, ascending by λ.
    Tabulated(Vec<[f64; 3]>),
    /// Formula for η over `range_nm`, with optional κ rows `[λ_nm, κ]`.
    Dispersion {
        /// The dispersion formula (the runtime enum, serde-clean).
        formula: DispersionFormula,
        /// `[lo, hi]` in nanometres.
        range_nm: [f64; 2],
        /// Optional κ table; `None` => κ = 0.
        #[serde(default)]
        k: Option<Vec<[f64; 2]>>,
    },
}

impl IorDatasetDto {
    /// Builds a DTO from a runtime dataset.
    pub fn from_runtime(ds: &IorDataset) -> Self {
        let data = match &ds.data {
            IorData::Tabulated(s) => IorDataDto::Tabulated(
                s.iter()
                    .map(|r| [r.wavelength.value() as f64, r.eta as f64, r.k as f64])
                    .collect(),
            ),
            IorData::Dispersion { formula, range, k } => IorDataDto::Dispersion {
                formula: formula.clone(),
                range_nm: [range.0.value() as f64, range.1.value() as f64],
                k: k.as_ref().map(|t| {
                    t.iter()
                        .map(|&(w, kv)| [w.value() as f64, kv as f64])
                        .collect()
                }),
            },
        };
        IorDatasetDto {
            schema_version: IOR_DATASET_SCHEMA_VERSION,
            medium: ds.medium.name().to_string(),
            name: ds.name.clone(),
            reference: ds.reference.clone(),
            comments: ds.comments.clone(),
            data,
        }
    }

    /// Converts to a runtime dataset, validating `schema_version` and `medium`.
    /// `path` is used only for error messages.
    pub fn into_runtime(self, path: &str) -> Result<IorDataset, IorFileError> {
        if self.schema_version != IOR_DATASET_SCHEMA_VERSION {
            return Err(IorFileError::SchemaVersion {
                path: path.into(),
                found: self.schema_version,
                expected: IOR_DATASET_SCHEMA_VERSION,
            });
        }
        let medium = Medium::from_str(&self.medium)
            .map_err(|_| IorFileError::UnknownMedium(path.into(), self.medium.clone()))?;
        let data = match self.data {
            IorDataDto::Tabulated(rows) => IorData::Tabulated(
                rows.into_iter()
                    .map(|[w, e, k]| IorRecord::new(nanometres!(w as f32), e as f32, k as f32))
                    .collect(),
            ),
            IorDataDto::Dispersion {
                formula,
                range_nm,
                k,
            } => IorData::Dispersion {
                formula,
                range: (
                    nanometres!(range_nm[0] as f32),
                    nanometres!(range_nm[1] as f32),
                ),
                k: k.map(|rows| {
                    rows.into_iter()
                        .map(|[w, kv]| (nanometres!(w as f32), kv as f32))
                        .collect::<Box<[_]>>()
                }),
            },
        };
        Ok(IorDataset {
            medium,
            name: self.name,
            reference: self.reference,
            comments: self.comments,
            data,
        })
    }

    /// Reads a legacy `*.csv` file (`"wavelength, µm", "n"[, "k"]`, optional UTF-8 BOM, µm
    /// wavelengths) into a `Tabulated` DTO. `medium` and `name` are supplide by the caller.
    pub fn from_legacy_csv(
        path: &Path,
        medium: Medium,
        name: &str,
    ) -> Result<IorDatasetDto, IorFileError> {
        let p = path.display().to_string();
        let raw = std::fs::read(path).map_err(|source| IorFileError::Io {
            path: p.clone(),
            source,
        })?;
        // Strip UTF-8 BOM if present.
        let bytes = raw.strip_prefix(&[0xEF, 0xBB, 0xBF][..]).unwrap_or(&raw);
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(bytes);
        let n_cols = rdr
            .headers()
            .map_err(|e| IorFileError::Csv {
                path: p.clone(),
                msg: e.to_string(),
            })?
            .len();
        let mut rows: Vec<[f64; 3]> = Vec::new();
        for rec in rdr.records() {
            let rec = rec.map_err(|e| IorFileError::Csv {
                path: p.clone(),
                msg: e.to_string(),
            })?;
            let parse = |i: usize| -> Result<f64, IorFileError> {
                rec.get(i)
                    .ok_or_else(|| IorFileError::Csv {
                        path: p.clone(),
                        msg: format!("missing column {i}"),
                    })
                    .and_then(|s| {
                        s.trim().parse::<f64>().map_err(|e| IorFileError::Csv {
                            path: p.clone(),
                            msg: e.to_string(),
                        })
                    })
            };
            let lambda_nm = parse(0)? * 1000.0; // µm -> nm
            let eta = parse(1)?;
            let k = if n_cols >= 3 { parse(2)? } else { 0.0 };
            rows.push([lambda_nm, eta, k]);
        }
        rows.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        Ok(IorDatasetDto {
            schema_version: IOR_DATASET_SCHEMA_VERSION,
            medium: medium.name().to_string(),
            name: name.to_string(),
            reference: String::new(),
            comments: String::new(),
            data: IorDataDto::Tabulated(rows),
        })
    }

    /// Reads a `*.ior.ron` file into a dto.
    pub fn read(path: &Path) -> Result<IorDatasetDto, IorFileError> {
        let p = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| IorFileError::Io {
            path: p.clone(),
            source,
        })?;
        Self::parse(&text, Some(p))
    }

    /// Serialises a dto dataset to a `*.ior.ron` file (pretty-printed).
    pub fn write(&self, path: &Path) -> Result<(), IorFileError> {
        let p = path.display().to_string();
        let pretty = ron::ser::PrettyConfig::new().struct_names(false);
        let text = ron::ser::to_string_pretty(&self, pretty).map_err(|e| IorFileError::Ron {
            path: p.clone(),
            source: e.into(),
        })?;
        std::fs::write(path, text).map_err(|source| IorFileError::Io { path: p, source })
    }

    /// Parse a DTO from text.
    ///
    /// The text is expected to be the content of a `*.ior.ron` file, but this method doesn't
    /// enforce that or use the `path` for anything other than error messages.
    ///
    /// `path` is used only for error messages; if `None`, errors will have a generic `<string>`
    /// path.
    pub fn parse(src: &str, path: Option<String>) -> Result<IorDatasetDto, IorFileError> {
        let p = path.unwrap_or_else(|| "<string>".into());

        ron::from_str(&src).map_err(|e| IorFileError::Ron {
            path: p,
            source: e.into(),
        })
    }
}

/// Reads a `*.ior.ron` file into a runtime [`IorDataset`].
///
/// Convenience wrapper over [`IorDatasetDto::read`] + [`IorDatasetDto::into_runtime`].
pub fn read_dataset_file(path: &Path) -> Result<IorDataset, IorFileError> {
    IorDatasetDto::read(path)?.into_runtime(&path.display().to_string())
}

/// Serialises a runtime [`IorDataset`] to a `*.ior.ron` file (pretty-printed).
///
/// Convenience wrapper over [`IorDatasetDto::from_runtime`] + [`IorDatasetDto::write`].
pub fn write_dataset_file(path: &Path, dataset: &IorDataset) -> Result<(), IorFileError> {
    IorDatasetDto::from_runtime(dataset).write(path)
}

/// The `datafiles/ior/sources.toml` manifest: pinned upstream + per-dataset records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManifestDto {
    /// Upstream repo URL (informational).
    #[serde(default)]
    pub upstream: Option<String>,
    /// Default git ref/commit to fetch from when `add`/`update` don't specify one.
    #[serde(default)]
    pub default_ref: Option<String>,
    /// One record per `.ior.ron` file in this directory.
    #[serde(default, rename = "dataset")]
    pub datasets: Vec<DatasetEntry>,
}

/// One `[[dataset]]` record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// `.ior.ron` file name (relative to the manifest's directory).
    pub file: String,
    /// Lowercase medium name; must equal the file body's `medium`.
    pub medium: String,
    /// Whether this is the chosen dataset for its medium (≤1 per medium). Absent => false.
    #[serde(default)]
    pub default: bool,
    /// Catalog-relative upstream path (`<shelf>/<book>/nk/<page>.yml`); absent => local/manual
    /// dataset.
    #[serde(default)]
    pub path: Option<String>,
    /// Upstream git ref/commit this file was generated from.
    #[serde(default, rename = "ref")]
    pub git_ref: Option<String>,
    /// SHA-256 of the upstream source `.yml`.
    #[serde(default)]
    pub sha256: Option<String>,
    /// `false` => the upstream page identification is a best guess. Absent => true.
    #[serde(default = "default_true")]
    pub verified: bool,
}

fn default_true() -> bool { true }

impl ManifestDto {
    /// Reads `sources.toml`; returns an empty manifest if it doesn't exist.
    pub fn read(path: &Path) -> Result<ManifestDto, IorFileError> {
        if !path.exists() {
            return Ok(ManifestDto::default());
        }
        let p = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| IorFileError::Io {
            path: p.clone(),
            source,
        })?;
        Self::parse(&text, Some(p))
    }

    /// Writes `sources.toml`.
    pub fn write(&self, path: &Path) -> Result<(), IorFileError> {
        let p = path.display().to_string();
        let text =
            toml::to_string_pretty(self).map_err(|e| IorFileError::Toml(format!("{p}: {e}")))?;
        std::fs::write(path, text).map_err(|source| IorFileError::Io { path: p, source })
    }

    /// Parses manifest text.
    ///
    /// `path` is used only for error messages; if `None`, errors will have a generic `<manifest>`
    /// path.
    pub fn parse(src: &str, path: Option<String>) -> Result<ManifestDto, IorFileError> {
        let p = path.unwrap_or_else(|| "<manifest>".into());
        toml::from_str(&src).map_err(|e| IorFileError::Toml(format!("{p}: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::nm;

    fn sample_tabulated() -> IorDataset {
        IorDataset {
            medium: Medium::Aluminium,
            name: "Demo2024".into(),
            reference: "Some Author. Title. Journal (2024)".into(),
            comments: String::new(),
            data: IorData::Tabulated(Box::from([
                IorRecord::new(nm!(400.0), 1.0, 5.0),
                IorRecord::new(nm!(500.0), 2.0, 7.0),
            ])),
        }
    }

    fn sample_dispersion() -> IorDataset {
        IorDataset {
            medium: Medium::Pvc,
            name: "Demo".into(),
            reference: String::new(),
            comments: String::new(),
            data: IorData::Dispersion {
                formula: DispersionFormula::Sellmeier {
                    c0: 0.0,
                    terms: vec![(1.0, 0.01)],
                },
                range: (nm!(300.0), nm!(2000.0)),
                k: Some(Box::from([(nm!(300.0), 0.1f32), (nm!(2000.0), 0.4f32)])),
            },
        }
    }

    #[test]
    fn round_trip_tabulated_and_dispersion() {
        for ds in [sample_tabulated(), sample_dispersion()] {
            let dto = IorDatasetDto::from_runtime(&ds);
            let text =
                ron::ser::to_string_pretty(&dto, ron::ser::PrettyConfig::new().struct_names(false))
                    .unwrap();
            let back: IorDatasetDto = ron::from_str(&text).unwrap();
            let runtime = back.into_runtime("<test>").unwrap();
            assert_eq!(runtime, ds);
        }
    }

    #[test]
    fn unknown_schema_version_is_rejected() {
        let mut dto = IorDatasetDto::from_runtime(&sample_tabulated());
        dto.schema_version = 999;
        let err = dto.into_runtime("foo.ior.ron").unwrap_err();
        assert!(matches!(
            err,
            IorFileError::SchemaVersion { found: 999, .. }
        ));
    }

    #[test]
    fn unknown_medium_is_rejected() {
        let mut dto = IorDatasetDto::from_runtime(&sample_tabulated());
        dto.medium = "unobtainium".into();
        assert!(matches!(
            dto.into_runtime("foo.ior.ron").unwrap_err(),
            IorFileError::UnknownMedium(..)
        ));
    }

    #[test]
    fn write_then_read_file_round_trips(/* uses a tempdir */) {
        let dir = std::env::temp_dir().join(format!("vgn-ior-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("al_Demo2024.ior.ron");
        let ds = sample_tabulated();
        ds.write(&path).unwrap();
        assert_eq!(IorDataset::read(&path).unwrap(), ds);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_round_trips(/* uses a tempdir */) {
        let dir = std::env::temp_dir().join(format!("vgn-ior-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sources.toml");
        let m = ManifestDto {
            upstream: Some("https://example/db".into()),
            default_ref: Some("abc123".into()),
            datasets: vec![
                DatasetEntry {
                    file: "al_McPeak2015.ior.ron".into(),
                    medium: "al".into(),
                    default: true,
                    path: Some("main/Al/nk/McPeak.yml".into()),
                    git_ref: Some("abc123".into()),
                    sha256: None,
                    verified: true,
                },
                DatasetEntry {
                    file: "cu_Demo.ior.ron".into(),
                    medium: "cu".into(),
                    default: false,
                    path: None,
                    git_ref: None,
                    sha256: None,
                    verified: false,
                },
            ],
        };
        m.write(&path).unwrap();
        let back = ManifestDto::read(&path).unwrap();
        assert_eq!(back.datasets.len(), 2);
        assert!(back.datasets[0].default && back.datasets[0].verified);
        assert!(!back.datasets[1].default && !back.datasets[1].verified);
        // `verified` defaults to true when absent:
        std::fs::write(&path, "[[dataset]]\nfile=\"x.ior.ron\"\nmedium=\"al\"\n").unwrap();
        assert!(ManifestDto::read(&path).unwrap().datasets[0].verified);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn legacy_csv_parses_all_four_repo_files() {
        // Run from the workspace root (cargo test sets CWD to the crate dir, so go up two levels).
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let cases: [(&str, Medium, usize); 4] = [
            ("datafiles/ior/air_iors_[0.23-1.69].csv", Medium::Air, 2), // no BOM, 2 cols
            (
                "datafiles/ior/al_iors_[0.15-1.7]_McPeak2015.csv",
                Medium::Aluminium,
                3,
            ), // no BOM, 3 cols
            (
                "datafiles/ior/al_iors_[0.225-1.0]_Cheng2016.csv",
                Medium::Aluminium,
                3,
            ), // BOM, 3 cols
            ("datafiles/ior/cu_iors_[0.3-1.7].csv", Medium::Copper, 3), // BOM, 3 cols
        ];
        for (rel, medium, _cols) in cases {
            let path = root.join(rel);
            let dto = IorDatasetDto::from_legacy_csv(&path, medium, "Legacy")
                .unwrap_or_else(|e| panic!("{rel}: {e}"));
            let runtime = dto.clone().into_runtime(rel).unwrap();
            match runtime.data {
                IorData::Tabulated(s) => {
                    assert!(!s.is_empty(), "{rel}: empty");
                    // ascending, nm units (first sample for these files is > 100 nm)
                    assert!(
                        s[0].wavelength.value() > 100.0,
                        "{rel}: looks like µm, not nm"
                    );
                    assert!(
                        s.windows(2).all(|w| w[0].wavelength <= w[1].wavelength),
                        "{rel}: not sorted"
                    );
                },
                _ => panic!("{rel}: expected Tabulated"),
            }
        }
    }
}

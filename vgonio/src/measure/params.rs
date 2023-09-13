//! Measurement parameters.
pub use crate::measure::{bsdf::params::*, microfacet::params::*};

use crate::error::RuntimeError;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    fs::File,
    hash::Hash,
    io::BufReader,
    path::{Path, PathBuf},
};
use vgcore::error::VgonioError;

/// Describes the different kind of measurements with parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementParams {
    /// Measure the BSDF of a micro-surface.
    Bsdf(BsdfMeasurementParams),
    /// Measure the micro-facet area distribution function of a micro-surface.
    #[serde(alias = "microfacet-area-distribution-function")]
    Adf(AdfMeasurementParams),
    /// Measure the micro-facet masking/shadowing function.
    #[serde(alias = "microfacet-masking-shadowing-function")]
    Msf(MsfMeasurementParams),
}

impl MeasurementParams {
    /// Whether the measurement parameters are valid.
    pub fn validate(self) -> Result<Self, VgonioError> {
        match self {
            Self::Bsdf(bsdf) => Ok(Self::Bsdf(bsdf.validate()?)),
            Self::Adf(mfd) => Ok(Self::Adf(mfd.validate()?)),
            Self::Msf(mfs) => Ok(Self::Msf(mfs.validate()?)),
        }
    }

    /// Whether the measurement is a BSDF measurement.
    pub fn is_bsdf(&self) -> bool { matches!(self, Self::Bsdf { .. }) }

    /// Whether the measurement is a micro-facet distribution measurement.
    pub fn is_microfacet_distribution(&self) -> bool { matches!(self, Self::Adf { .. }) }

    /// Whether the measurement is a micro-surface shadowing-masking function
    /// measurement.
    pub fn is_micro_surface_shadow_masking(&self) -> bool { matches!(self, Self::Msf { .. }) }

    /// Get the BSDF measurement parameters.
    pub fn bsdf(&self) -> Option<&BsdfMeasurementParams> {
        if let MeasurementParams::Bsdf(bsdf) = self {
            Some(bsdf)
        } else {
            None
        }
    }

    /// Get the micro-facet distribution measurement parameters.
    pub fn microfacet_distribution(&self) -> Option<&AdfMeasurementParams> {
        if let MeasurementParams::Adf(mfd) = self {
            Some(mfd)
        } else {
            None
        }
    }

    /// Get the micro-surface shadowing-masking function measurement parameters.
    pub fn micro_surface_shadow_masking(&self) -> Option<&MsfMeasurementParams> {
        if let MeasurementParams::Msf(mfd) = self {
            Some(mfd)
        } else {
            None
        }
    }
}

/// Description of BSDF measurement.
/// This is used to describe the parameters of different kinds of measurements.
/// The measurement description file uses the [YAML](https://yaml.org/) format.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub struct Measurement {
    /// Type of measurement.
    #[serde(rename = "type")]
    pub params: MeasurementParams,
    /// Surfaces to be measured. surface's path can be prefixed with either
    /// `usr://` or `sys://` to indicate the user-defined data file path
    /// or system-defined data file path.
    pub surfaces: Vec<PathBuf>, // TODO(yang): use Cow<'a, Vec<PathBuf>> to avoid copying the path
}

/// Kind of different measurements.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementKind {
    /// BSDF measurement.
    Bsdf = 0x00,
    /// Microfacet area distribution function measurement.
    Adf = 0x01,
    /// Microfacet Masking-shadowing function measurement.
    Msf = 0x02,
}

impl Display for MeasurementKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementKind::Bsdf => {
                write!(f, "BSDF")
            }
            MeasurementKind::Adf => {
                write!(f, "MDF")
            }
            MeasurementKind::Msf => {
                write!(f, "MSF")
            }
        }
    }
}

impl From<u8> for MeasurementKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Bsdf,
            0x01 => Self::Adf,
            0x02 => Self::Msf,
            _ => panic!("Invalid measurement kind! {}", value),
        }
    }
}

impl Measurement {
    /// Loads the measurement from a path. The path can be either a file path
    /// or a directory path. In the latter case, files with the extension
    /// `.yaml` or `.yml` are loaded.
    ///
    /// # Arguments
    /// * `path` - Path to the measurement file or directory, must be in
    ///   canonical form.
    pub fn load(path: &Path) -> Result<Vec<Measurement>, VgonioError> {
        if path.exists() {
            if path.is_dir() {
                Self::load_from_dir(path)
            } else {
                Self::load_from_file(path)
            }
        } else {
            Err(VgonioError::from_io_error(
                std::io::ErrorKind::NotFound.into(),
                format!("Path does not exist: {}", path.display()),
            ))
        }
    }

    /// Loads the measurement from a directory.
    /// # Arguments
    /// * `path` - Path to the measurement directory, must be in canonical form
    ///   and must exist.
    fn load_from_dir(dir: &Path) -> Result<Vec<Measurement>, VgonioError> {
        let mut measurements = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(|err| {
            VgonioError::from_io_error(err, format!("Failed to read directory: {}", dir.display()))
        })? {
            let entry = entry.map_err(|err| {
                VgonioError::from_io_error(
                    err,
                    format!("Failed to read directory: {}", dir.display()),
                )
            })?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "yaml" || ext == "yml" {
                        measurements.append(&mut Self::load_from_file(&path)?);
                    }
                }
            }
        }
        Ok(measurements)
    }

    /// Loads measurement descriptions from a file. A file may contain multiple
    /// descriptions, separated by `---` followed by a newline.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the file containing the measurement descriptions,
    ///   must be in canonical form and must exist.
    fn load_from_file(filepath: &Path) -> Result<Vec<Measurement>, VgonioError> {
        let mut file = File::open(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!(
                    "Failed to open measurement description file: {}",
                    filepath.display()
                ),
            )
        })?;
        let reader = BufReader::new(&mut file);
        let measurements = serde_yaml::Deserializer::from_reader(reader)
            .map(|doc| {
                Measurement::deserialize(doc)
                    .map_err(|err| {
                        VgonioError::new(
                            "Failed to deserialize measurement description",
                            Some(Box::new(RuntimeError::from(err))),
                        )
                    })
                    .and_then(|measurement| measurement.validate())
            })
            .collect::<Result<Vec<_>, VgonioError>>()?;

        Ok(measurements)
    }

    /// Validate the measurement description.
    pub fn validate(self) -> Result<Self, VgonioError> {
        log::info!("Validating measurement description...");
        let details = self.params.validate()?;
        Ok(Self {
            params: details,
            ..self
        })
    }

    /// Measurement kind in the form of a string.
    pub fn name(&self) -> &'static str {
        match self.params {
            MeasurementParams::Bsdf { .. } => "BSDF measurement",
            MeasurementParams::Adf { .. } => "microfacet-distribution measurement",
            MeasurementParams::Msf { .. } => "micro-surface-shadow-masking measurement",
        }
    }
}

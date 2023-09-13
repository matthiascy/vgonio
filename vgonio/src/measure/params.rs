//! Measurement parameters.
pub use crate::measure::{bsdf::params::*, microfacet::params::*};

use crate::{
    app::cache::{Asset, Handle},
    error::RuntimeError,
    fitting::MeasuredMdfData,
    measure::{
        bsdf::{
            detector::{Detector, DetectorScheme},
            emitter::{Emitter, RegionShape},
            rtc::RtcMethod,
            BsdfKind, MeasuredBsdfData,
        },
        microfacet::{MeasuredAdfData, MeasuredMsfData},
    },
    Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalPartition,
};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt::{Display, Formatter},
    fs::File,
    hash::Hash,
    io::BufReader,
    path::{Path, PathBuf},
};
use vgcore::{
    error::VgonioError,
    math::Aabb,
    units::{mm, nanometres, rad, LengthUnit, Millimetres, Radians, SolidAngle, UMillimetre},
};
use vgsurf::{MicroSurface, MicroSurfaceMesh};

/// Describes the radius of measurement.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Radius {
    /// Radius is dynamically deduced from the dimension of the surface.
    /// Note: This value will be updated before each measurement.
    Auto(#[serde(skip)] Millimetres),

    /// Radius is given explicitly.
    Fixed(Millimetres),
}

impl Default for Radius {
    fn default() -> Self { Self::Auto(mm!(1.0)) }
}

impl Display for Radius {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto(_) => write!(f, "auto"),
            Self::Fixed(v) => write!(f, "fixed: {v}"),
        }
    }
}

impl Radius {
    /// Whether the radius is dynamically deduced from the dimension of the
    /// surface.
    pub fn is_auto(&self) -> bool {
        match self {
            Radius::Auto(_) => true,
            Radius::Fixed(_) => false,
        }
    }

    /// Whether the radius is valid for the given surface.
    pub fn is_valid(&self) -> bool {
        match self {
            Radius::Auto(_) => true,
            Radius::Fixed(m) => m.value() > 0.0,
        }
    }

    /// Get the radius value.
    pub fn value(&self) -> Millimetres {
        match self {
            Radius::Auto(value) => *value,
            Radius::Fixed(value) => *value,
        }
    }

    /// Get a mutable reference to the radius value.
    pub fn value_mut(&mut self) -> &mut Millimetres {
        match self {
            Radius::Auto(value) => value,
            Radius::Fixed(value) => value,
        }
    }

    /// Evaluates the radius of the sphere/hemisphere enclosing the surface
    /// mesh.
    ///
    /// The returned value of the radius in the same unit as the
    /// `MicroSurfaceMesh`.
    pub fn estimate(&self, mesh: &MicroSurfaceMesh) -> f32 {
        match self {
            Radius::Auto(_) => mesh.bounds.max_extent() * std::f32::consts::SQRT_2,
            Radius::Fixed(r) => mesh.unit.factor_convert_from::<UMillimetre>() * r.value(),
        }
    }

    /// Evaluates the radius of the sphere/hemisphere enclosing the surface mesh
    /// with the bounds and unit of the given `MicroSurface`.
    pub fn estimate_with_bounds(&self, bounds: Aabb, unit: LengthUnit) -> f32 {
        match self {
            Radius::Auto(_) => bounds.max_extent() * std::f32::consts::SQRT_2,
            Radius::Fixed(r) => unit.factor_convert_from::<UMillimetre>() * r.value(),
        }
    }

    /// Evaluates the radius for the disk covering the surface mesh.
    pub fn estimate_disk_radius(&self, mesh: &MicroSurfaceMesh) -> f32 {
        match self {
            Radius::Auto(_) => mesh.bounds.max_extent() * 0.7,
            Radius::Fixed(_) => panic!("Disk radius is not supported for fixed radius"),
        }
    }

    /// Evaluates the radius for the disk covering the surface mesh with the
    /// bounds of the given `MicroSurface`.
    pub fn estimate_disk_radius_with_bounds(&self, bounds: Aabb) -> f32 {
        match self {
            Radius::Auto(_) => bounds.max_extent() * 0.7,
            Radius::Fixed(_) => panic!("Disk radius is not supported for fixed radius"),
        }
    }
}

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

/// Measurement data source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementDataSource {
    /// Measurement data is loaded from a file.
    Loaded(PathBuf),
    /// Measurement data is generated from a micro-surface.
    Measured(Handle<MicroSurface>),
}

impl MeasurementDataSource {
    /// Returns the path to the measurement data if it is loaded from a file.
    pub fn path(&self) -> Option<&Path> {
        match self {
            MeasurementDataSource::Loaded(p) => Some(p.as_path()),
            MeasurementDataSource::Measured(_) => None,
        }
    }
}

/// Different kinds of measurement data.
#[derive(Debug, Clone)]
pub enum MeasuredData {
    /// Bidirectional scattering distribution function.
    Bsdf(MeasuredBsdfData),
    /// Microfacet distribution function.
    Adf(MeasuredAdfData),
    /// Shadowing-masking function.
    Msf(MeasuredMsfData),
}

impl MeasuredData {
    /// Returns the measurement kind.
    pub fn kind(&self) -> MeasurementKind {
        match self {
            MeasuredData::Adf(_) => MeasurementKind::Adf,
            MeasuredData::Msf(_) => MeasurementKind::Msf,
            MeasuredData::Bsdf(_) => MeasurementKind::Bsdf,
        }
    }

    /// Returns the BSDF data if it is a BSDF.
    pub fn bsdf_data(&self) -> Option<&MeasuredBsdfData> {
        match self {
            MeasuredData::Bsdf(bsdf) => Some(bsdf),
            _ => None,
        }
    }

    /// Returns the MADF data if it is a MADF.
    pub fn adf_data(&self) -> Option<&MeasuredAdfData> {
        match self {
            MeasuredData::Adf(madf) => Some(madf),
            _ => None,
        }
    }

    /// Returns the MMSF data if it is a MMSF.
    pub fn msf_data(&self) -> Option<&MeasuredMsfData> {
        match self {
            MeasuredData::Msf(mmsf) => Some(mmsf),
            _ => None,
        }
    }

    pub fn mdf_data(&self) -> Option<MeasuredMdfData> {
        match self {
            MeasuredData::Bsdf(_) => None,
            MeasuredData::Adf(adf) => Some(MeasuredMdfData::Adf(Cow::Borrowed(adf))),
            MeasuredData::Msf(msf) => Some(MeasuredMdfData::Msf(Cow::Borrowed(msf))),
        }
    }

    /// Returns the zenith range of the measurement data if it is a MADF or
    /// MMSF.
    pub fn adf_or_msf_zenith(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(madf) => Some(madf.params.zenith),
            MeasuredData::Msf(mmsf) => Some(mmsf.params.zenith),
            MeasuredData::Bsdf(_) => None,
        }
    }

    /// Returns the azimuth range of the measurement data if it is a MADF or
    /// MMSF.
    pub fn madf_or_mmsf_azimuth(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(madf) => Some(madf.params.azimuth),
            MeasuredData::Msf(mmsf) => Some(mmsf.params.azimuth),
            MeasuredData::Bsdf(_) => None,
        }
    }

    // TODO: to be removed
    /// Returns the samples of the measurement data.
    pub fn samples(&self) -> &[f32] {
        match self {
            MeasuredData::Adf(madf) => &madf.samples,
            MeasuredData::Msf(mmsf) => &mmsf.samples,
            MeasuredData::Bsdf(_bsdf) => todo!("implement this"),
        }
    }
}

// TODO: add support for storing data in the memory in a compressed
//       format(maybe LZ4).
/// Structure for storing measurement data in the memory especially
/// when loading from a file.
#[derive(Debug, Clone)]
pub struct MeasurementData {
    /// Internal tag for displaying the measurement data in the GUI.
    pub name: String,
    /// Origin of the measurement data.
    pub source: MeasurementDataSource,
    /// Measurement data.
    pub measured: MeasuredData,
}

impl Asset for MeasurementData {}

impl PartialEq for MeasurementData {
    fn eq(&self, other: &Self) -> bool { self.source == other.source }
}

impl MeasurementData {
    /// Returns the kind of the measurement data.
    pub fn kind(&self) -> MeasurementKind { self.measured.kind() }

    /// Returns the Area Distribution Function data slice for the given
    /// azimuthal angle in radians.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    ///
    /// Azimuthal angle will be wrapped around to the range [0, 2π).
    ///
    /// 2π will be mapped to 0.
    ///
    /// # Arguments
    ///
    /// * `azimuth_m` - Azimuthal angle of the microfacet normal in radians.
    pub fn ndf_data_slice(&self, azimuth_m: Radians) -> (&[f32], Option<&[f32]>) {
        debug_assert!(self.kind() == MeasurementKind::Adf);
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_m_idx = self_azimuth.index_of(azimuth_m);
        let opposite_azimuth_m = azimuth_m.opposite();
        let opposite_index = if self_azimuth.start <= opposite_azimuth_m
            && opposite_azimuth_m <= self_azimuth.stop
        {
            Some(self_azimuth.index_of(opposite_azimuth_m))
        } else {
            None
        };
        (
            self.ndf_data_slice_inner(azimuth_m_idx),
            opposite_index.map(|index| self.ndf_data_slice_inner(index)),
        )
    }

    /// Returns a data slice of the Area Distribution Function for the given
    /// azimuthal angle index.
    fn ndf_data_slice_inner(&self, azimuth_idx: usize) -> &[f32] {
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        debug_assert!(self.kind() == MeasurementKind::Adf);
        debug_assert!(
            azimuth_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        let self_zenith = self.measured.adf_or_msf_zenith().unwrap();
        &self.measured.samples()[azimuth_idx * self_zenith.step_count_wrapped()
            ..(azimuth_idx + 1) * self_zenith.step_count_wrapped()]
    }

    /// Returns the Masking Shadowing Function data slice for the given
    /// microfacet normal and azimuthal angle of the incident direction.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    pub fn msf_data_slice(
        &self,
        azimuth_m: Radians,
        zenith_m: Radians,
        azimuth_i: Radians,
    ) -> (&[f32], Option<&[f32]>) {
        debug_assert!(
            self.kind() == MeasurementKind::Msf,
            "measurement data kind should be MicrofacetMaskingShadowing"
        );
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let self_zenith = self.measured.adf_or_msf_zenith().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_i = azimuth_i.wrap_to_tau();
        let zenith_m = zenith_m.clamp(self_zenith.start, self_zenith.stop);
        let azimuth_m_idx = self_azimuth.index_of(azimuth_m);
        let zenith_m_idx = self_zenith.index_of(zenith_m);
        let azimuth_i_idx = self_azimuth.index_of(azimuth_i);
        let opposite_azimuth_i = azimuth_i.opposite();
        let opposite_azimuth_i_idx = if self_azimuth.start <= opposite_azimuth_i
            && opposite_azimuth_i <= self_azimuth.stop
        {
            Some(self_azimuth.index_of(opposite_azimuth_i))
        } else {
            None
        };
        (
            self.msf_data_slice_inner(azimuth_m_idx, zenith_m_idx, azimuth_i_idx),
            opposite_azimuth_i_idx
                .map(|index| self.msf_data_slice_inner(azimuth_m_idx, zenith_m_idx, index)),
        )
    }

    /// Returns a data slice of the Masking Shadowing Function for the given
    /// indices.
    fn msf_data_slice_inner(
        &self,
        azimuth_m_idx: usize,
        zenith_m_idx: usize,
        azimuth_i_idx: usize,
    ) -> &[f32] {
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let self_zenith = self.measured.adf_or_msf_zenith().unwrap();
        debug_assert!(self.kind() == MeasurementKind::Msf);
        debug_assert!(
            azimuth_m_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        debug_assert!(
            azimuth_i_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        debug_assert!(
            zenith_m_idx < self_zenith.step_count_wrapped(),
            "index out of range"
        );
        let zenith_bin_count = self_zenith.step_count_wrapped();
        let azimuth_bin_count = self_azimuth.step_count_wrapped();
        let offset = azimuth_m_idx * zenith_bin_count * azimuth_bin_count * zenith_bin_count
            + zenith_m_idx * azimuth_bin_count * zenith_bin_count
            + azimuth_i_idx * zenith_bin_count;
        &self.measured.samples()[offset..offset + zenith_bin_count]
    }
}

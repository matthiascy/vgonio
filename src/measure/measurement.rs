//! Measurement parameters.
use crate::{
    app::cache::{Asset, Handle},
    measure::{
        bsdf::{BsdfKind, MeasuredBsdfData},
        collector::CollectorScheme,
        emitter::RegionShape,
        microfacet::{MeasuredMadfData, MeasuredMmsfData},
        Collector, Emitter, RtcMethod,
    },
    msurf::{MicroSurface, MicroSurfaceMesh},
    units::{deg, mm, nanometres, rad, Millimetres, Radians, SolidAngle, UMillimetre},
    Error, Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalDomain,
    SphericalPartition,
};
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    fmt::{Display, Formatter},
    fs::File,
    hash::Hash,
    io::BufReader,
    path::{Path, PathBuf},
};

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

    /// Evaluate the radius of the sphere/hemisphere enclosing the surface mesh.
    ///
    /// The returned value of the radius in the same unit as the
    /// `MicroSurfaceMesh`.
    pub fn estimate(&self, mesh: &MicroSurfaceMesh) -> f32 {
        match self {
            Radius::Auto(_) => mesh.bounds.max_extent() * std::f32::consts::SQRT_2,
            Radius::Fixed(r) => mesh.unit.factor_convert_from::<UMillimetre>() * r.value,
        }
    }

    /// Evaluate the radius for the disk covering the surface mesh.
    pub fn estimate_disk_radius(&self, mesh: &MicroSurfaceMesh) -> f32 {
        match self {
            Radius::Auto(_) => mesh.bounds.max_extent() * 0.55,
            Radius::Fixed(_) => panic!("Disk radius is not supported for fixed radius"),
        }
    }
}

/// Possible ways to conduct a simulation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SimulationKind {
    /// Ray optics is used during the simulation.
    GeomOptics(RtcMethod),

    /// Wave optics is used during the simulation.
    WaveOptics,
}

impl TryFrom<u8> for SimulationKind {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::GeomOptics(RtcMethod::Grid)),
            #[cfg(feature = "embree")]
            0x01 => Ok(Self::GeomOptics(RtcMethod::Embree)),
            #[cfg(feature = "optix")]
            0x02 => Ok(Self::GeomOptics(RtcMethod::Optix)),
            0x03 => Ok(Self::WaveOptics),
            _ => Err(format!("Invalid simulation kind {}", value)),
        }
    }
}

/// Parameters for BSDF measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BsdfMeasurementParams {
    /// The measurement kind.
    pub kind: BsdfKind,

    /// The simulation kind.
    pub sim_kind: SimulationKind,

    /// Incident medium of the measurement.
    pub incident_medium: Medium,

    /// Transmitted medium of the measurement (medium of the surface).
    pub transmitted_medium: Medium,

    /// Description of the emitter.
    pub emitter: Emitter,

    /// Description of the collector.
    pub collector: Collector,
}

impl Default for BsdfMeasurementParams {
    fn default() -> Self {
        Self {
            kind: BsdfKind::Brdf,
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Grid),
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Air,
            emitter: Emitter {
                num_rays: 1000,
                max_bounces: 10,
                radius: Radius::Auto(mm!(0.0)),
                zenith: RangeByStepSizeInclusive::new(
                    deg!(0.0).in_radians(),
                    deg!(90.0).in_radians(),
                    deg!(5.0).in_radians(),
                ),
                azimuth: RangeByStepSizeInclusive::new(
                    deg!(0.0).in_radians(),
                    deg!(360.0).in_radians(),
                    deg!(60.0).in_radians(),
                ),
                shape: RegionShape::SphericalCap {
                    zenith: deg!(5.0).in_radians(),
                },
                solid_angle: SolidAngle::from_angle_ranges(
                    (rad!(0.0), deg!(5.0).in_radians()),
                    (rad!(0.0), rad!(2.0 * std::f32::consts::PI)),
                ),
                spectrum: RangeByStepSizeInclusive::new(
                    nanometres!(400.0),
                    nanometres!(700.0),
                    nanometres!(1.0),
                ),
            },
            collector: Collector {
                radius: Radius::Auto(mm!(0.0)),
                scheme: CollectorScheme::Partitioned {
                    domain: SphericalDomain::Upper,
                    partition: SphericalPartition::EqualArea {
                        zenith: RangeByStepCountInclusive::new(
                            deg!(0.0).in_radians(),
                            deg!(90.0).in_radians(),
                            1,
                        ),
                        azimuth: RangeByStepSizeInclusive::new(
                            deg!(0.0).in_radians(),
                            deg!(360.0).in_radians(),
                            deg!(10.0).in_radians(),
                        ),
                    },
                },
            },
        }
    }
}

impl BsdfMeasurementParams {
    /// Returns the number of samples for the emitter.
    pub fn bsdf_data_samples_count(&self) -> usize { self.emitter.bsdf_data_samples_count() }

    /// Whether the measurement parameters are valid.
    pub fn validate(self) -> Result<Self, Error> {
        log::info!("Validating measurement description...");
        let emitter = &self.emitter;
        if emitter.num_rays < 1 {
            return Err(Error::InvalidEmitter("number of rays must be at least 1"));
        }

        if emitter.max_bounces < 1 {
            return Err(Error::InvalidEmitter(
                "maximum number of bounces must be at least 1",
            ));
        }

        if !(emitter.radius().is_valid()
            && emitter.zenith.step_size > rad!(0.0)
            && emitter.azimuth.step_size > rad!(0.0))
        {
            return Err(Error::InvalidEmitter(
                "emitter's radius, zenith step size and azimuth step size must be positive",
            ));
        }

        match emitter.shape {
            RegionShape::SphericalCap { zenith } => {
                if zenith <= rad!(0.0) || zenith > deg!(360.0) {
                    return Err(Error::InvalidEmitter(
                        "emitter's zenith angle must be in the range [0°, 360°]",
                    ));
                }
            }
            RegionShape::SphericalRect { zenith, azimuth } => {
                if zenith.0 <= rad!(0.0) || zenith.1 > deg!(360.0) {
                    return Err(Error::InvalidEmitter(
                        "emitter's zenith angle must be in the range [0°, 360°]",
                    ));
                }

                if azimuth.0 <= rad!(0.0) || azimuth.1 > deg!(360.0) {
                    return Err(Error::InvalidEmitter(
                        "emitter's azimuth angle must be in the range [0°, 360°]",
                    ));
                }
            }
            _ => {}
        }

        if emitter.spectrum.start < nanometres!(0.0) || emitter.spectrum.stop < nanometres!(0.0) {
            return Err(Error::InvalidEmitter(
                "emitter's spectrum must be in the range [0nm, ∞)",
            ));
        }

        if emitter.spectrum.start > emitter.spectrum.stop {
            return Err(Error::InvalidEmitter(
                "emitter's spectrum start must be less than or equal to its end",
            ));
        }

        let collector = &self.collector;
        if !collector.radius.is_valid() {
            return Err(Error::InvalidCollector(
                "collector's radius must be positive",
            ));
        }

        let is_valid_scheme = match &collector.scheme {
            CollectorScheme::Partitioned { partition, .. } => match partition {
                SphericalPartition::EqualAngle { zenith, azimuth } => (
                    zenith.start != zenith.stop && azimuth.start != azimuth.stop,
                    "zenith and azimuth's start and stop must not be equal",
                ),
                SphericalPartition::EqualArea { zenith, azimuth } => (
                    zenith.start != zenith.stop && azimuth.start != azimuth.stop,
                    "zenith and azimuth's start and stop must not be equal",
                ),
                SphericalPartition::EqualProjectedArea { zenith, azimuth } => (
                    zenith.start != zenith.stop && azimuth.start != azimuth.stop,
                    "zenith and azimuth's start and stop must not be equal",
                ),
            },
            CollectorScheme::SingleRegion { shape, .. } => match shape {
                RegionShape::SphericalCap { zenith } => {
                    if !zenith.is_positive() {
                        (false, "collector's zenith angle must be positive")
                    } else {
                        (true, "")
                    }
                }
                RegionShape::SphericalRect { .. } => (true, ""),
                _ => (true, ""),
            },
        };

        if !is_valid_scheme.0 {
            return Err(Error::InvalidCollector(is_valid_scheme.1));
        }

        Ok(self)
    }
}

const DEFAULT_AZIMUTH_RANGE: RangeByStepSizeInclusive<Radians> =
    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TWO_PI, deg!(5.0).in_radians());

const DEFAULT_ZENITH_RANGE: RangeByStepSizeInclusive<Radians> =
    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::HALF_PI, deg!(2.0).in_radians());

/// Parameters for microfacet area distribution measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MadfMeasurementParams {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSizeInclusive<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSizeInclusive<Radians>,
}

impl Default for MadfMeasurementParams {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }
}

impl MadfMeasurementParams {
    /// Returns the number of samples with the current parameters.
    pub fn samples_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// Validate the parameters.
    pub fn validate(self) -> Result<Self, Error> {
        if !(self.azimuth.start >= Radians::ZERO
            && self.azimuth.stop
                <= (Radians::TWO_PI + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
        {
            return Err(Error::InvalidParameter(
                "Microfacet distribution measurement: azimuth angle must be in the range [0°, \
                 360°]",
            ));
        }
        if !(self.zenith.start >= Radians::ZERO && self.zenith.start <= Radians::HALF_PI) {
            return Err(Error::InvalidParameter(
                "Microfacet distribution measurement: zenith angle must be in the range [0°, 90°]",
            ));
        }
        if !(self.azimuth.step_size > rad!(0.0) && self.zenith.step_size > rad!(0.0)) {
            return Err(Error::InvalidParameter(
                "Microfacet distribution measurement: azimuth and zenith step sizes must be \
                 positive!",
            ));
        }

        Ok(self)
    }
}

/// Parameters for microfacet masking and shadowing function measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MmsfMeasurementParams {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSizeInclusive<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSizeInclusive<Radians>,
    /// Discritization resolution during the measurement (area estimation).
    pub resolution: u32,
}

impl Default for MmsfMeasurementParams {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
            resolution: 512,
        }
    }
}

impl MmsfMeasurementParams {
    pub fn samples_count(&self) -> usize {
        (self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()).pow(2)
    }

    /// Counts the number of samples (on hemisphere) that will be taken during
    /// the measurement.
    pub fn measurement_location_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// Validate the parameters when reading from a file.
    pub fn validate(self) -> Result<Self, Error> {
        if !(self.azimuth.start >= Radians::ZERO
            && self.azimuth.stop
                <= (Radians::TWO_PI + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
        {
            return Err(Error::InvalidParameter(
                "Microfacet shadowing-masking measurement: azimuth angle must be in the range \
                 [0°, 360°]",
            ));
        }
        if !(self.zenith.start >= Radians::ZERO
            && self.zenith.start
                <= (Radians::HALF_PI + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
        {
            return Err(Error::InvalidParameter(
                "Microfacet shadowing-masking measurement: zenith angle must be in the range [0°, \
                 90°]",
            ));
        }
        if !(self.azimuth.step_size > rad!(0.0) && self.zenith.step_size > rad!(0.0)) {
            return Err(Error::InvalidParameter(
                "Microfacet shadowing-masking measurement: azimuth and zenith step sizes must be \
                 positive!",
            ));
        }

        Ok(self)
    }
}

/// Describes the different kind of measurements with parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementKindDescription {
    /// Measure the BSDF of a micro-surface.
    Bsdf(BsdfMeasurementParams),
    /// Measure the micro-facet area distribution function of a micro-surface.
    #[serde(alias = "microfacet-area-distribution-function")]
    Madf(MadfMeasurementParams),
    /// Measure the micro-facet masking/shadowing function.
    #[serde(alias = "microfacet-masking-shadowing-function")]
    Mmsf(MmsfMeasurementParams),
}

impl MeasurementKindDescription {
    /// Whether the measurement parameters are valid.
    pub fn validate(self) -> Result<Self, Error> {
        match self {
            Self::Bsdf(bsdf) => Ok(Self::Bsdf(bsdf.validate()?)),
            Self::Madf(mfd) => Ok(Self::Madf(mfd.validate()?)),
            Self::Mmsf(mfs) => Ok(Self::Mmsf(mfs.validate()?)),
        }
    }

    /// Whether the measurement is a BSDF measurement.
    pub fn is_bsdf(&self) -> bool { matches!(self, Self::Bsdf { .. }) }

    /// Whether the measurement is a micro-facet distribution measurement.
    pub fn is_microfacet_distribution(&self) -> bool { matches!(self, Self::Madf { .. }) }

    /// Whether the measurement is a micro-surface shadowing-masking function
    /// measurement.
    pub fn is_micro_surface_shadow_masking(&self) -> bool { matches!(self, Self::Mmsf { .. }) }

    /// Get the BSDF measurement parameters.
    pub fn bsdf(&self) -> Option<&BsdfMeasurementParams> {
        if let MeasurementKindDescription::Bsdf(bsdf) = self {
            Some(bsdf)
        } else {
            None
        }
    }

    /// Get the micro-facet distribution measurement parameters.
    pub fn microfacet_distribution(&self) -> Option<&MadfMeasurementParams> {
        if let MeasurementKindDescription::Madf(mfd) = self {
            Some(mfd)
        } else {
            None
        }
    }

    /// Get the micro-surface shadowing-masking function measurement parameters.
    pub fn micro_surface_shadow_masking(&self) -> Option<&MmsfMeasurementParams> {
        if let MeasurementKindDescription::Mmsf(mfd) = self {
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
    pub desc: MeasurementKindDescription,
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
    /// Micro-facet area distribution measurement.
    MicrofacetAreaDistribution = 0x01,
    /// Micro-surface shadowing-masking function measurement.
    MicrofacetMaskingShadowing = 0x02,
}

impl Display for MeasurementKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementKind::Bsdf => {
                write!(f, "BSDF")
            }
            MeasurementKind::MicrofacetAreaDistribution => {
                write!(f, "NDF")
            }
            MeasurementKind::MicrofacetMaskingShadowing => {
                write!(f, "Masking/Shadowing")
            }
        }
    }
}

impl From<u8> for MeasurementKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Bsdf,
            0x01 => Self::MicrofacetAreaDistribution,
            0x02 => Self::MicrofacetMaskingShadowing,
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
    pub fn load(path: &Path) -> Result<Vec<Measurement>, Error> {
        if path.exists() {
            if path.is_dir() {
                Self::load_from_dir(path)
            } else {
                Self::load_from_file(path)
            }
        } else {
            Err(Error::DirectoryOrFileNotFound(path.to_path_buf()))
        }
    }

    /// Loads the measurement from a directory.
    /// # Arguments
    /// * `path` - Path to the measurement directory, must be in canonical form
    ///   and must exist.
    fn load_from_dir(dir: &Path) -> Result<Vec<Measurement>, Error> {
        let mut measurements = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
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
    fn load_from_file(filepath: &Path) -> Result<Vec<Measurement>, Error> {
        let mut file = File::open(filepath)?;
        let reader = BufReader::new(&mut file);
        // TODO: invoke generation of patches and samples
        let measurements = serde_yaml::Deserializer::from_reader(reader)
            .map(|doc| {
                Measurement::deserialize(doc)
                    .map_err(Error::from)
                    .and_then(|measurement| measurement.validate())
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(measurements)
    }

    /// Validate the measurement description.
    pub fn validate(self) -> Result<Self, Error> {
        log::info!("Validating measurement description...");
        let details = self.desc.validate()?;
        Ok(Self {
            desc: details,
            ..self
        })
    }

    /// Measurement kind in the form of a string.
    pub fn name(&self) -> &'static str {
        match self.desc {
            MeasurementKindDescription::Bsdf { .. } => "BSDF measurement",
            MeasurementKindDescription::Madf { .. } => "microfacet-distribution measurement",
            MeasurementKindDescription::Mmsf { .. } => "micro-surface-shadow-masking measurement",
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
    /// Micro-facet area distribution.
    Madf(MeasuredMadfData),
    /// Micro-facet shadowing-masking function.
    Mmsf(MeasuredMmsfData),
    /// Bidirectional scattering distribution function.
    Bsdf(MeasuredBsdfData),
}

impl MeasuredData {
    /// Returns the measurement kind.
    pub fn kind(&self) -> MeasurementKind {
        match self {
            MeasuredData::Madf(_) => MeasurementKind::MicrofacetAreaDistribution,
            MeasuredData::Mmsf(_) => MeasurementKind::MicrofacetMaskingShadowing,
            MeasuredData::Bsdf(_) => MeasurementKind::Bsdf,
        }
    }

    pub fn bsdf_data(&self) -> Option<&MeasuredBsdfData> {
        match self {
            MeasuredData::Bsdf(bsdf) => Some(bsdf),
            _ => None,
        }
    }

    pub fn madf_data(&self) -> Option<&MeasuredMadfData> {
        match self {
            MeasuredData::Madf(madf) => Some(madf),
            _ => None,
        }
    }

    pub fn mmsf_data(&self) -> Option<&MeasuredMmsfData> {
        match self {
            MeasuredData::Mmsf(mmsf) => Some(mmsf),
            _ => None,
        }
    }

    /// Returns the zenith range of the measurement data if it is a MADF or
    /// MMSF.
    pub fn madf_or_mmsf_zenith(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Madf(madf) => Some(madf.params.zenith),
            MeasuredData::Mmsf(mmsf) => Some(mmsf.params.zenith),
            MeasuredData::Bsdf(_) => None,
        }
    }

    /// Returns the azimuth range of the measurement data if it is a MADF or
    /// MMSF.
    pub fn madf_or_mmsf_azimuth(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Madf(madf) => Some(madf.params.azimuth),
            MeasuredData::Mmsf(mmsf) => Some(mmsf.params.azimuth),
            MeasuredData::Bsdf(_) => None,
        }
    }

    // TODO: to be removed
    /// Returns the samples of the measurement data.
    pub fn samples(&self) -> &[f32] {
        match self {
            MeasuredData::Madf(madf) => &madf.samples,
            MeasuredData::Mmsf(mmsf) => &mmsf.samples,
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
    pub fn adf_data_slice(&self, azimuth_m: Radians) -> (&[f32], Option<&[f32]>) {
        debug_assert!(self.kind() == MeasurementKind::MicrofacetAreaDistribution);
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_m_idx = self_azimuth.index_of(azimuth_m.into());
        let opposite_azimuth_m = azimuth_m.opposite();
        let opposite_index = if self_azimuth.start <= opposite_azimuth_m
            && opposite_azimuth_m <= self_azimuth.stop
        {
            Some(self_azimuth.index_of(opposite_azimuth_m))
        } else {
            None
        };
        (
            self.adf_data_slice_inner(azimuth_m_idx),
            opposite_index.map(|index| self.adf_data_slice_inner(index)),
        )
    }

    /// Returns a data slice of the Area Distribution Function for the given
    /// azimuthal angle index.
    pub fn adf_data_slice_inner(&self, azimuth_idx: usize) -> &[f32] {
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        debug_assert!(self.kind() == MeasurementKind::MicrofacetAreaDistribution);
        debug_assert!(
            azimuth_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        let self_zenith = self.measured.madf_or_mmsf_zenith().unwrap();
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
            self.kind() == MeasurementKind::MicrofacetMaskingShadowing,
            "measurement data kind should be MicrofacetMaskingShadowing"
        );
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let self_zenith = self.measured.madf_or_mmsf_zenith().unwrap();
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
        let self_zenith = self.measured.madf_or_mmsf_zenith().unwrap();
        debug_assert!(self.kind() == MeasurementKind::MicrofacetMaskingShadowing);
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

//! Measurement parameters.
use crate::{
    app::{
        cache::{Asset, Handle},
        gui::{Plottable, PlottingMode},
    },
    io,
    io::vgmo::AngleRange,
    measure::{
        bsdf::BsdfKind, collector::CollectorScheme, emitter::RegionShape, Collector, Emitter,
        RtcMethod,
    },
    msurf::MicroSurface,
    units::{deg, mm, nanometres, rad, Millimetres, Radians, SolidAngle},
    Error, Medium, RangeByStepCount, RangeByStepSize, SphericalDomain, SphericalPartition,
};
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    fmt::{Display, Formatter},
    fs::File,
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

    /// Get the radius value without the unit.
    pub fn value(&self) -> Millimetres {
        match self {
            Radius::Auto(value) => *value,
            Radius::Fixed(value) => *value,
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

/// Parameters for BSDF measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BsdfMeasurement {
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

impl Default for BsdfMeasurement {
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
                zenith: RangeByStepSize::<Radians> {
                    start: deg!(0.0).in_radians(),
                    stop: deg!(90.0).in_radians(),
                    step_size: deg!(5.0).in_radians(),
                },
                azimuth: RangeByStepSize::<Radians> {
                    start: deg!(0.0).in_radians(),
                    stop: deg!(360.0).in_radians(),
                    step_size: deg!(120.0).in_radians(),
                },
                shape: RegionShape::SphericalCap {
                    zenith: deg!(5.0).in_radians(),
                },
                solid_angle: SolidAngle::from_angle_ranges(
                    (rad!(0.0), deg!(5.0).in_radians()),
                    (rad!(0.0), rad!(2.0 * std::f32::consts::PI)),
                ),
                spectrum: RangeByStepSize {
                    start: nanometres!(400.0),
                    stop: nanometres!(700.0),
                    step_size: nanometres!(1.0),
                },
            },
            collector: Collector {
                radius: Radius::Auto(mm!(0.0)),
                scheme: CollectorScheme::Partitioned {
                    domain: SphericalDomain::Upper,
                    partition: SphericalPartition::EqualArea {
                        zenith: RangeByStepCount {
                            start: deg!(0.0).in_radians(),
                            stop: deg!(90.0).in_radians(),
                            step_count: 1,
                        },
                        azimuth: RangeByStepSize {
                            start: deg!(0.0).in_radians(),
                            stop: deg!(360.0).in_radians(),
                            step_size: deg!(10.0).in_radians(),
                        },
                    },
                },
            },
        }
    }
}

impl BsdfMeasurement {
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
            },
        };

        if !is_valid_scheme.0 {
            return Err(Error::InvalidCollector(is_valid_scheme.1));
        }

        Ok(self)
    }
}

const DEFAULT_AZIMUTH_RANGE: RangeByStepSize<Radians> = RangeByStepSize {
    start: Radians::ZERO,
    stop: Radians::TWO_PI,
    step_size: deg!(5.0).in_radians(),
};

const DEFAULT_ZENITH_RANGE: RangeByStepSize<Radians> = RangeByStepSize {
    start: Radians::ZERO,
    stop: Radians::HALF_PI,
    step_size: deg!(2.0).in_radians(),
};

/// Parameters for microfacet distribution measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicrofacetAreaDistributionMeasurement {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSize<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSize<Radians>,
}

impl Default for MicrofacetAreaDistributionMeasurement {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }
}

impl MicrofacetAreaDistributionMeasurement {
    /// Returns the number of samples that will be taken along the zenith angle.
    ///
    /// Here one is added to the zenith step count to account for the zenith
    /// angle of 90°.
    pub fn zenith_step_count_inclusive(&self) -> usize { self.zenith.step_count() + 1 }

    /// Returns the number of samples that will be taken along the azimuth
    /// angle.
    ///
    /// Here no additional step is added to the azimuth step count due to that
    /// in most cases the azimuth angle sampling range will be [0°, 360°] and
    /// thus the azimuth angle of 360° will be sampled as it is the same as the
    /// azimuth angle of 0°.
    pub fn azimuth_step_count_inclusive(&self) -> usize {
        if self.azimuth.start == Radians::ZERO && self.azimuth.stop == Radians::TWO_PI {
            self.azimuth.step_count()
        } else {
            self.azimuth.step_count() + 1
        }
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

/// Parameters for microfacet shadowing masking measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicrofacetMaskingShadowingMeasurement {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSize<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSize<Radians>,
    /// Discritization resolution during the measurement (area estimation).
    pub resolution: u32,
}

impl Default for MicrofacetMaskingShadowingMeasurement {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
            resolution: 512,
        }
    }
}

impl MicrofacetMaskingShadowingMeasurement {
    /// Counts the number of samples (on hemisphere) that will be taken during
    /// the measurement.
    pub fn measurement_location_count(&self) -> usize {
        let azimuth_count =
            if self.azimuth.start == Radians::ZERO && self.azimuth.stop == Radians::TWO_PI {
                self.azimuth.step_count()
            } else {
                self.azimuth.step_count() + 1
            };
        self.azimuth.step_count();
        let zenith_count = self.zenith.step_count() + 1;
        azimuth_count * zenith_count
    }

    /// Returns the number of samples that will be taken along the zenith angle.
    ///
    /// Here one is added to the zenith step count to account for the zenith
    /// angle of 90°.
    pub fn zenith_step_count_inclusive(&self) -> usize { self.zenith.step_count() + 1 }

    /// Returns the number of samples that will be taken along the azimuth
    /// angle.
    ///
    /// Here no additional step is added to the azimuth step count due to that
    /// in most cases the azimuth angle sampling range will be [0°, 360°] and
    /// thus the azimuth angle of 360° will be sampled as it is the same as the
    /// azimuth angle of 0°.
    pub fn azimuth_step_count_inclusive(&self) -> usize {
        if self.azimuth.start == Radians::ZERO && self.azimuth.stop == Radians::TWO_PI {
            self.azimuth.step_count()
        } else {
            self.azimuth.step_count() + 1
        }
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
    Bsdf(BsdfMeasurement),
    /// Measure the micro-facet area distribution function of a micro-surface.
    #[serde(alias = "microfacet-area-distribution-function")]
    Madf(MicrofacetAreaDistributionMeasurement),
    /// Measure the micro-facet masking/shadowing function.
    #[serde(alias = "microfacet-masking-shadowing-function")]
    Mmsf(MicrofacetMaskingShadowingMeasurement),
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
    pub fn bsdf(&self) -> Option<&BsdfMeasurement> {
        if let MeasurementKindDescription::Bsdf(bsdf) = self {
            Some(bsdf)
        } else {
            None
        }
    }

    /// Get the micro-facet distribution measurement parameters.
    pub fn microfacet_distribution(&self) -> Option<&MicrofacetAreaDistributionMeasurement> {
        if let MeasurementKindDescription::Madf(mfd) = self {
            Some(mfd)
        } else {
            None
        }
    }

    /// Get the micro-surface shadowing-masking function measurement parameters.
    pub fn micro_surface_shadow_masking(&self) -> Option<&MicrofacetMaskingShadowingMeasurement> {
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
    pub fn path(&self) -> Option<&Path> {
        match self {
            MeasurementDataSource::Loaded(p) => Some(p.as_path()),
            MeasurementDataSource::Measured(_) => None,
        }
    }
}

// TODO: add support for storing data in the memory in a compressed
//       format(maybe LZ4).
/// Structure for storing measurement data in the memory especially
/// when loading from a file.
#[derive(Debug, Clone)]
pub struct MeasurementData {
    pub kind: MeasurementKind,
    pub azimuth: AngleRange,
    pub zenith: AngleRange,
    pub source: MeasurementDataSource,
    /// Internal tag for displaying the measurement data in the GUI.
    pub name: String,
    pub data: Vec<f32>,
}

impl Asset for MeasurementData {}

impl Plottable for MeasurementData {
    fn mode(&self) -> PlottingMode {
        match self.kind {
            MeasurementKind::Bsdf => PlottingMode::Bsdf,
            MeasurementKind::MicrofacetAreaDistribution => PlottingMode::Adf,
            MeasurementKind::MicrofacetMaskingShadowing => PlottingMode::Msf,
        }
    }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

impl MeasurementData {
    /// Loads the measurement data from a file.
    pub fn read_from_file(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let (header, data) = io::vgmo::read(&mut reader).map_err(|err| {
            Error::ReadFile(io::ReadFileError {
                path: path.to_owned().into_boxed_path(),
                kind: err,
            })
        })?;
        let path = path.to_path_buf();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("invalid file stem")
            .to_string();
        Ok(MeasurementData {
            kind: header.kind,
            azimuth: header.azimuth_range,
            zenith: header.zenith_range,
            source: MeasurementDataSource::Loaded(path),
            name,
            data,
        })
    }

    /// Returns the Area Distribution Function data slice for the given
    /// azimuthal angle in radians.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    ///
    /// Azimuthal angle will be wrapped around to the range [0, 2π].
    ///
    /// 2π will be mapped to 0.
    pub fn adf_data_slice(&self, azimuth: f32) -> (&[f32], Option<&[f32]>) {
        debug_assert!(self.kind == MeasurementKind::MicrofacetAreaDistribution);
        let angle = {
            azimuth - (0.5 * azimuth / std::f32::consts::PI).floor() * std::f32::consts::PI * 2.0
        };

        let index = ((angle - self.azimuth.start) / self.azimuth.bin_width) as usize;

        let mut opposite_angle = angle + std::f32::consts::PI;
        if opposite_angle > std::f32::consts::PI * 2.0 {
            opposite_angle -= std::f32::consts::PI * 2.0;
        }
        if opposite_angle < 0.0 {
            opposite_angle += std::f32::consts::PI * 2.0;
        }

        let opposite_index =
            if self.azimuth.start <= opposite_angle && opposite_angle <= self.azimuth.end {
                Some(((opposite_angle - self.azimuth.start) / self.azimuth.bin_width) as usize)
            } else {
                None
            };

        (
            self.adf_data_slice_by_azimuth_index(index),
            opposite_index.map(|index| self.adf_data_slice_by_azimuth_index(index)),
        )
    }

    /// Returns a data slice of the Area Distribution Function for the given
    /// azimuthal angle index.
    pub fn adf_data_slice_by_azimuth_index(&self, index: usize) -> &[f32] {
        debug_assert!(self.kind == MeasurementKind::MicrofacetAreaDistribution);
        debug_assert!(
            index < self.azimuth.bin_count as usize,
            "index out of range"
        );
        &self.data
            [index * self.zenith.bin_count as usize..(index + 1) * self.zenith.bin_count as usize]
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
        azimuth_m: f32,
        zenith_m: f32,
        azimuth_i: f32,
    ) -> (&[f32], Option<&[f32]>) {
        debug_assert!(
            self.kind == MeasurementKind::MicrofacetMaskingShadowing,
            "measurement data kind should be MicrofacetMaskingShadowing"
        );
        let zenith_m = zenith_m.max(0.0).min(std::f32::consts::FRAC_PI_2);
        let azimuth_m = {
            azimuth_m
                - (0.5 * azimuth_m / std::f32::consts::PI).floor() * std::f32::consts::PI * 2.0
        };
        let azimuth_m_idx = ((azimuth_m - self.azimuth.start) / self.azimuth.bin_width) as usize;
        let zenith_m_idx = ((zenith_m - self.zenith.start) / self.zenith.bin_width) as usize;

        let azimuth_i = {
            azimuth_i
                - (0.5 * azimuth_i / std::f32::consts::PI).floor() * std::f32::consts::PI * 2.0
        };
        let azimuth_i_idx = ((azimuth_i - self.azimuth.start) / self.azimuth.bin_width) as usize;

        let mut opposite_azimuth_i = azimuth_i + std::f32::consts::PI;
        if opposite_azimuth_i > std::f32::consts::PI * 2.0 {
            opposite_azimuth_i -= std::f32::consts::PI * 2.0;
        }
        if opposite_azimuth_i < 0.0 {
            opposite_azimuth_i += std::f32::consts::PI * 2.0;
        }

        let opposite_azimuth_i_idx =
            if self.azimuth.start <= opposite_azimuth_i && opposite_azimuth_i <= self.azimuth.end {
                Some(((opposite_azimuth_i - self.azimuth.start) / self.azimuth.bin_width) as usize)
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
        debug_assert!(self.kind == MeasurementKind::MicrofacetMaskingShadowing);
        debug_assert!(
            azimuth_m_idx < self.azimuth.bin_count as usize,
            "index out of range"
        );
        debug_assert!(
            azimuth_i_idx < self.azimuth.bin_count as usize,
            "index out of range"
        );
        debug_assert!(
            zenith_m_idx < self.zenith.bin_count as usize,
            "index out of range"
        );
        let offset = azimuth_m_idx * zenith_m_idx * azimuth_m_idx * self.zenith.bin_count as usize;
        &self.data[offset..offset + self.zenith.bin_count as usize]
    }
}

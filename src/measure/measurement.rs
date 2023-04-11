//! Measurement parameters.
use crate::{
    common::{Medium, RangeByStepCount, RangeByStepSize, SphericalDomain, SphericalPartition},
    measure::{
        bsdf::BsdfKind, collector::CollectorScheme, emitter::RegionShape, Collector, Emitter,
        RtcMethod,
    },
    units::{deg, mm, nanometres, rad, Millimetres, Radians, SolidAngle},
    Error,
};
use serde::{Deserialize, Serialize};
use std::{
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
pub struct MicrofacetNormalDistributionMeasurement {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSize<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSize<Radians>,
}

impl Default for MicrofacetNormalDistributionMeasurement {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }
}

impl MicrofacetNormalDistributionMeasurement {
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
pub enum MeasurementKind {
    /// Measure the BSDF of a micro-surface.
    Bsdf(BsdfMeasurement),
    /// Measure the micro-facet normal distribution function of a micro-surface.
    #[serde(alias = "microfacet-normal-distribution-function")]
    Mndf(MicrofacetNormalDistributionMeasurement),
    /// Measure the micro-facet masking/shadowing function.
    #[serde(alias = "microfacet-masking-shadowing-function")]
    Mmsf(MicrofacetMaskingShadowingMeasurement),
}

impl MeasurementKind {
    /// Whether the measurement parameters are valid.
    pub fn validate(self) -> Result<Self, Error> {
        match self {
            Self::Bsdf(bsdf) => Ok(Self::Bsdf(bsdf.validate()?)),
            Self::Mndf(mfd) => Ok(Self::Mndf(mfd.validate()?)),
            Self::Mmsf(mfs) => Ok(Self::Mmsf(mfs.validate()?)),
        }
    }

    /// Whether the measurement is a BSDF measurement.
    pub fn is_bsdf(&self) -> bool { matches!(self, Self::Bsdf { .. }) }

    /// Whether the measurement is a micro-facet distribution measurement.
    pub fn is_microfacet_distribution(&self) -> bool { matches!(self, Self::Mndf { .. }) }

    /// Whether the measurement is a micro-surface shadowing-masking function
    /// measurement.
    pub fn is_micro_surface_shadow_masking(&self) -> bool { matches!(self, Self::Mmsf { .. }) }

    /// Get the BSDF measurement parameters.
    pub fn bsdf(&self) -> Option<&BsdfMeasurement> {
        if let MeasurementKind::Bsdf(bsdf) = self {
            Some(bsdf)
        } else {
            None
        }
    }

    /// Get the micro-facet distribution measurement parameters.
    pub fn microfacet_distribution(&self) -> Option<&MicrofacetNormalDistributionMeasurement> {
        if let MeasurementKind::Mndf(mfd) = self {
            Some(mfd)
        } else {
            None
        }
    }

    /// Get the micro-surface shadowing-masking function measurement parameters.
    pub fn micro_surface_shadow_masking(&self) -> Option<&MicrofacetMaskingShadowingMeasurement> {
        if let MeasurementKind::Mmsf(mfd) = self {
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
    pub kind: MeasurementKind,
    /// Surfaces to be measured. surface's path can be prefixed with either
    /// `usr://` or `sys://` to indicate the user-defined data file path
    /// or system-defined data file path.
    pub surfaces: Vec<PathBuf>, // TODO(yang): use Cow<'a, Vec<PathBuf>> to avoid copying the path
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
        let details = self.kind.validate()?;
        Ok(Self {
            kind: details,
            ..self
        })
    }

    /// Measurement kind in the form of a string.
    pub fn name(&self) -> &'static str {
        match self.kind {
            MeasurementKind::Bsdf { .. } => "BSDF measurement",
            MeasurementKind::Mndf { .. } => "microfacet-distribution measurement",
            MeasurementKind::Mmsf { .. } => "micro-surface-shadow-masking measurement",
        }
    }
}

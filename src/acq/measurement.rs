use crate::{
    acq::{
        bsdf::BsdfKind,
        collector::CollectorScheme,
        emitter::RegionShape,
        util::{RangeByStepCount, RangeByStepSize, SphericalDomain, SphericalPartition},
        Collector, Emitter, Medium, RtcMethod,
    },
    units::{degrees, metres, nanometres, radians, Metres, Radians, SolidAngle},
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
    Auto(#[serde(skip)] Metres),

    /// Radius is given explicitly.
    Fixed(Metres),
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
    pub fn value(&self) -> Metres {
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
    GeomOptics {
        /// Method used to trace rays.
        method: RtcMethod,
    },

    /// Wave optics is used during the simulation.
    WaveOptics,
}

/// Parameters for BSDF measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BsdfMeasurement {
    /// The measurement kind.
    pub bsdf_kind: BsdfKind,

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
            bsdf_kind: BsdfKind::InPlaneBrdf,
            sim_kind: SimulationKind::GeomOptics {
                method: RtcMethod::Grid,
            },
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Air,
            emitter: Emitter {
                num_rays: 1000,
                max_bounces: 10,
                radius: Radius::Auto(metres!(0.0)),
                zenith: RangeByStepSize::<Radians> {
                    start: degrees!(0.0).in_radians(),
                    stop: degrees!(90.0).in_radians(),
                    step_size: degrees!(5.0).in_radians(),
                },
                azimuth: RangeByStepSize::<Radians> {
                    start: degrees!(0.0).in_radians(),
                    stop: degrees!(360.0).in_radians(),
                    step_size: degrees!(120.0).in_radians(),
                },
                shape: RegionShape::SphericalCap {
                    zenith: degrees!(5.0).in_radians(),
                },
                solid_angle: SolidAngle::from_angle_ranges(
                    (radians!(0.0), degrees!(5.0).in_radians()),
                    (radians!(0.0), radians!(2.0 * std::f32::consts::PI)),
                ),
                spectrum: RangeByStepSize {
                    start: nanometres!(400.0),
                    stop: nanometres!(700.0),
                    step_size: nanometres!(1.0),
                },
                samples: vec![],
            },
            collector: Collector {
                radius: Radius::Auto(metres!(0.0)),
                scheme: CollectorScheme::Partitioned {
                    domain: SphericalDomain::Upper,
                    partition: SphericalPartition::EqualArea {
                        zenith: RangeByStepCount {
                            start: degrees!(0.0).in_radians(),
                            stop: degrees!(90.0).in_radians(),
                            step_count: 0,
                        },
                        azimuth: RangeByStepSize {
                            start: degrees!(0.0).in_radians(),
                            stop: degrees!(0.0).in_radians(),
                            step_size: degrees!(0.0).in_radians(),
                        },
                    },
                },
                patches: None,
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
            && emitter.zenith.step_size > radians!(0.0)
            && emitter.azimuth.step_size > radians!(0.0))
        {
            return Err(Error::InvalidEmitter(
                "emitter's radius, zenith step size and azimuth step size must be positive",
            ));
        }

        match emitter.shape {
            RegionShape::SphericalCap { zenith } => {
                if zenith <= radians!(0.0) || zenith > degrees!(360.0) {
                    return Err(Error::InvalidEmitter(
                        "emitter's zenith angle must be in the range [0°, 360°]",
                    ));
                }
            }
            RegionShape::SphericalRect { zenith, azimuth } => {
                if zenith.0 <= radians!(0.0) || zenith.1 > degrees!(360.0) {
                    return Err(Error::InvalidEmitter(
                        "emitter's zenith angle must be in the range [0°, 360°]",
                    ));
                }

                if azimuth.0 <= radians!(0.0) || azimuth.1 > degrees!(360.0) {
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
            CollectorScheme::Individual { shape, .. } => match shape {
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
    step_size: degrees!(5.0).in_radians(),
};

const DEFAULT_ZENITH_RANGE: RangeByStepSize<Radians> = RangeByStepSize {
    start: Radians::ZERO,
    stop: Radians::HALF_PI,
    step_size: degrees!(2.0).in_radians(),
};

/// Parameters for microfacet distribution measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicrofacetDistributionMeasurement {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSize<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSize<Radians>,
}

impl Default for MicrofacetDistributionMeasurement {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }
}

impl MicrofacetDistributionMeasurement {
    /// Validate the parameters.
    pub fn validate(self) -> Result<Self, Error> {
        if !(self.azimuth.start >= Radians::ZERO && self.azimuth.stop <= Radians::TWO_PI) {
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
        if !(self.azimuth.step_size > radians!(0.0) && self.zenith.step_size > radians!(0.0)) {
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
pub struct MicrofacetShadowingMaskingMeasurement {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSize<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSize<Radians>,
    /// Discritization resolution during the measurement (area estimation).
    pub resolution: u32,
}

impl Default for MicrofacetShadowingMaskingMeasurement {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
            resolution: 512,
        }
    }
}

impl MicrofacetShadowingMaskingMeasurement {
    pub fn validate(self) -> Result<Self, Error> {
        if !(self.azimuth.start >= Radians::ZERO && self.azimuth.stop <= Radians::TWO_PI) {
            return Err(Error::InvalidParameter(
                "Microfacet shadowing-masking measurement: azimuth angle must be in the range \
                 [0°, 360°]",
            ));
        }
        if !(self.zenith.start >= Radians::ZERO && self.zenith.start <= Radians::HALF_PI) {
            return Err(Error::InvalidParameter(
                "Microfacet shadowing-masking measurement: zenith angle must be in the range [0°, \
                 90°]",
            ));
        }
        if !(self.azimuth.step_size > radians!(0.0) && self.zenith.step_size > radians!(0.0)) {
            return Err(Error::InvalidParameter(
                "Microfacet shadowing-masking measurement: azimuth and zenith step sizes must be \
                 positive!",
            ));
        }

        Ok(self)
    }
}

/// Describes the different kind of measuremnts with parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementKind {
    /// Measure the BSDF of a micro-surface.
    Bsdf(BsdfMeasurement),
    /// Measure the micro-facet distribution of a micro-surface.
    MicrofacetDistribution(MicrofacetDistributionMeasurement),
    /// Measure the micro-surface shadowing-masking functions.
    MicrofacetShadowingMasking(MicrofacetShadowingMaskingMeasurement),
}

impl MeasurementKind {
    /// Whether the measurement parameters are valid.
    pub fn validate(self) -> Result<Self, Error> {
        match self {
            Self::Bsdf(bsdf) => Ok(Self::Bsdf(bsdf.validate()?)),
            Self::MicrofacetDistribution(mfd) => Ok(Self::MicrofacetDistribution(mfd.validate()?)),
            Self::MicrofacetShadowingMasking(mfs) => {
                Ok(Self::MicrofacetShadowingMasking(mfs.validate()?))
            }
        }
    }
}

impl MeasurementKind {
    /// Whether the measurement is a BSDF measurement.
    pub fn is_bsdf(&self) -> bool { matches!(self, Self::Bsdf { .. }) }

    /// Whether the measurement is a micro-facet distribution measurement.
    pub fn is_microfacet_distribution(&self) -> bool {
        matches!(self, Self::MicrofacetDistribution { .. })
    }

    /// Whether the measurement is a micro-surface shadowing-masking function
    /// measurement.
    pub fn is_micro_surface_shadow_masking(&self) -> bool {
        matches!(self, Self::MicrofacetShadowingMasking { .. })
    }

    /// Get the BSDF measurement parameters.
    pub fn bsdf(&self) -> Option<&BsdfMeasurement> {
        if let MeasurementKind::Bsdf(bsdf) = self {
            Some(bsdf)
        } else {
            None
        }
    }

    /// Get the micro-facet distribution measurement parameters.
    pub fn microfacet_distribution(&self) -> Option<&MicrofacetDistributionMeasurement> {
        if let MeasurementKind::MicrofacetDistribution(mfd) = self {
            Some(mfd)
        } else {
            None
        }
    }

    /// Get the micro-surface shadowing-masking function measurement parameters.
    pub fn micro_surface_shadow_masking(&self) -> Option<&MicrofacetShadowingMaskingMeasurement> {
        if let MeasurementKind::MicrofacetShadowingMasking(mfd) = self {
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
            MeasurementKind::MicrofacetDistribution { .. } => "microfacet-distribution measurement",
            MeasurementKind::MicrofacetShadowingMasking { .. } => {
                "micro-surface-shadow-masking measurement"
            }
        }
    }
}

use crate::{
    acq::{
        bsdf::BsdfKind,
        collector::CollectorScheme,
        degrees,
        emitter::RegionShape,
        nanometres, radians,
        util::{RangeByStepSize, RangeByStepCount, SphericalDomain, SphericalPartition},
        Collector, Emitter, Medium, Metres, Radians,
        RayTracingMethod, SolidAngle,
    },
    Error,
};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use std::fmt::{Display, Formatter};
use crate::acq::metres;
use serde::{Deserialize, Serialize};

/// Describes the radius of measurement.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Radius {
    /// Radius is dynamically deduced from the dimension of the surface.
    /// Note: This value will be updated before each measurement.
    Auto(#[serde(skip)]Metres),

    /// Radius is given explicitly.
    Fixed(Metres),
}

impl Display for Radius {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto(_) => write!(f, "auto"),
            Self::Fixed(v) => write!(f, "fixed: {}", v),
        }
    }
}

impl Radius {
    /// Whether the radius is dynamically deduced from the dimension of the surface.
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

/// Supported type of measurement.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MeasurementKind {
    /// Bidirectional Scattering Distribution Function
    Bsdf(BsdfKind),

    /// Normal Distribution Function
    Ndf,
}

/// Possible ways to conduct a simulation.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationKind {
    /// Ray optics is used during the simulation.
    GeomOptics { method: RayTracingMethod },

    /// Wave optics is used during the simulation.
    WaveOptics,
}

/// Description of the measurement.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// The measurement kind.
    pub kind: MeasurementKind,

    /// The simulation kind.
    pub sim_kind: SimulationKind,

    /// Incident medium of the measurement.
    pub incident_medium: Medium,

    /// Transmitted medium of the measurement (medium of the surface).
    pub transmitted_medium: Medium,

    /// Surfaces to be measured. surface's path can be prefixed with either
    /// `user://` or `local://` to indicate the user-defined data file path or
    /// system-defined data file path.
    pub surfaces: Vec<PathBuf>,

    /// Description of the emitter.
    pub emitter: Emitter,

    /// Description of the collector.
    pub collector: Collector,
}

impl Measurement {
    /// Load measurement descriptions from a file. A file may contain multiple
    /// descriptions.
    pub fn load_from_file(filepath: &Path) -> Result<Vec<Measurement>, Error> {
        let mut file = File::open(filepath)?;
        let content = {
            let mut str_ = String::new();
            file.read_to_string(&mut str_)?;
            str_
        };
        // TODO: invoke generation of patches and samples
        let measurements = serde_yaml::Deserializer::from_str(&content)
            .into_iter()
            .map(|doc|
                Measurement::deserialize(doc).map_err(Error::from)
                    .and_then(|mut measurement| {
                        measurement.validate()
                    })
            )
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(measurements)
    }

    /// Validate the measurement description.
    pub fn validate(self) -> Result<Self, Error> {
        log::info!("Validating measurement description...");
        let emitter = &self.emitter;
        let collector = &self.collector;
        if emitter.num_rays < 1 {
            return Err(Error::InvalidEmitter("number of rays must be at least 1"));
        }

        if emitter.max_bounces < 1 {
            return Err(Error::InvalidEmitter("maximum number of bounces must be at least 1"));
        }

        if !(emitter.radius.is_valid() && emitter.zenith.step_size > radians!(0.0) && emitter.azimuth.step_size > radians!(0.0)) {
            return Err(Error::InvalidEmitter("emitter's radius, zenith step size and azimuth step size must be positive"));
        }

        if !collector.radius.is_valid() {
            return Err(Error::InvalidCollector("collector's radius must be positive"));
        }

        let is_valid_scheme = match &collector.scheme {
            CollectorScheme::Partitioned { partition, .. } => {
                match partition {
                    SphericalPartition::EqualAngle { zenith, azimuth } => {
                        (zenith.start != zenith.stop && azimuth.start != azimuth.stop, "zenith and azimuth's start and stop must not be equal")
                    }
                    SphericalPartition::EqualArea { zenith, azimuth } => {
                        (zenith.start != zenith.stop && azimuth.start != azimuth.stop, "zenith and azimuth's start and stop must not be equal")
                    }
                    SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                        (zenith.start != zenith.stop && azimuth.start != azimuth.stop, "zenith and azimuth's start and stop must not be equal")
                    }
                }
            }
            CollectorScheme::Individual { shape, .. } => {
                match shape {
                    RegionShape::SphericalCap { zenith } => {
                        if !zenith.is_positive() {
                            (false, "collector's zenith angle must be positive")
                        } else {
                            (true, "")
                        }
                    }
                    RegionShape::SphericalRect { .. } => {
                        (true, "")
                    }
                }
            }
        };

        if !is_valid_scheme.0 {
            return Err(Error::InvalidCollector(is_valid_scheme.1));
        }

        Ok(self)
    }
}

impl Default for Measurement {
    fn default() -> Self {
        Self {
            kind: MeasurementKind::Ndf,
            sim_kind: SimulationKind::GeomOptics {
                method: RayTracingMethod::Grid,
            },
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Air,
            surfaces: vec![],
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
                    domain: SphericalDomain::UpperHemisphere,
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

#[test]
fn scene_desc_serialization() {
    use std::io::{Cursor, Write};
    use crate::acq::steradians;

    let desc = Measurement {
        kind: MeasurementKind::Bsdf(BsdfKind::InPlaneBrdf),
        sim_kind: SimulationKind::WaveOptics,
        incident_medium: Medium::Air,
        transmitted_medium: Medium::Air,
        surfaces: vec![PathBuf::from("/tmp/scene.obj")],
        collector: Collector {
            radius: Radius::Auto(metres!(0.0)),
            scheme: CollectorScheme::Partitioned {
                domain: SphericalDomain::UpperHemisphere,
                partition: SphericalPartition::EqualArea {
                    zenith: RangeByStepCount {
                        start: radians!(0.0),
                        stop: radians!(0.0),
                        step_count: 0,
                    },
                    azimuth: RangeByStepSize {
                        start: radians!(0.0),
                        stop: radians!(0.0),
                        step_size: radians!(0.0),
                    },
                },
            },
            patches: None,
        },
        emitter: Emitter {
            num_rays: 0,
            max_bounces: 0,
            radius: Radius::Auto(metres!(0.0)),
            spectrum: RangeByStepSize {
                start: nanometres!(380.0),
                stop: nanometres!(780.0),
                step_size: nanometres!(10.0),
            },
            solid_angle: steradians!(0.0),
            zenith: RangeByStepSize {
                start: radians!(0.0),
                stop: radians!(90.0),
                step_size: radians!(0.0),
            },
            azimuth: RangeByStepSize {
                start: radians!(0.0),
                stop: radians!(360.0),
                step_size: radians!(0.0),
            },
            shape: RegionShape::SphericalCap {
                zenith: radians!(0.0),
            },
            samples: vec![],
        },
    };

    let serialized = serde_yaml::to_string(&desc).unwrap();

    let mut file = Cursor::new(vec![0u8; 128]);
    file.write_all(serialized.as_bytes()).unwrap();

    println!("{}", serialized);

    file.set_position(0);
    let deserialized_0: Measurement = serde_yaml::from_reader(file).unwrap();
    let deserialized_1: Measurement = serde_yaml::from_str(&serialized).unwrap();

    assert_eq!(desc, deserialized_0);
    assert_eq!(desc, deserialized_1);
    assert_eq!(deserialized_0, deserialized_1);
}

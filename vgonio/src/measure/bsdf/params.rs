use crate::{
    error::RuntimeError,
    measure::{
        bsdf::{
            detector::{Detector, DetectorScheme},
            emitter::{Emitter, RegionShape},
            rtc::RtcMethod,
            BsdfKind,
        },
        params::Radius,
    },
    Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalPartition,
};
use serde::{Deserialize, Serialize};
use vgcore::{
    error::VgonioError,
    units::{deg, mm, nm, rad, SolidAngle},
};

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

    /// Description of the detector.
    pub detector: Detector,
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
                spectrum: RangeByStepSizeInclusive::new(nm!(400.0), nm!(700.0), nm!(100.0)),
            },
            detector: Detector {
                radius: Radius::Auto(mm!(0.0)),
                scheme: DetectorScheme::Partitioned {
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
    pub fn bsdf_data_samples_count(&self) -> usize { self.emitter.samples_count() }

    /// Whether the measurement parameters are valid.
    pub fn validate(self) -> Result<Self, VgonioError> {
        log::info!("Validating measurement description...");
        let emitter = &self.emitter;
        if emitter.num_rays < 1 {
            return Err(VgonioError::new(
                "Number of rays must be at least 1",
                Some(Box::new(RuntimeError::InvalidEmitter)),
            ));
        }

        if emitter.max_bounces < 1 {
            return Err(VgonioError::new(
                "Number of bounces must be at least 1",
                Some(Box::new(RuntimeError::InvalidEmitter)),
            ));
        }

        if !(emitter.orbit_radius().is_valid()
            && emitter.zenith.step_size > rad!(0.0)
            && emitter.azimuth.step_size > rad!(0.0))
        {
            return Err(VgonioError::new(
                "Emitter's orbit radius, zenith and azimuth must be positive",
                Some(Box::new(RuntimeError::InvalidEmitter)),
            ));
        }

        match emitter.shape {
            RegionShape::SphericalCap { zenith } => {
                if zenith <= rad!(0.0) || zenith > deg!(360.0) {
                    return Err(VgonioError::new(
                        "Emitter's zenith angle must be in the range [0°, 90°]",
                        Some(Box::new(RuntimeError::InvalidEmitter)),
                    ));
                }
            }
            RegionShape::SphericalRect { zenith, azimuth } => {
                if zenith.0 <= rad!(0.0) || zenith.1 > deg!(360.0) {
                    return Err(VgonioError::new(
                        "Emitter's zenith angle must be in the range [0°, 90°]",
                        Some(Box::new(RuntimeError::InvalidEmitter)),
                    ));
                }

                if azimuth.0 <= rad!(0.0) || azimuth.1 > deg!(360.0) {
                    return Err(VgonioError::new(
                        "Emitter's azimuth angle must be in the range [0°, 360°]",
                        Some(Box::new(RuntimeError::InvalidEmitter)),
                    ));
                }
            }
            _ => {}
        }

        if emitter.spectrum.start < nm!(0.0) || emitter.spectrum.stop < nm!(0.0) {
            return Err(VgonioError::new(
                "Emitter's spectrum must be positive",
                Some(Box::new(RuntimeError::InvalidEmitter)),
            ));
        }

        if emitter.spectrum.start > emitter.spectrum.stop {
            return Err(VgonioError::new(
                "Emitter's spectrum start must be less than its stop",
                Some(Box::new(RuntimeError::InvalidEmitter)),
            ));
        }

        let collector = &self.detector;
        if !collector.radius.is_valid() {
            return Err(VgonioError::new(
                "Collector's radius must be positive",
                Some(Box::new(RuntimeError::InvalidCollector)),
            ));
        }

        let is_valid_scheme = match &collector.scheme {
            DetectorScheme::Partitioned { partition, .. } => match partition {
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
            DetectorScheme::SingleRegion { shape, .. } => match shape {
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
            return Err(VgonioError::new(
                is_valid_scheme.1,
                Some(Box::new(RuntimeError::InvalidCollector)),
            ));
        }

        Ok(self)
    }
}

use crate::{
    error::RuntimeError,
    measure::bsdf::{emitter::EmitterParams, receiver::ReceiverParams, rtc::RtcMethod, BsdfKind},
};
use serde::{Deserialize, Serialize};
use vgn_core::{
    error::VgonioError,
    math::Sph2,
    units::{deg, nm, rad},
    utils::{
        medium::MediumId,
        partition::{PartitionScheme, SphericalDomain},
        range::StepRangeIncl,
    },
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

    #[track_caller]
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::GeomOptics(RtcMethod::Grid)),
            0x01 => Ok(Self::GeomOptics(RtcMethod::Embree)),
            0x02 => Ok(Self::GeomOptics(RtcMethod::Optix)),
            0x03 => Ok(Self::WaveOptics),
            _ => Err(format!("Invalid simulation kind {}", value)),
        }
    }
}

impl SimulationKind {
    /// Returns the value as an u8. The verse of `TryFrom<u8>`.
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::GeomOptics(RtcMethod::Grid) => 0x00,
            Self::GeomOptics(RtcMethod::Embree) => 0x01,
            Self::GeomOptics(RtcMethod::Optix) => 0x02,
            Self::WaveOptics => 0x03,
        }
    }

    /// Verifies whether the selected simulation method is available in the
    /// current build.
    pub fn check_supported_for_measurement(&self) -> Result<(), String> {
        match self {
            Self::GeomOptics(RtcMethod::Embree) => {
                if cfg!(feature = "embree") {
                    Ok(())
                } else {
                    Err(
                        "SimulationKind is Embree, but this build does not enable `embree`."
                            .to_string(),
                    )
                }
            },
            Self::GeomOptics(RtcMethod::Optix) => {
                if cfg!(feature = "optix") {
                    Err("Optix simulation is not implemented.".to_string())
                } else {
                    Err(
                        "SimulationKind is Optix, but this build does not enable `optix`."
                            .to_string(),
                    )
                }
            },
            Self::GeomOptics(RtcMethod::Grid) => {
                Err("Grid simulation is temporarily deactivated.".to_string())
            },
            Self::WaveOptics => Err("Wave optics simulation is not yet implemented.".to_string()),
        }
    }
}

/// Parameters for BSDF measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BsdfMeasurementParams {
    /// The measurement kind.
    pub kind: BsdfKind,

    /// The simulation kind.
    pub sim_kind: SimulationKind,

    /// Incident medium of the measurement.
    pub incident_medium: MediumId,

    /// Transmitted medium of the measurement (surface medium).
    pub transmitted_medium: MediumId,

    /// Description of the emitter.
    pub emitter: EmitterParams,

    /// Description of the detector.
    pub receivers: Vec<ReceiverParams>,

    /// Whether to apply the Fresnel equations during the simulation.
    pub fresnel: bool,
}

impl Default for BsdfMeasurementParams {
    fn default() -> Self {
        Self {
            kind: BsdfKind::Brdf,
            #[cfg(feature = "embree")]
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Embree),
            #[cfg(not(feature = "embree"))]
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Grid),
            incident_medium: MediumId::AIR,
            transmitted_medium: MediumId::AIR,
            emitter: EmitterParams {
                num_rays: 1000,
                num_sectors: 1,
                max_bounces: 10,
                zenith: StepRangeIncl::new(
                    deg!(0.0).in_radians(),
                    deg!(90.0).in_radians(),
                    deg!(5.0).in_radians(),
                ),
                azimuth: StepRangeIncl::new(
                    deg!(0.0).in_radians(),
                    deg!(360.0).in_radians(),
                    deg!(60.0).in_radians(),
                ),
                spectrum: StepRangeIncl::new(nm!(400.0), nm!(700.0), nm!(100.0)),
            },
            receivers: vec![ReceiverParams {
                domain: SphericalDomain::Upper,
                precision: Sph2::new(deg!(2.0).in_radians(), deg!(5.0).in_radians()),
                scheme: PartitionScheme::Beckers,
            }],
            fresnel: true,
        }
    }
}

impl BsdfMeasurementParams {
    /// Returns the number of measurement points.
    pub fn n_wi(&self) -> usize { self.emitter.measurement_points_count() }

    /// Returns the number of wavelengths in the spectrum.
    pub fn n_spectrum(&self) -> usize { self.emitter.spectrum.step_count() }

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

        if !(emitter.zenith.step_size > rad!(0.0) && emitter.azimuth.step_size > rad!(0.0)) {
            return Err(VgonioError::new(
                "Emitter's orbit radius, zenith and azimuth must be positive",
                Some(Box::new(RuntimeError::InvalidEmitter)),
            ));
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

        for collector in &self.receivers {
            if !collector.precision.is_positive() {
                return Err(VgonioError::new(
                    "Detector's precision must be positive",
                    Some(Box::new(RuntimeError::InvalidDetector)),
                ));
            }
        }

        Ok(self)
    }

    /// Returns the total number of samples that will be collected.
    pub fn samples_count(&self, i: usize) -> Option<usize> {
        self.receivers
            .get(i)
            .map(|r| r.num_patches() * self.emitter.measurement_points_count())
    }

    /// Verifies whether the selected simulation method is available in the
    /// current build.
    pub fn check_supported_for_measurement(&self) -> Result<(), String> {
        self.sim_kind.check_supported_for_measurement()
    }

    /// Checks if the incident and transmitted media are air.
    pub fn is_both_air_medium(&self) -> bool {
        self.incident_medium == MediumId::AIR && self.transmitted_medium == MediumId::AIR
    }

    /// Returns the parameters as a HashMap.
    pub fn to_exr_extra_info(
        &self,
    ) -> std::collections::HashMap<exr::meta::attribute::Text, exr::meta::attribute::AttributeValue>
    {
        // TODO: handle multiple receivers
        use exr::meta::attribute::{AttributeValue, Text};
        let mut hash_map = std::collections::HashMap::new();
        hash_map.insert(
            Text::new_or_panic("vg.kind"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.kind))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.sim_kind"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.sim_kind))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.incident_medium"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.incident_medium))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.transmitted_medium"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.transmitted_medium))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.fresnel"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.fresnel))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.num_rays"),
            AttributeValue::I32(self.emitter.num_rays as i32),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.max_bounces"),
            AttributeValue::I32(self.emitter.max_bounces as i32),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.zenith.start"),
            AttributeValue::F32(self.emitter.zenith.start.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.zenith.stop"),
            AttributeValue::F32(self.emitter.zenith.stop.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.zenith.step_size"),
            AttributeValue::F32(self.emitter.zenith.step_size.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.azimuth.start"),
            AttributeValue::F32(self.emitter.azimuth.start.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.azimuth.stop"),
            AttributeValue::F32(self.emitter.azimuth.stop.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.azimuth.step_size"),
            AttributeValue::Text(Text::new_or_panic(format!(
                "{}",
                self.emitter.azimuth.step_size.in_degrees().as_f32()
            ))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.spectrum.start"),
            AttributeValue::F32(self.emitter.spectrum.start.as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.spectrum.stop"),
            AttributeValue::F32(self.emitter.spectrum.stop.as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.emitter.spectrum.step_size"),
            AttributeValue::F32(self.emitter.spectrum.step_size.as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("vg.detector.domain"),
            AttributeValue::Text(Text::new_or_panic(format!(
                "{:?}",
                self.receivers[0].domain
            ))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.detector.scheme"),
            AttributeValue::Text(Text::new_or_panic(format!(
                "{:?}",
                self.receivers[0].scheme
            ))),
        );
        hash_map.insert(
            Text::new_or_panic("vg.detector.precision_theta"),
            AttributeValue::F32(self.receivers[0].precision.theta.in_degrees().as_f32()),
        );
        if self.receivers[0].scheme == PartitionScheme::EqualAngle {
            hash_map.insert(
                Text::new_or_panic("vg.detector.precision_phi"),
                AttributeValue::F32(self.receivers[0].precision.phi.in_degrees().as_f32()),
            );
        }
        hash_map
    }

    #[cfg(test)]
    /// Creates a minimal set of parameters for testing.
    pub fn default_for_test() -> Self {
        Self {
            kind: BsdfKind::Brdf,
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Embree),
            incident_medium: MediumId::VACUUM,
            transmitted_medium: MediumId::AL,
            emitter: EmitterParams::default_for_test(),
            receivers: vec![
                ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision: Sph2 {
                        theta: rad!(10.0),
                        phi: rad!(10.0),
                    },
                    scheme: PartitionScheme::Beckers,
                },
                ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision: Sph2 {
                        theta: rad!(45.0),
                        phi: rad!(2.0),
                    },
                    scheme: PartitionScheme::EqualAngle,
                },
            ],
            fresnel: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_kind_decodes_well_known_values() {
        assert!(matches!(
            SimulationKind::try_from(0x00),
            Ok(SimulationKind::GeomOptics(RtcMethod::Grid))
        ));
        assert!(matches!(
            SimulationKind::try_from(0x01),
            Ok(SimulationKind::GeomOptics(RtcMethod::Embree))
        ));
        assert!(matches!(
            SimulationKind::try_from(0x02),
            Ok(SimulationKind::GeomOptics(RtcMethod::Optix))
        ));
        assert!(matches!(
            SimulationKind::try_from(0x03),
            Ok(SimulationKind::WaveOptics)
        ));
    }

    #[test]
    fn simulation_kind_rejects_unknown_values() {
        assert!(SimulationKind::try_from(0xFF).is_err());
    }

    #[test]
    fn simulation_kind_support_check_matches_enabled_backends() {
        let grid = SimulationKind::GeomOptics(RtcMethod::Grid);
        let embree = SimulationKind::GeomOptics(RtcMethod::Embree);
        let optix = SimulationKind::GeomOptics(RtcMethod::Optix);
        let wave = SimulationKind::WaveOptics;

        assert!(grid.check_supported_for_measurement().is_err());
        assert!(wave.check_supported_for_measurement().is_err());

        #[cfg(feature = "embree")]
        assert!(embree.check_supported_for_measurement().is_ok());
        #[cfg(not(feature = "embree"))]
        assert!(embree.check_supported_for_measurement().is_err());

        assert!(optix.check_supported_for_measurement().is_err());
    }
}

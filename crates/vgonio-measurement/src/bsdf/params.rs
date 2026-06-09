use crate::{
    bsdf::{emitter::EmitterParams, receiver::ReceiverParams, rtc::RtcMethod, BsdfKind},
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
    pub fn check_supported_for_measurement(
        &self,
        backend: &dyn crate::backend::BsdfRtBackend,
    ) -> Result<(), String> {
        // Backend selection is runtime now (registration-time, app-side); the
        // RtcMethod no longer drives dispatch. Geom-optics with a non-null
        // backend is supported; wave-optics has no backend.
        match self {
            Self::GeomOptics(_) if backend.is_null() => Err(
                "no BSDF backend compiled in; rebuild vgonio-app --features embree".to_string(),
            ),
            Self::GeomOptics(_) => Ok(()),
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
                None,
            ));
        }

        if emitter.max_bounces < 1 {
            return Err(VgonioError::new(
                "Number of bounces must be at least 1",
                None,
            ));
        }

        if !(emitter.zenith.step_size > rad!(0.0) && emitter.azimuth.step_size > rad!(0.0)) {
            return Err(VgonioError::new(
                "Emitter's orbit radius, zenith and azimuth must be positive",
                None,
            ));
        }

        if emitter.spectrum.start < nm!(0.0) || emitter.spectrum.stop < nm!(0.0) {
            return Err(VgonioError::new(
                "Emitter's spectrum must be positive",
                None,
            ));
        }

        if emitter.spectrum.start > emitter.spectrum.stop {
            return Err(VgonioError::new(
                "Emitter's spectrum start must be less than its stop",
                None,
            ));
        }

        for collector in &self.receivers {
            if !collector.precision.is_positive() {
                return Err(VgonioError::new(
                    "Detector's precision must be positive",
                    None,
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
    pub fn check_supported_for_measurement(
        &self,
        backend: &dyn crate::backend::BsdfRtBackend,
    ) -> Result<(), String> {
        self.sim_kind.check_supported_for_measurement(backend)
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
    fn simulation_kind_support_check_uses_runtime_backend() {
        use crate::backend::{
            BsdfRtBackend, DeviceHandle, GeometryHandle, NullBsdfBackend, SceneHandle,
            SingleSimResult,
        };
        use crate::bsdf::emitter::EmitterCircularSector;
        #[cfg(not(feature = "vdbg"))]
        use vgn_core::optics::Ior;
        use vgn_core::math::Sph2;
        use vgn_io::MicroSurfaceMesh;

        // Non-null stub backend; trait methods are unreachable in this test
        // (only `is_null()` — the default `false` — is consulted).
        struct StubBackend;
        impl BsdfRtBackend for StubBackend {
            fn create_resources(
                &self,
                _: &MicroSurfaceMesh,
            ) -> (DeviceHandle, SceneHandle, GeometryHandle) {
                unreachable!()
            }
            #[allow(clippy::too_many_arguments)]
            fn simulate_single_point(
                &self,
                _: Sph2,
                _: &EmitterCircularSector<'_>,
                _: &MicroSurfaceMesh,
                _: &GeometryHandle,
                _: &SceneHandle,
                #[cfg(not(feature = "vdbg"))] _: bool,
                #[cfg(not(feature = "vdbg"))] _: &[Ior],
                #[cfg(not(feature = "vdbg"))] _: &[Ior],
            ) -> SingleSimResult {
                unreachable!()
            }
        }

        let grid = SimulationKind::GeomOptics(RtcMethod::Grid);
        let embree = SimulationKind::GeomOptics(RtcMethod::Embree);
        let wave = SimulationKind::WaveOptics;
        let null = NullBsdfBackend;
        let stub = StubBackend;

        // Null backend: every geom-optics kind is unsupported; wave always is.
        assert!(grid.check_supported_for_measurement(&null).is_err());
        assert!(embree.check_supported_for_measurement(&null).is_err());
        assert!(wave.check_supported_for_measurement(&null).is_err());

        // A real (non-null) backend supports geom-optics; wave is unimplemented.
        assert!(grid.check_supported_for_measurement(&stub).is_ok());
        assert!(embree.check_supported_for_measurement(&stub).is_ok());
        assert!(wave.check_supported_for_measurement(&stub).is_err());
    }
}

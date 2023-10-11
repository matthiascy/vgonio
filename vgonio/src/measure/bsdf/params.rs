use crate::{
    error::RuntimeError,
    measure::bsdf::{
        emitter::EmitterParams,
        receiver::{DataRetrievalMode, PartitionScheme, ReceiverParams},
        rtc::RtcMethod,
        BsdfKind,
    },
    Medium, RangeByStepSizeInclusive, SphericalDomain,
};
use serde::{Deserialize, Serialize};
use vgcore::{
    error::VgonioError,
    units::{deg, nm, rad},
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

impl SimulationKind {
    /// Returns the value as a u8. The verse of `TryFrom<u8>`.
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::GeomOptics(RtcMethod::Grid) => 0x00,
            #[cfg(feature = "embree")]
            Self::GeomOptics(RtcMethod::Embree) => 0x01,
            #[cfg(feature = "optix")]
            Self::GeomOptics(RtcMethod::Optix) => 0x02,
            Self::WaveOptics => 0x03,
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
    pub emitter: EmitterParams,

    /// Description of the detector.
    pub receiver: ReceiverParams,
}

impl Default for BsdfMeasurementParams {
    fn default() -> Self {
        Self {
            kind: BsdfKind::Brdf,
            #[cfg(feature = "embree")]
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Embree),
            #[cfg(not(feature = "embree"))]
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Grid),
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Air,
            emitter: EmitterParams {
                num_rays: 1000,
                max_bounces: 10,
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
                spectrum: RangeByStepSizeInclusive::new(nm!(400.0), nm!(700.0), nm!(100.0)),
            },
            receiver: ReceiverParams {
                domain: SphericalDomain::Upper,
                precision: deg!(2.0).to_radians(),
                scheme: PartitionScheme::Beckers,
                retrieval_mode: DataRetrievalMode::BsdfOnly,
            },
        }
    }
}

impl BsdfMeasurementParams {
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

        let collector = &self.receiver;
        if !collector.precision.is_positive() {
            return Err(VgonioError::new(
                "Detector's precision must be positive",
                Some(Box::new(RuntimeError::InvalidDetector)),
            ));
        }

        Ok(self)
    }

    /// Returns the total number of samples that will be collected.
    pub fn samples_count(&self) -> usize {
        self.emitter.measurement_points_count() * self.receiver.num_patches()
    }

    /// Returns the parameters as a HashMap.
    pub fn to_exr_extra_info(
        &self,
    ) -> std::collections::HashMap<exr::meta::attribute::Text, exr::meta::attribute::AttributeValue>
    {
        use exr::meta::attribute::{AttributeValue, Text};
        let mut hash_map = std::collections::HashMap::new();
        hash_map.insert(
            Text::new_or_panic("kind"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.kind))),
        );
        hash_map.insert(
            Text::new_or_panic("sim_kind"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.sim_kind))),
        );
        hash_map.insert(
            Text::new_or_panic("incident_medium"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.incident_medium))),
        );
        hash_map.insert(
            Text::new_or_panic("transmitted_medium"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.transmitted_medium))),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.num_rays"),
            AttributeValue::I32(self.emitter.num_rays as i32),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.max_bounces"),
            AttributeValue::I32(self.emitter.max_bounces as i32),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.zenith.start"),
            AttributeValue::F32(self.emitter.zenith.start.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.zenith.stop"),
            AttributeValue::F32(self.emitter.zenith.stop.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.zenith.step_size"),
            AttributeValue::F32(self.emitter.zenith.step_size.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.azimuth.start"),
            AttributeValue::F32(self.emitter.azimuth.start.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.azimuth.stop"),
            AttributeValue::F32(self.emitter.azimuth.stop.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.azimuth.step_size"),
            AttributeValue::Text(Text::new_or_panic(format!(
                "{}",
                self.emitter.azimuth.step_size.in_degrees().as_f32()
            ))),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.spectrum.start"),
            AttributeValue::F32(self.emitter.spectrum.start.as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.spectrum.stop"),
            AttributeValue::F32(self.emitter.spectrum.stop.as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("emitter.spectrum.step_size"),
            AttributeValue::F32(self.emitter.spectrum.step_size.as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("detector.domain"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.receiver.domain))),
        );
        hash_map.insert(
            Text::new_or_panic("detector.precision"),
            AttributeValue::F32(self.receiver.precision.in_degrees().as_f32()),
        );
        hash_map.insert(
            Text::new_or_panic("detector.scheme"),
            AttributeValue::Text(Text::new_or_panic(format!("{:?}", self.receiver.scheme))),
        );
        hash_map
    }
}

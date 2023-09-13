use crate::{error::RuntimeError, RangeByStepSizeInclusive};
use serde::{Deserialize, Serialize};
use vgcore::{
    error::VgonioError,
    units::{deg, rad, Radians},
};

/// Default azimuth angle range for the measurement: [0°, 360°] with 5° step
/// size.
const DEFAULT_AZIMUTH_RANGE: RangeByStepSizeInclusive<Radians> =
    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TWO_PI, deg!(5.0).in_radians());

/// Default zenith angle range for the measurement: [0°, 90°] with 2° step size.
pub const DEFAULT_ZENITH_RANGE: RangeByStepSizeInclusive<Radians> =
    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::HALF_PI, deg!(2.0).in_radians());

/// Parameters for microfacet area distribution measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdfMeasurementParams {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSizeInclusive<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSizeInclusive<Radians>,
}

impl Default for AdfMeasurementParams {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }
}

impl AdfMeasurementParams {
    /// Returns the number of samples with the current parameters.
    pub fn samples_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// Validate the parameters.
    pub fn validate(self) -> Result<Self, VgonioError> {
        if !(self.azimuth.start >= Radians::ZERO
            && self.azimuth.stop
                <= (Radians::TWO_PI + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
        {
            return Err(VgonioError::new(
                "Microfacet distribution measurement: azimuth angle must be in the range [0°, \
                 360°]",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }
        if !(self.zenith.start >= Radians::ZERO && self.zenith.start <= Radians::HALF_PI) {
            return Err(VgonioError::new(
                "Microfacet distribution measurement: zenith angle must be in the range [0°, 90°]",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }
        if !(self.azimuth.step_size > rad!(0.0) && self.zenith.step_size > rad!(0.0)) {
            return Err(VgonioError::new(
                "Microfacet distribution measurement: azimuth and zenith step sizes must be \
                 positive!",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }

        Ok(self)
    }
}

/// Parameters for microfacet geometric attenuation (masking/shadowing) function
/// measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MsfMeasurementParams {
    /// Azimuthal angle sampling range.
    pub azimuth: RangeByStepSizeInclusive<Radians>,
    /// Polar angle sampling range.
    pub zenith: RangeByStepSizeInclusive<Radians>,
    /// Discretisation resolution during the measurement (area estimation).
    pub resolution: u32,
    /// Whether to strictly use only the facet normals falling into the sampling
    /// range.
    pub strict: bool,
}

impl Default for MsfMeasurementParams {
    fn default() -> Self {
        Self {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
            resolution: 512,
            strict: false,
        }
    }
}

impl MsfMeasurementParams {
    /// Returns the number of samples with the current parameters.
    pub fn samples_count(&self) -> usize {
        (self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()).pow(2)
    }

    /// Counts the number of samples (on hemisphere) that will be taken during
    /// the measurement.
    pub fn measurement_points_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// Validate the parameters when reading from a file.
    pub fn validate(self) -> Result<Self, VgonioError> {
        if !(self.azimuth.start >= Radians::ZERO
            && self.azimuth.stop
                <= (Radians::TWO_PI + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
        {
            return Err(VgonioError::new(
                "Microfacet shadowing-masking measurement: azimuth angle must be in the range \
                 [0°, 360°]",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }
        if !(self.zenith.start >= Radians::ZERO
            && self.zenith.start
                <= (Radians::HALF_PI + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
        {
            return Err(VgonioError::new(
                "Microfacet shadowing-masking measurement: zenith angle must be in the range [0°, \
                 90°]",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }
        if !(self.azimuth.step_size > rad!(0.0) && self.zenith.step_size > rad!(0.0)) {
            return Err(VgonioError::new(
                "Microfacet shadowing-masking measurement: azimuth and zenith step sizes must be \
                 positive!",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }

        Ok(self)
    }
}

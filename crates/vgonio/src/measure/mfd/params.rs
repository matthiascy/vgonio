use crate::error::RuntimeError;
use base::{
    error::VgonioError,
    math::Sph2,
    partition::PartitionScheme,
    range::RangeByStepSizeInclusive,
    units::{deg, rad, Radians},
};
use serde::{Deserialize, Deserializer, Serialize};

/// Default azimuth angle range for the measurement: [0°, 360°] with 5° step
/// size.
const DEFAULT_AZIMUTH_RANGE: RangeByStepSizeInclusive<Radians> =
    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TWO_PI, deg!(5.0).in_radians());

/// Default zenith angle range for the measurement: [0°, 90°] with 2° step size.
pub const DEFAULT_ZENITH_RANGE: RangeByStepSizeInclusive<Radians> =
    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::HALF_PI, deg!(2.0).in_radians());

/// Parameters for microfacet area distribution measurement.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NdfMeasurementParams {
    /// The way to measure the area distribution function.
    pub mode: NdfMeasurementMode,
    /// Whether to crop the surface to a disk during the measurement.
    pub crop_to_disk: bool,
    /// Whether to use the actual facet area during the measurement.
    /// If `false`, the area distribution function is normalised by
    /// the number of facets.
    pub use_facet_area: bool,
}

/// How to measure the area distribution function.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NdfMeasurementMode {
    /// Measure the area distribution function at the different measurement
    /// points.
    #[serde(rename = "points")]
    ByPoints {
        /// Azimuthal angle sampling range.
        azimuth: RangeByStepSizeInclusive<Radians>,
        /// Polar angle sampling range.
        zenith: RangeByStepSizeInclusive<Radians>,
    },
    /// Measure the area distribution function by partitioning the hemisphere
    /// covering the surface using Beckers' partitioning scheme.
    #[serde(rename = "partition")]
    ByPartition {
        /// Partition precision along the zenith angle.
        precision: Radians,
    },
}

impl Default for NdfMeasurementMode {
    fn default() -> Self {
        NdfMeasurementMode::ByPoints {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }
}

impl NdfMeasurementMode {
    /// Default parameters when the measurement mode is `ByPoints`.
    pub fn default_by_points() -> Self {
        NdfMeasurementMode::ByPoints {
            azimuth: DEFAULT_AZIMUTH_RANGE,
            zenith: DEFAULT_ZENITH_RANGE,
        }
    }

    /// Default parameters when the measurement mode is `ByPartition`.
    pub fn default_by_partition() -> Self {
        NdfMeasurementMode::ByPartition {
            precision: Radians::from_degrees(2.0),
        }
    }

    /// Returns the `PartitionScheme` for data collection with the current
    /// parameters, especially when storing the data in a file.
    pub fn partition_scheme_for_data_collection(&self) -> PartitionScheme {
        match self {
            NdfMeasurementMode::ByPoints { .. } => PartitionScheme::EqualAngle,
            NdfMeasurementMode::ByPartition { .. } => PartitionScheme::Beckers,
        }
    }

    /// Returns the corresponding partition precision.
    pub fn partition_precision_for_data_collection(&self) -> Sph2 {
        match self {
            NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                Sph2::new(zenith.step_size, azimuth.step_size)
            }
            NdfMeasurementMode::ByPartition { precision, .. } => Sph2::new(*precision, rad!(0.0)),
        }
    }

    /// Returns the corresponding measurement points.
    ///
    /// # Returns
    ///
    /// - `Some((azimuth, zenith))` if the measurement mode is `ByPoints`.
    /// - `None` if the measurement mode is `ByPartition`.
    pub fn as_mode_by_points(
        &self,
    ) -> Option<(
        RangeByStepSizeInclusive<Radians>,
        RangeByStepSizeInclusive<Radians>,
    )> {
        match self {
            NdfMeasurementMode::ByPoints { azimuth, zenith } => Some((*azimuth, *zenith)),
            NdfMeasurementMode::ByPartition { .. } => None,
        }
    }
}

impl NdfMeasurementParams {
    /// Returns the number of samples with the current parameters.
    pub fn samples_count(&self) -> usize {
        match self.mode {
            NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                azimuth.step_count_wrapped() * zenith.step_count_wrapped()
            }
            NdfMeasurementMode::ByPartition { precision } => {
                PartitionScheme::Beckers.num_patches(Sph2::new(precision, rad!(0.0)))
            }
        }
    }

    /// Counts the number of samples (on hemisphere) that will be taken during
    /// the measurement.
    pub fn expected_samples_count_by_points(
        azi: &RangeByStepSizeInclusive<Radians>,
        zen: &RangeByStepSizeInclusive<Radians>,
    ) -> usize {
        azi.step_count_wrapped() * zen.step_count_wrapped()
    }

    /// Counts the number of samples (on hemisphere) that will be taken during
    /// the measurement.
    pub fn expected_samples_count_by_partition(precision: Sph2, scheme: PartitionScheme) -> usize {
        scheme.num_patches(precision)
    }

    /// Validate the parameters.
    pub fn validate(self) -> Result<Self, VgonioError> {
        match self.mode {
            NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                if !(azimuth.start >= Radians::ZERO
                    && azimuth.stop
                        <= (Radians::TWO_PI
                            + rad!(f32::EPSILON + std::f32::consts::PI * f32::EPSILON)))
                {
                    return Err(VgonioError::new(
                        "Microfacet distribution measurement: azimuth angle must be in the range \
                         [0°, 360°]",
                        Some(Box::new(RuntimeError::InvalidParameters)),
                    ));
                }
                if !(zenith.start >= Radians::ZERO && zenith.start <= Radians::HALF_PI) {
                    return Err(VgonioError::new(
                        "Microfacet distribution measurement: zenith angle must be in the range \
                         [0°, 90°]",
                        Some(Box::new(RuntimeError::InvalidParameters)),
                    ));
                }
                if !(azimuth.step_size > rad!(0.0) && zenith.step_size > rad!(0.0)) {
                    return Err(VgonioError::new(
                        "Microfacet distribution measurement: azimuth and zenith step sizes must \
                         be positive!",
                        Some(Box::new(RuntimeError::InvalidParameters)),
                    ));
                }
            }
            NdfMeasurementMode::ByPartition { precision } => {
                if precision <= rad!(0.0) {
                    return Err(VgonioError::new(
                        "Microfacet distribution measurement: theta precision must be positive!",
                        Some(Box::new(RuntimeError::InvalidParameters)),
                    ));
                }
            }
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
    pub fn expected_samples_count(
        azi: &RangeByStepSizeInclusive<Radians>,
        zen: &RangeByStepSizeInclusive<Radians>,
    ) -> usize {
        (azi.step_count_wrapped() * zen.step_count_wrapped()).pow(2)
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

/// Parameters for microfacet slope distribution function measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SdfMeasurementParams {
    /// Maximum slope to visualise when embedding the slope on a disk.
    /// Typically, this is used to write the slope distribution function to an
    /// image.
    #[serde(deserialize_with = "deserialize_max_slope")]
    pub max_slope: f32,
}

fn deserialize_max_slope<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let val_str: String = Deserialize::deserialize(deserializer)?;
    let trimmed = val_str.trim();
    match trimmed.parse::<f32>() {
        Ok(max_slope) => Ok(max_slope),
        Err(_) => {
            let (constant, rest) = if trimmed.ends_with("pi") {
                (std::f32::consts::PI, trimmed.trim_end_matches("pi"))
            } else if trimmed.ends_with("tau") {
                (std::f32::consts::TAU, trimmed.trim_end_matches("tau"))
            } else {
                return Err(Error::custom(format!(
                    "invalid value for SdfMeasurementParams: {}",
                    trimmed
                )));
            };
            let rest = rest.trim();
            if rest.contains('/') {
                let mut split = rest.split('/');
                let numerator = split.next().unwrap();
                let denominator = split.next().unwrap();
                if split.next().is_some() {
                    return Err(Error::custom(format!(
                        "invalid value for SdfMeasurementParams: {}",
                        trimmed
                    )));
                }
                let numerator = numerator.parse::<f32>().map_err(Error::custom)?;
                let denominator = denominator.parse::<f32>().map_err(Error::custom)?;
                if denominator == 0.0 {
                    return Err(Error::custom(format!(
                        "invalid value for SdfMeasurementParams: {}",
                        trimmed
                    )));
                }
                Ok(constant * numerator / denominator)
            } else if rest.is_empty() {
                Ok(constant)
            } else {
                let value = rest.parse::<f32>().map_err(Error::custom)?;
                Ok(constant * value)
            }
        }
    }
}

impl SdfMeasurementParams {
    /// Validate the parameters.
    pub fn validate(self) -> Result<Self, VgonioError> {
        if self.max_slope <= 0.0 {
            return Err(VgonioError::new(
                "Microfacet slope distribution measurement: max_slope must be positive!",
                Some(Box::new(RuntimeError::InvalidParameters)),
            ));
        }
        Ok(self)
    }
}

#[test]
fn serialisation_deserialisation() {
    {
        let params = SdfMeasurementParams { max_slope: 1.0 };
        {
            let serialised = serde_yaml::to_string(&params).unwrap();
            assert_eq!(serialised, "max_slope: 1.0\n");

            let deserialised: SdfMeasurementParams = serde_yaml::from_str(&serialised).unwrap();
            assert_eq!(deserialised, params);
        }
    }
    {
        let params_str = "max_slope: 3pi\n";
        let deserialised: SdfMeasurementParams = serde_yaml::from_str(params_str).unwrap();
        assert_eq!(
            deserialised,
            SdfMeasurementParams {
                max_slope: std::f32::consts::PI * 3.0
            }
        );
    }
    {
        let params_str = "max_slope: 3.5/7pi\n";
        let deserialised: SdfMeasurementParams = serde_yaml::from_str(params_str).unwrap();
        assert_eq!(
            deserialised,
            SdfMeasurementParams {
                max_slope: std::f32::consts::PI * 0.5
            }
        );
    }
}

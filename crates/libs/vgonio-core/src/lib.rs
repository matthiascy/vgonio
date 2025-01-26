//! # vgonio-core
//! Core library for vgonio.
//! Contains all the basic types and functions for the vgonio project.

// Enable macro 2.0
#![feature(decl_macro)]
// Enable const trait implementation
#![feature(const_trait_impl)]
#![feature(const_format_args)]
#![feature(adt_const_params)]
#![feature(structural_match)]
// Enable _mm_rcp14_ss
#![feature(stdarch_x86_avx512)]
#![feature(seek_stream_len)]
#![warn(missing_docs)]
// TODO: Enable this feature when it is stable to use const generics in the
// whole project. #![feature(effects)]
#![cfg(feature = "bxdf")]
#![feature(associated_type_defaults)]

use bxdf::{brdf::measured::MeasuredBrdfKind, BrdfProxy};
use error::VgonioError;
use optics::ior::IorRegistry;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    hash::Hash,
    marker::{ConstParamTy, StructuralPartialEq},
    str::FromStr,
};
use units::Nanometres;

pub mod error;
pub mod io;
pub mod math;
pub mod units;
pub mod utils;

pub mod cli;

pub mod optics;

#[cfg(feature = "bxdf")]
pub mod bxdf;

/// Indicates whether something is uniform in all directions or not.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Copy, Clone, ConstParamTy)]
pub enum Symmetry {
    /// Uniformity in all directions.
    #[cfg_attr(feature = "cli", clap(alias = "iso"))]
    Isotropic,
    /// Non-uniformity in some directions.
    #[cfg_attr(feature = "cli", clap(alias = "aniso"))]
    Anisotropic,
}

impl Symmetry {
    /// Returns whether it's isotropic.
    pub const fn is_isotropic(&self) -> bool { matches!(self, Self::Isotropic) }

    /// Returns whether it's anisotropic.
    pub const fn is_anisotropic(&self) -> bool { matches!(self, Self::Anisotropic) }
}

impl Display for Symmetry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Symmetry::Isotropic => "Isotropic",
                Symmetry::Anisotropic => "Anisotropic",
            }
        )
    }
}

impl StructuralPartialEq for Symmetry {}

impl Eq for Symmetry {}

impl PartialEq<Self> for Symmetry {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Symmetry::Isotropic, Symmetry::Isotropic)
                | (Symmetry::Anisotropic, Symmetry::Anisotropic)
        )
    }
}

impl Hash for Symmetry {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Symmetry::Isotropic => 0.hash(state),
            Symmetry::Anisotropic => 1.hash(state),
        }
    }
}

/// Version of anything in vgonio.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Version {
    /// Major version.
    pub major: u8,
    /// Minor version.
    pub minor: u8,
    /// Patch version.
    pub patch: u8,
}

impl Version {
    /// Creates a new version.
    pub const fn new(major: u8, minor: u8, patch: u8) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns the version as a string.
    pub fn as_string(&self) -> String { format!("v{}.{}.{}", self.major, self.minor, self.patch) }

    /// Returns the version as a u32.
    pub const fn as_u32(&self) -> u32 {
        (self.major as u32) << 16 | (self.minor as u32) << 8 | (self.patch as u32)
    }

    /// Converts a version from a u32.
    pub const fn from_u32(v: u32) -> Self {
        Self {
            major: ((v >> 16) & 0xFF) as u8,
            minor: ((v >> 8) & 0xFF) as u8,
            patch: (v & 0xFF) as u8,
        }
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_string())
    }
}

/// Metrics to use for the error/distance computation.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorMetric {
    /// L1 norm: sum of the absolute differences between the observed data and
    /// the predicted model values.
    L1,
    /// L2 norm: square root of the sum of the squared differences between the
    /// observed data and the predicted model values.
    L2,
    /// Mean squared error: the average of the squared differences between the
    /// observed data and the predicted model values.
    Mse,
    /// Root mean squared error: square root of the mean squared error.
    Rmse,
    /// 0.5 times the sum of squared residuals.
    #[default]
    Nllsq,
}

impl Display for ErrorMetric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorMetric::L1 => write!(f, "l1"),
            ErrorMetric::L2 => write!(f, "l2"),
            ErrorMetric::Mse => write!(f, "mse"),
            ErrorMetric::Rmse => write!(f, "rmse"),
            ErrorMetric::Nllsq => write!(f, "nllsq"),
        }
    }
}

/// Weighting function to apply to the observed data and predicted model values.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Weighting {
    /// Nothing applied to the measured data and model values.
    #[cfg_attr(feature = "cli", clap(name = "none"))]
    #[default]
    None,
    /// From paper "BRDF Models for Accurate and Efficient Rendering of Glossy
    /// Surfaces".
    /// ln(1 + cos_theta_i * d) where d is the measured data or model value.
    /// This weighting function is trying to reduce the influence of the
    /// grazing angles.
    #[cfg_attr(feature = "cli", clap(name = "lncos"))]
    LnCos,
}

use utils::medium::Medium;

/// The level of the measured BRDF.
///
/// This is used to indicate the level of the measured BRDF that includes the
/// energy of rays at the given bounce.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum BrdfLevel {
    /// The level of the measured BRDF that includes the energy of rays at
    /// all bounces.
    #[default]
    L0 = 0,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the first bounce.
    L1 = 1,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the second bounce.
    L2 = 2,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the third bounce.
    L3 = 3,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the fourth bounce.
    L4 = 4,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the fifth bounce.
    L5 = 5,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the sixth bounce.
    L6 = 6,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the seventh bounce.
    L7 = 7,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the eighth bounce.
    L8 = 8,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the ninth bounce.
    L9 = 9,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the tenth bounce.
    L10 = 10,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the eleventh bounce.
    L11 = 11,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the twelfth bounce.
    L12 = 12,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the thirteenth bounce.
    L13 = 13,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the fourteenth bounce.
    L14 = 14,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the fifteenth bounce.
    L15 = 15,
    /// The level of the measured BRDF that includes the energy of rays at
    /// the sixteenth bounce.
    L16 = 16,
    /// The level of the measured BRDF that includes the energy of rays at
    /// bounces greater than 1.
    L1Plus = 17,
    /// Non-existent level.
    None = 18,
}

static_assertions::assert_eq_size!(BrdfLevel, u32);
// Make sure that the niche optimisation happens.
static_assertions::assert_eq_size!(Option<BrdfLevel>, u32);

impl BrdfLevel {
    /// Returns the level as a u32.
    pub const fn as_u32(&self) -> u32 { *self as u32 }

    /// Returns whether the level is valid.
    pub const fn is_valid(&self) -> bool { *self as u32 != BrdfLevel::None as u32 }
}

impl From<usize> for BrdfLevel {
    fn from(n: usize) -> Self {
        match n as u32 {
            0 => BrdfLevel::L0,
            1 => BrdfLevel::L1,
            2 => BrdfLevel::L2,
            3 => BrdfLevel::L3,
            4 => BrdfLevel::L4,
            5 => BrdfLevel::L5,
            6 => BrdfLevel::L6,
            7 => BrdfLevel::L7,
            8 => BrdfLevel::L8,
            9 => BrdfLevel::L9,
            10 => BrdfLevel::L10,
            11 => BrdfLevel::L11,
            12 => BrdfLevel::L12,
            13 => BrdfLevel::L13,
            14 => BrdfLevel::L14,
            15 => BrdfLevel::L15,
            16 => BrdfLevel::L16,
            17 => BrdfLevel::L1Plus,
            _ => BrdfLevel::None,
        }
    }
}

impl From<u32> for BrdfLevel {
    fn from(n: u32) -> Self { BrdfLevel::from(n as usize) }
}

impl FromStr for BrdfLevel {
    type Err = VgonioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let val = s.to_lowercase();
        let val = val.trim();
        if val.len() > 3 {
            return Err(VgonioError::new(
                "Invalid BRDF level. The level must be between l0 and l16 or l1+.",
                None,
            ));
        }
        match val {
            "l1+" => Ok(BrdfLevel::L1Plus),
            _ => {
                let level = s.strip_prefix('l').ok_or_else(|| {
                    VgonioError::new(
                        "Invalid BRDF level. The level must be between l0 and l16",
                        None,
                    )
                })?;
                Ok(BrdfLevel::from(level.parse::<u32>().map_err(|err| {
                    VgonioError::new(format!("Invalid BRDF level. {}", err), Some(Box::new(err)))
                })?))
            },
        }
    }
}

impl Display for BrdfLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BrdfLevel::L1Plus => f.write_str("l1+"),
            _ => write!(f, "l{}", self.as_u32()),
        }
    }
}

/// Trait for the different kinds of measurement data.
///
/// Measurement data can be of different kinds, such as
/// - Normal Distribution Function (NDF)
/// - Masking Shadowing Function (MSF)
/// - Slope Distribution Function (SDF)
/// - Bidirectional Scattering Distribution Function (BSDF)
pub trait AnyMeasured: Debug {
    /// Returns the kind of the measurement.
    fn kind(&self) -> MeasurementKind;

    /// Returns true if the measurement contains multiple levels of BRDF data.
    fn has_multiple_levels(&self) -> bool { false }

    /// Casts the measurement data to a trait object for downcasting to the
    /// concrete type.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Casts the measurement data to a mutable trait object for downcasting to
    /// the concrete type.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Returns the BRDF data at the given level if the measurement data
    /// contains multiple levels of BRDF data.
    #[allow(unused)]
    fn as_any_brdf(&self, level: BrdfLevel) -> Option<&dyn AnyMeasuredBrdf> { None }
}

impl dyn AnyMeasured {
    /// Downcasts the measurement data to the concrete type.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: AnyMeasured + 'static,
    {
        self.as_any().downcast_ref()
    }

    /// Downcasts the measurement data to the mutable concrete type.
    pub fn downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: AnyMeasured + 'static,
    {
        self.as_any_mut().downcast_mut()
    }
}

#[macro_export]
/// Boilerplate macro for implementing the `AnyMeasured` trait for a type.
///
/// This macro is used to implement the `AnyMeasured` trait for a type.
///
/// The macro takes two arguments:
/// - `$t`: The type to implement the `AnyMeasured` trait for.
/// - `$kind`: The kind of the measurement.
///
/// NOTE: The macro doesn't cover the case where the type has multiple levels of
/// BRDF data. Please implement the `AnyMeasured` trait manually for such types.
macro_rules! impl_any_measured_trait {
    // Non-BRDF types.
    ($t:ty, $kind:ident) => {
        impl AnyMeasured for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::$kind }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }
    };
    // Single-level BRDF types.
    (@single_level_brdf $t:ty) => {
        impl AnyMeasured for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::Bsdf }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

            fn as_any_brdf(&self, _level: BrdfLevel) -> Option<&dyn AnyMeasuredBrdf> { Some(self) }
        }
    };
}

/// Kind of different measurements.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeasurementKind {
    /// BSDF measurement.
    Bsdf = 0x00,
    /// Microfacet area distribution function measurement.
    Ndf = 0x01,
    /// Microfacet Masking-shadowing function measurement.
    Gaf = 0x02,
    /// Microfacet slope distribution function measurement.
    Sdf = 0x03,
}

impl Display for MeasurementKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementKind::Bsdf => {
                write!(f, "BSDF")
            },
            MeasurementKind::Ndf => {
                write!(f, "NDF")
            },
            MeasurementKind::Gaf => {
                write!(f, "GAF")
            },
            MeasurementKind::Sdf => {
                write!(f, "SDF")
            },
        }
    }
}

impl From<u8> for MeasurementKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Bsdf,
            0x01 => Self::Ndf,
            0x02 => Self::Gaf,
            0x03 => Self::Sdf,
            _ => panic!("Invalid measurement kind! {}", value),
        }
    }
}

impl MeasurementKind {
    /// Returns the measurement kind in the form of a string slice in
    /// lowercase ASCII characters.
    pub fn ascii_str(&self) -> &'static str {
        match self {
            MeasurementKind::Bsdf => "bsdf",
            MeasurementKind::Ndf => "ndf",
            MeasurementKind::Gaf => "gaf",
            MeasurementKind::Sdf => "sdf",
        }
    }
}

/// Common interface for measured BRDFs.
pub trait AnyMeasuredBrdf: Sync + Send {
    /// Returns the kind of the measured BxDF.
    fn kind(&self) -> MeasuredBrdfKind;

    /// Returns the wavelengths at which the BxDF is measured.
    fn spectrum(&self) -> &[Nanometres];

    /// Returns the transmitted medium.
    fn transmitted_medium(&self) -> Medium;

    /// Returns the incident medium.
    fn incident_medium(&self) -> Medium;

    /// Returns a proxy for the measured BRDF.
    fn proxy(&self, iors: &IorRegistry) -> BrdfProxy;

    /// Casts the measured BRDF to any type for later downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Casts the measured BRDF to any mutable type for later downcasting.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Boilerplate macro for implementing the `AnyMeasuredBrdf` trait for a type.
#[macro_export]
macro_rules! any_measured_brdf_trait_common_impl {
    ($t:ty, $kind:ident) => {
        fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::$kind }

        fn spectrum(&self) -> &[Nanometres] { &self.spectrum.as_ref() }

        fn transmitted_medium(&self) -> Medium { self.transmitted_medium }

        fn incident_medium(&self) -> Medium { self.incident_medium }

        fn as_any(&self) -> &dyn std::any::Any { self }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_metric() {
        assert_eq!(ErrorMetric::L1.to_string(), "l1");
        assert_eq!(ErrorMetric::L2.to_string(), "l2");
        assert_eq!(ErrorMetric::Mse.to_string(), "mse");
        assert_eq!(ErrorMetric::Rmse.to_string(), "rmse");
        assert_eq!(ErrorMetric::Nllsq.to_string(), "nllsq");
    }
}

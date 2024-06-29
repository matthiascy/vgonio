//! # vgonio-core
//! Core library for vgonio.
//! Contains all the basic types and functions for the vgonio project.

// Enable macro 2.0
#![feature(decl_macro)]
// Enable const trait implementation
#![feature(const_trait_impl)]
// Enable const fn floating point arithmetic
#![feature(const_fn_floating_point_arithmetic)]
// Enable const mut references
#![feature(const_mut_refs)]
#![feature(const_format_args)]
#![feature(adt_const_params)]
#![feature(structural_match)]
// Enable _mm_rcp14_ss
#![feature(stdarch_x86_avx512)]
#![feature(seek_stream_len)]
#![warn(missing_docs)]
#![feature(effects)]

use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    marker::{ConstParamTy, StructuralPartialEq},
};

mod asset;
pub mod error;
pub mod io;
pub mod math;
pub mod units;

pub use asset::*;

#[cfg(feature = "winit")]
pub mod input;
pub mod medium;
pub mod optics;
pub mod range;

pub mod partition;

/// Indicates whether something is uniform in all directions or not.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Copy, Clone, Hash)]
pub enum Isotropy {
    /// Uniformity in all directions.
    #[cfg_attr(feature = "cli", clap(alias = "iso"))]
    Isotropic,
    /// Non-uniformity in some directions.
    #[cfg_attr(feature = "cli", clap(alias = "aniso"))]
    Anisotropic,
}

impl Display for Isotropy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Isotropy::Isotropic => "Isotropic",
                Isotropy::Anisotropic => "Anisotropic",
            }
        )
    }
}

impl StructuralPartialEq for Isotropy {}

impl Eq for Isotropy {}

impl PartialEq<Self> for Isotropy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Isotropy::Isotropic, Isotropy::Isotropic) => true,
            (Isotropy::Anisotropic, Isotropy::Anisotropic) => true,
            _ => false,
        }
    }
}

impl ConstParamTy for Isotropy {}

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

/// Utility functions.
pub mod utils {
    use chrono::{DateTime, Local};

    /// Returns the current time as an ISO 8601 (RFC 3339) timestamp.
    pub fn iso_timestamp() -> String {
        chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
    }

    /// Returns the current time as an ISO 8601 (RFC 3339) timestamp without the
    /// timezone and the colon in the time field.
    pub fn iso_timestamp_short(datetime: DateTime<Local>) -> String {
        datetime.format("%Y-%m-%dT%H-%M-%S").to_string()
    }

    /// Converts a date time to an ISO 8601 (RFC 3339) timestamp.
    pub fn iso_timestamp_from_datetime(dt: &chrono::DateTime<chrono::Local>) -> String {
        dt.to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
    }

    /// Converts a date time to an ISO 8601 (RFC 3339) timestamp without the
    /// timezone and with the colon in the time field.
    pub fn iso_timestamp_display(dt: &chrono::DateTime<chrono::Local>) -> String {
        dt.format("%Y-%m-%d %H:%M:%S").to_string()
    }
}

/// Metrics to use for the error/distance computation.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorMetric {
    /// Mean squared error.
    Mse,
    /// Most commonly used error metrics in non-linear least squares fitting.
    /// Which is the half of the sum of the squares of the differences between
    /// the measured data and the model.
    #[default]
    Nlls,
}

/// Trait for the different kinds of measurement data.
///
/// Measurement data can be of different kinds, such as
/// - Normal Distribution Function (NDF)
/// - Masking Shadowing Function (MSF)
/// - Slope Distribution Function (SDF)
/// - Bidirectional Scattering Distribution Function (BSDF)
pub trait MeasuredData: Debug {
    /// Returns the kind of the measurement.
    fn kind(&self) -> MeasurementKind;
    /// Returns whether the measurement data is a Clausen representation.
    fn is_clausen(&self) -> bool { false }
    /// Casts the measurement data to a trait object for downcasting to the
    /// concrete type.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Casts the measurement data to a mutable trait object for downcasting to
    /// the concrete type.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl dyn MeasuredData {
    /// Downcasts the measurement data to the concrete type.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: MeasuredData + 'static,
    {
        self.as_any().downcast_ref()
    }

    /// Downcasts the measurement data to the mutable concrete type.
    pub fn downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: MeasuredData + 'static,
    {
        self.as_any_mut().downcast_mut()
    }
}

#[macro_export]
/// Boilerplate macro for implementing the `MeasuredData` trait for a type.
macro_rules! impl_measured_data_trait {
    ($t:ty, $kind:ident, $is_clausen:expr) => {
        impl MeasuredData for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::$kind }

            fn is_clausen(&self) -> bool { $is_clausen }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
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
    Msf = 0x02,
    /// Microfacet slope distribution function measurement.
    Sdf = 0x03,
}

impl Display for MeasurementKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementKind::Bsdf => {
                write!(f, "BSDF")
            }
            MeasurementKind::Ndf => {
                write!(f, "ADF")
            }
            MeasurementKind::Msf => {
                write!(f, "MSF")
            }
            MeasurementKind::Sdf => {
                write!(f, "SDF")
            }
        }
    }
}

impl From<u8> for MeasurementKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Bsdf,
            0x01 => Self::Ndf,
            0x02 => Self::Msf,
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
            MeasurementKind::Ndf => "adf",
            MeasurementKind::Msf => "msf",
            MeasurementKind::Sdf => "sdf",
        }
    }
}

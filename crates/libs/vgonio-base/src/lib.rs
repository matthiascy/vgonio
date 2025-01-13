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

use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    hash::Hash,
    marker::{ConstParamTy, StructuralPartialEq},
};
use units::Nanometres;

mod asset;
pub mod error;
pub mod handle;
pub mod io;
pub mod math;
pub mod units;

pub use asset::*;
use crate::medium::Medium;

#[cfg(feature = "winit")]
pub mod input;
pub mod medium;
pub mod optics;
pub mod range;

pub mod partition;

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

/// The kind of the measured BRDF.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasuredBrdfKind {
    #[cfg_attr(feature = "cli", clap(name = "clausen"))]
    /// The measured BRDF by Clausen.
    Clausen,
    #[cfg_attr(feature = "cli", clap(name = "merl"))]
    /// The MERL BRDF dataset.
    Merl,
    #[cfg_attr(feature = "cli", clap(name = "utia"))]
    /// The measured BRDF by UTIA at Czech Technical University.
    Utia,
    #[cfg_attr(feature = "cli", clap(name = "rgl"))]
    /// The measured BRDF by Dupuy and Jakob in RGL at EPFL.
    Rgl,
    #[cfg_attr(feature = "cli", clap(name = "vgonio"))]
    /// The simulated BRDF by vgonio.
    Vgonio,
    #[cfg_attr(feature = "cli", clap(name = "yan2018"))]
    /// The BRDF model by Yan et al. 2018.
    Yan2018,
    #[cfg_attr(feature = "cli", clap(name = "unknown"))]
    /// Unknown.
    Unknown,
}

/// Common interface for measured BxDFs.
pub trait MeasuredBrdfData {
    /// Returns the kind of the measured BxDF.
    fn kind(&self) -> MeasuredBrdfKind;

    /// Returns the wavelengths at which the BxDF is measured.
    fn spectrum(&self) -> &[Nanometres];

    /// Returns the transmitted medium.
    fn transmitted_medium(&self) -> Medium;

    /// Returns the incident medium.
    fn incident_medium(&self) -> Medium;
}

/// Boilerplate macro for implementing the `MeasuredBrdfData` trait for a type.
#[macro_export]
macro_rules! impl_measured_brdf_data_trait {
    ($t:ty, $kind:ident) => {
        impl MeasuredBrdfData for $t {
            fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::$kind }

            fn spectrum(&self) -> &[Nanometres] { &self.spectrum.as_ref() }

            fn transmitted_medium(&self) -> Medium { self.transmitted_medium }

            fn incident_medium(&self) -> Medium { self.incident_medium }
        }
    };
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
    // TODO: considering removing this method.
    /// Returns the kind of the BRDF if it's a BRDF measurement.
    fn brdf_kind(&self) -> Option<MeasuredBrdfKind> { None }
    /// Casts the measurement data to a trait object for downcasting to the
    /// concrete type.
    fn as_any(&self) -> &dyn std::any::Any;
    /// Casts the measurement data to a mutable trait object for downcasting to
    /// the concrete type.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    /// Casts the measurement data to a MeasuredBrdf trait object.
    fn as_brdf_data(&self) -> Option<&dyn MeasuredBrdfData> { None }
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
    ($t:ty, $kind:ident, $brdf_kind:expr) => {
        impl MeasuredData for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::$kind }

            fn brdf_kind(&self) -> Option<MeasuredBrdfKind> {
                if $brdf_kind.is_some() {
                    $brdf_kind
                } else {
                    None
                }
            }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }
    };
    (@brdf $t:ty, $kind:ident, $brdf_kind:expr) => {
        impl MeasuredData for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::$kind }

            fn brdf_kind(&self) -> Option<MeasuredBrdfKind> {
                if $brdf_kind.is_some() {
                    $brdf_kind
                } else {
                    None
                }
            }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

            fn as_brdf_data(&self) -> Option<&dyn MeasuredBrdfData> { Some(self) }
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

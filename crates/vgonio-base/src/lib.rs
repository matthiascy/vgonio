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

use std::{
    fmt::{Display, Formatter},
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
    /// Returns the current time as an ISO 8601 (RFC 3339) timestamp.
    pub fn iso_timestamp() -> String {
        chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
    }

    /// Returns the current time as an ISO 8601 (RFC 3339) timestamp without the
    /// timezone and the colon in the time field.
    pub fn iso_timestamp_short() -> String {
        chrono::Local::now().format("%Y-%m-%dT%H-%M-%S").to_string()
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

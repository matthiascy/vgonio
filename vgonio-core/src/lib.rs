//! # vgonio-core
//! Core library for vgonio.
//! Contains all the basic types and functions for the vgonio project.

// Enable _mm_rcp14_ss
#![feature(stdsimd)]
// Enable macro 2.0
#![feature(decl_macro)]
// Enable const trait implementation
#![feature(const_trait_impl)]
// Enable const fn floating point arithmetic
#![feature(const_fn_floating_point_arithmetic)]
// Enable const mut references
#![feature(const_mut_refs)]
#![warn(missing_docs)]

pub mod error;
pub mod io;
pub mod math;
pub mod units;

/// Indicates whether something is uniform in all directions or not.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Isotropy {
    /// Uniformity in all directions.
    Isotropic,
    /// Non-uniformity in some directions.
    Anisotropic,
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
}

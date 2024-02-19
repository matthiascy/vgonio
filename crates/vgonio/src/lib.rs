//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(vec_push_within_capacity)]
#![feature(assert_matches)]
#![feature(stmt_expr_attributes)]
#![feature(adt_const_params)]
#![feature(seek_stream_len)]
#![feature(trait_upcasting)]
#![feature(new_uninit)]
#![warn(missing_docs)]

extern crate core;
mod app;
mod error;
pub mod fitting;
mod io;
pub mod measure;
mod partition;
mod range;

use base::{error::VgonioError, units::Radians};
pub use range::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    str::FromStr,
};

/// Main entry point for the application.
pub fn run() -> Result<(), VgonioError> {
    use app::args::CliArgs;
    use clap::Parser;

    let args = CliArgs::parse();
    let config = app::init(&args, std::time::SystemTime::now())?;

    match args.command {
        None => app::gui::run(config),
        Some(cmd) => app::cli::run(cmd, config),
    }
}

/// The domain of the spherical coordinate.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SphericalDomain {
    /// Simulation happens only on upper part of the sphere.
    #[default]
    #[serde(rename = "upper_hemisphere")]
    Upper = 0x01,

    /// Simulation happens only on lower part of the sphere.
    #[serde(rename = "lower_hemisphere")]
    Lower = 0x02,

    /// Simulation happens on the whole sphere.
    #[serde(rename = "whole_sphere")]
    Whole = 0x00,
}

impl Display for SphericalDomain {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upper => write!(f, "upper hemisphere"),
            Self::Lower => write!(f, "lower hemisphere"),
            Self::Whole => write!(f, "whole sphere"),
        }
    }
}

impl SphericalDomain {
    /// Range of zenith angle in radians of the upper hemisphere.
    pub const ZENITH_RANGE_UPPER_DOMAIN: (Radians, Radians) = (Radians::ZERO, Radians::HALF_PI);
    /// Range of zenith angle in radians of the lower hemisphere.
    pub const ZENITH_RANGE_LOWER_DOMAIN: (Radians, Radians) = (Radians::HALF_PI, Radians::PI);
    /// Range of zenith angle in radians of the whole sphere.
    pub const ZENITH_RANGE_WHOLE_DOMAIN: (Radians, Radians) = (Radians::ZERO, Radians::TWO_PI);

    /// Returns the zenith range of the domain.
    pub const fn zenith_range(&self) -> (Radians, Radians) {
        match self {
            Self::Upper => Self::ZENITH_RANGE_UPPER_DOMAIN,
            Self::Lower => Self::ZENITH_RANGE_LOWER_DOMAIN,
            Self::Whole => Self::ZENITH_RANGE_WHOLE_DOMAIN,
        }
    }

    /// Returns the azimuth range of the domain.
    pub const fn azimuth_range(&self) -> (Radians, Radians) { (Radians::ZERO, Radians::TWO_PI) }

    /// Returns the zenith angle difference between the maximum and minimum
    /// zenith angle.
    pub const fn zenith_angle_diff(&self) -> Radians {
        match self {
            SphericalDomain::Upper | SphericalDomain::Lower => Radians::HALF_PI,
            SphericalDomain::Whole => Radians::PI,
        }
    }

    /// Clamps the given azimuthal and zenith angle to shape's boundaries.
    ///
    /// # Arguments
    ///
    /// `zenith` - zenith angle in radians.
    /// `azimuth` - azimuthal angle in radians.
    ///
    /// # Returns
    ///
    /// `(zenith, azimuth)` - clamped zenith and azimuth angles in radians.
    #[inline]
    pub fn clamp(&self, zenith: Radians, azimuth: Radians) -> (Radians, Radians) {
        (self.clamp_zenith(zenith), self.clamp_azimuth(azimuth))
    }

    /// Clamps the given zenith angle to shape's boundaries.
    #[inline]
    pub fn clamp_zenith(&self, zenith: Radians) -> Radians {
        let (zenith_min, zenith_max) = self.zenith_range();
        zenith.clamp(zenith_min, zenith_max)
    }

    /// Clamps the given azimuthal angle to shape's boundaries.
    #[inline]
    pub fn clamp_azimuth(&self, azimuth: Radians) -> Radians {
        azimuth.clamp(Radians::ZERO, Radians::TWO_PI)
    }

    /// Returns true if the domain is the upper hemisphere.
    pub fn is_upper_hemisphere(&self) -> bool { matches!(self, SphericalDomain::Upper) }

    /// Returns true if the domain is the lower hemisphere.
    pub fn is_lower_hemisphere(&self) -> bool { matches!(self, SphericalDomain::Lower) }

    /// Returns true if the domain is the full sphere.
    pub fn is_full_sphere(&self) -> bool { matches!(self, SphericalDomain::Whole) }
}

/// Machine epsilon for `f32`.
pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

/// Returns the gamma factor for a floating point number.
pub const fn gamma_f32(n: f32) -> f32 { (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON) }

#[cfg(test)]
mod tests {
    use crate::SphericalDomain;
    use base::units::deg;

    #[test]
    fn spherical_domain_clamp() {
        let domain = SphericalDomain::Upper;
        let angle = deg!(91.0);
        let clamped = domain.clamp_zenith(angle.into());
        assert_eq!(clamped, deg!(90.0));

        let domain = SphericalDomain::Lower;
        let angle = deg!(191.0);
        let clamped = domain.clamp_zenith(angle.into());
        assert_eq!(clamped, deg!(180.0));
    }

    /// Bumps a floating-point value up to the next representable value.
    #[inline]
    pub fn next_f32_up(f: f32) -> f32 {
        if f.is_infinite() && f > 0.0 {
            f
        } else if f == -0.0 {
            0.0
        } else {
            let bits = f.to_bits();
            if f >= 0.0 {
                f32::from_bits(bits + 1)
            } else {
                f32::from_bits(bits - 1)
            }
        }
    }

    /// Bumps a floating-point value down to the next representable value.
    #[inline]
    pub fn next_f32_down(f: f32) -> f32 {
        if f.is_infinite() && f < 0.0 {
            f
        } else if f == -0.0 {
            0.0
        } else {
            let bits = f.to_bits();
            if f > 0.0 {
                f32::from_bits(bits - 1)
            } else {
                f32::from_bits(bits + 1)
            }
        }
    }
}

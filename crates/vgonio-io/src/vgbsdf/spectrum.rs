//! Schema for `spectrum.toml`.
//!
//! Two modes under v1:
//!   - "nm": wavelengths.len() = EXR channel count, one channel per wavelength. Used by BSDF
//!     (always spectral).
//!   - "scalar": wavelengths is empty; the EXR has exactly one channel named "value". Used by
//!     non-spectral outputs (NDF, MSF, SDF, heightfield).
//!
//! The two-mode design fixes the inconsistency where SpectrumToml::nm(vec![])
//! lied about an archive that actually had one channel.

use serde::{Deserialize, Serialize};

/// TOML schema describing the wavelength axis of an archive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpectrumToml {
    /// "nm" for spectral data, "scalar" for non-spectral.
    pub units: String,
    /// For "nm": one entry per channel. For "scalar": must be empty.
    pub wavelengths: Vec<f32>,
}

impl SpectrumToml {
    /// Construct the spectral form with the given per-channel wavelengths (nm).
    ///
    /// # Panics
    ///
    /// Panics if `wavelengths` is empty; use [`SpectrumToml::scalar`] for
    /// non-spectral outputs instead.
    pub fn nm(wavelengths: Vec<f32>) -> Self {
        assert!(
            !wavelengths.is_empty(),
            "SpectrumToml::nm requires at least one wavelength; use scalar() for non-spectral \
             outputs"
        );
        Self {
            units: "nm".into(),
            wavelengths,
        }
    }

    /// Construct the scalar/non-spectral form. The associated EXR layer has
    /// exactly one channel, conventionally named "value".
    pub fn scalar() -> Self {
        Self {
            units: "scalar".into(),
            wavelengths: Vec::new(),
        }
    }

    /// True when the archive carries spectral data (one channel per wavelength).
    pub fn is_spectral(&self) -> bool { self.units == "nm" && !self.wavelengths.is_empty() }

    /// True when the archive carries a single scalar channel.
    pub fn is_scalar(&self) -> bool { self.units == "scalar" && self.wavelengths.is_empty() }

    /// Expected EXR channel count for this spectrum. 1 for scalar; N for spectral.
    pub fn expected_channel_count(&self) -> usize {
        if self.is_scalar() {
            1
        } else {
            self.wavelengths.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_spectral() {
        let s = SpectrumToml::nm(vec![400.0, 500.0, 600.0, 700.0]);
        let str = toml::to_string_pretty(&s).unwrap();
        let r: SpectrumToml = toml::from_str(&str).unwrap();
        assert_eq!(s, r);
        assert!(s.is_spectral());
        assert!(!s.is_scalar());
        assert_eq!(s.expected_channel_count(), 4);
    }

    #[test]
    fn round_trip_scalar() {
        let s = SpectrumToml::scalar();
        let str = toml::to_string_pretty(&s).unwrap();
        let r: SpectrumToml = toml::from_str(&str).unwrap();
        assert_eq!(s, r);
        assert!(!s.is_spectral());
        assert!(s.is_scalar());
        assert_eq!(s.expected_channel_count(), 1);
        assert!(str.contains(r#"units = "scalar""#));
        assert!(str.contains("wavelengths = []"));
    }

    #[test]
    #[should_panic(expected = "at least one wavelength")]
    fn nm_with_empty_wavelengths_panics() { let _ = SpectrumToml::nm(vec![]); }
}

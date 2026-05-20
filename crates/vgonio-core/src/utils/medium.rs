//! Medium of the surface.

mod error;
pub use error::MediumLoadError;

mod id;
pub use id::MediumId;

pub(crate) mod intern;

mod dto;

mod registry;
pub use registry::{MediumEntry, MediumRegistry, MediumSource};

use crate::error::VgonioError;
use std::str::FromStr;

// TODO: better to read the medium from a file, instead of hardcoding it here.

/// Medium of the surface.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Medium {
    /// Vacuum.
    #[serde(rename = "vac")]
    Vacuum = 0x00,
    /// Air.
    #[serde(rename = "air")]
    Air = 0x01,
    /// Aluminium.
    #[serde(rename = "al")]
    Aluminium = 0x02,
    /// Copper.
    #[serde(rename = "cu")]
    Copper = 0x03,
    /// Nickel.
    #[serde(rename = "ni")]
    Nickel = 0x04,
    /// Polyvinyl chloride.
    #[serde(rename = "pvc")]
    Pvc = 0x05,
    /// Chromium.
    #[serde(rename = "cr")]
    Chromium = 0x06,
    /// Unknown.
    #[serde(rename = "unknown")]
    Unknown,
}

impl From<u8> for Medium {
    fn from(value: u8) -> Self {
        match value {
            0x00 => Self::Vacuum,
            0x01 => Self::Air,
            0x02 => Self::Aluminium,
            0x03 => Self::Copper,
            0x04 => Self::Nickel,
            0x05 => Self::Pvc,
            0x06 => Self::Chromium,
            _ => panic!("Invalid medium kind: {}", value),
        }
    }
}

impl From<[u8; 3]> for Medium {
    fn from(value: [u8; 3]) -> Self {
        match &value[0..3] {
            [b'v', b'a', b'c'] => Self::Vacuum,
            [b'a', b'i', b'r'] => Self::Air,
            [b'a', b'l', 0] => Self::Aluminium,
            [b'c', b'u', 0] => Self::Copper,
            [b'n', b'i', 0] => Self::Nickel,
            [b'p', b'v', b'c'] => Self::Pvc,
            [b'c', b'r', b'0'] => Self::Chromium,
            _ => panic!("Invalid medium kind: {:?}", value),
        }
    }
}

/// Describes the kind of material.
pub enum MaterialKind {
    /// Material is a conductor.
    Conductor,
    /// Material is a dielectric.
    Insulator,
}

impl Medium {
    /// Returns the catalog of the medium.
    pub fn kind(&self) -> MaterialKind {
        match self {
            Self::Air | Self::Vacuum => MaterialKind::Insulator,
            Self::Aluminium | Self::Copper | Self::Nickel | Self::Chromium => {
                MaterialKind::Conductor
            },
            Self::Pvc => MaterialKind::Insulator,
            Self::Unknown => panic!("Unknown medium"),
        }
    }

    /// The canonical lowercase name (inverse of [`Medium::from_str`]).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Vacuum => "vacuum",
            Self::Air => "air",
            Self::Aluminium => "al",
            Self::Copper => "cu",
            Self::Nickel => "ni",
            Self::Pvc => "pvc",
            Self::Chromium => "cr",
            Self::Unknown => "unknown",
        }
    }

    /// Serializes the medium to a buffer.
    pub fn write_to_buf(&self, buf: &mut [u8]) {
        debug_assert!(buf.len() >= 3, "Medium needs at least 3 bytes of space");
        match self {
            Self::Vacuum => buf[0..3].copy_from_slice(b"vac"),
            Self::Air => buf[0..3].copy_from_slice(b"air"),
            Self::Aluminium => buf[0..3].copy_from_slice(b"al\0"),
            Self::Copper => buf[0..3].copy_from_slice(b"cu\0"),
            Self::Nickel => buf[0..3].copy_from_slice(b"ni\0"),
            Self::Pvc => buf[0..3].copy_from_slice(b"pvc"),
            Self::Chromium => buf[0..3].copy_from_slice(b"cr\0"),
            Self::Unknown => panic!("Unknown medium"),
        }
    }

    /// Deserializes the medium from a buffer.
    pub fn read_from_buf(buf: &[u8]) -> Self {
        debug_assert!(buf.len() >= 3, "Medium needs at least 3 bytes of space");
        match &buf[0..3] {
            b"vac" => Self::Vacuum,
            b"air" => Self::Air,
            b"al\0" => Self::Aluminium,
            b"cu\0" => Self::Copper,
            b"ni\0" => Self::Nickel,
            b"pvc" => Self::Pvc,
            b"cr\0" => Self::Chromium,
            _ => panic!("Invalid medium kind {:?}", &buf[0..3]),
        }
    }
}

impl FromStr for Medium {
    type Err = VgonioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "air" => Ok(Self::Air),
            "vacuum" => Ok(Self::Vacuum),
            "al" => Ok(Self::Aluminium),
            "cu" => Ok(Self::Copper),
            "ni" => Ok(Self::Nickel),
            "pvc" => Ok(Self::Pvc),
            "cr" => Ok(Self::Chromium),
            &_ => Err(VgonioError::new("Unknown medium", None)),
        }
    }
}

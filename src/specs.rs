//! This module defines the specification of the VGMO file formats.

use crate::measure::measurement::MeasurementKind;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt,
    io::{BufWriter, Write},
};

/// Data encoding while storing the data to the disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum, Serialize, Deserialize)]
pub enum DataEncoding {
    /// The data is encoded as ascii text (plain text).
    Ascii = 0x23, // '#'
    /// The data is encoded as binary data.
    Binary = 0x21, // '!'
}

impl fmt::Display for DataEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataEncoding::Ascii => write!(f, "ascii"),
            DataEncoding::Binary => write!(f, "binary"),
        }
    }
}

impl DataEncoding {
    /// Returns true if the data is encoded as ascii text.
    pub fn is_ascii(&self) -> bool {
        match self {
            DataEncoding::Ascii => true,
            DataEncoding::Binary => false,
        }
    }

    /// Returns true if the data is encoded as binary data.
    pub fn is_binary(&self) -> bool {
        match self {
            DataEncoding::Ascii => false,
            DataEncoding::Binary => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum, Serialize, Deserialize)]
pub enum DataCompression {
    /// No compression.
    None = 0x00,
    /// Zlib compression.
    Zlib = 0xFF,
}

impl fmt::Display for DataCompression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataCompression::None => write!(f, "none"),
            DataCompression::Zlib => write!(f, "zlib"),
        }
    }
}

impl DataCompression {
    /// Returns true if the data is not compressed.
    pub fn is_none(&self) -> bool {
        match self {
            DataCompression::None => false,
            DataCompression::Zlib => true,
        }
    }

    /// Returns true if the data is compressed with zlib.
    pub fn is_zlib(&self) -> bool {
        match self {
            DataCompression::None => false,
            DataCompression::Zlib => true,
        }
    }
}

/// The range of the angle in the measurement, in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleRange {
    /// The start angle of the measurement, in radians.
    pub start: f32,
    /// The end angle of the measurement, in radians.
    pub end: f32,
    /// The number of angle bins of the measurement.
    pub bin_count: u32,
    /// The width of each angle bin, in radians.
    pub bin_width: f32,
}

/// Measurement output format.
pub struct Vgmo<'a> {
    pub kind: MeasurementKind,
    pub encoding: DataEncoding,
    pub compression: DataCompression,
    pub azimuth_range: AngleRange,
    pub zenith_range: AngleRange,
    pub samples: Cow<'a, [f32]>,
}

impl<'a> Vgmo<'a> {
    pub const MAGIC: &'static [u8] = b"VGMO";

    pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> std::io::Result<()> {
        let mut header = [0x20; 48];
        header[0..4].copy_from_slice(Self::MAGIC);
        header[4] = self.kind as u8;
        header[5] = self.encoding as u8;
        header[6] = self.compression as u8;
        header[8..12].copy_from_slice(&self.azimuth_range.start.to_le_bytes());
        header[12..16].copy_from_slice(&self.azimuth_range.end.to_le_bytes());
        header[16..20].copy_from_slice(&self.azimuth_range.bin_width.to_le_bytes());
        header[20..24].copy_from_slice(&self.azimuth_range.bin_count.to_le_bytes());
        header[24..28].copy_from_slice(&self.zenith_range.start.to_le_bytes());
        header[28..32].copy_from_slice(&self.zenith_range.end.to_le_bytes());
        header[32..36].copy_from_slice(&self.zenith_range.bin_width.to_le_bytes());
        header[36..40].copy_from_slice(&self.zenith_range.bin_count.to_le_bytes());
        header[40..44].copy_from_slice(&(self.samples.len() as u32).to_le_bytes());
        header[47] = 0x0A; // LF
        writer.write_all(&header)?;

        match self.encoding {
            DataEncoding::Ascii => match self.compression {
                DataCompression::None => {
                    for (i, s) in self.samples.iter().enumerate() {
                        let val = if i as u32 % self.zenith_range.bin_count
                            == self.zenith_range.bin_count - 1
                        {
                            format!("{s}\n")
                        } else {
                            format!("{s} ")
                        };
                        writer.write_all(val.as_bytes())?;
                    }
                }
                DataCompression::Zlib => {
                    let mut encoder =
                        flate2::write::ZlibEncoder::new(writer, flate2::Compression::default());
                    for (i, s) in self.samples.iter().enumerate() {
                        let val = if i as u32 % self.zenith_range.bin_count
                            == self.zenith_range.bin_count - 1
                        {
                            format!("{s}\n")
                        } else {
                            format!("{s} ")
                        };
                        encoder.write_all(val.as_bytes())?;
                    }
                    encoder.finish()?;
                }
            },
            DataEncoding::Binary => match self.compression {
                DataCompression::None => {
                    for s in self.samples.iter() {
                        writer.write_all(&s.to_le_bytes())?;
                    }
                }
                DataCompression::Zlib => {
                    let mut encoder =
                        flate2::write::ZlibEncoder::new(writer, flate2::Compression::default());
                    for s in self.samples.iter() {
                        encoder.write_all(&s.to_le_bytes())?;
                    }
                    encoder.finish()?;
                }
            },
        }

        Ok(())
    }
}

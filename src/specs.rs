//! This module defines the specification of the VGMO file formats.

use crate::{
    measure::measurement::MeasurementKind,
    units::{LengthUnit, LengthUnitEnum},
};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt,
    io::{BufWriter, Write},
};

/// Data encoding while storing the data to the disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum, Serialize, Deserialize)]
#[repr(u8)]
pub enum DataEncoding {
    /// The data is encoded as ascii text (plain text).
    Ascii = 0x23, // '#'
    /// The data is encoded as binary data.
    Binary = 0x21, // '!'
}

impl From<u8> for DataEncoding {
    fn from(value: u8) -> Self {
        match value {
            0x23 => DataEncoding::Ascii,
            0x21 => DataEncoding::Binary,
            _ => panic!("Invalid data encoding: {}", value),
        }
    }
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
#[repr(u8)]
pub enum DataCompression {
    /// No compression.
    None = 0x00,
    /// Zlib compression.
    Zlib = 0xFF,
}

impl From<u8> for DataCompression {
    fn from(value: u8) -> Self {
        match value {
            0x00 => DataCompression::None,
            0xFF => DataCompression::Zlib,
            _ => panic!("Invalid data compression: {}", value),
        }
    }
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
        write_data_samples(
            writer,
            self.encoding,
            self.compression,
            &self.samples,
            self.zenith_range.bin_count,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct VgmsHeader {
    pub rows: u32,
    pub cols: u32,
    pub du: f32,
    pub dv: f32,
    pub unit: LengthUnitEnum,
    pub sample_data_size: u8,
    pub encoding: DataEncoding,
    pub compression: DataCompression,
}

impl VgmsHeader {
    pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> std::io::Result<()> {
        let mut header = [0x20u8; 32];
        header[0..4].copy_from_slice(Vgms::MAGIC);
        header[4..8].copy_from_slice(&self.rows.to_le_bytes());
        header[8..12].copy_from_slice(&self.cols.to_le_bytes());
        header[12..16].copy_from_slice(&self.du.to_le_bytes());
        header[16..20].copy_from_slice(&self.dv.to_le_bytes());
        header[20] = self.unit as u8;
        header[21] = self.sample_data_size;
        header[22] = self.encoding as u8;
        header[23] = self.compression as u8;
        header[31] = 0x0A; // LF
        writer.write_all(&header)
    }

    /// Reads the header from a byte slice without checking the magic number.
    pub fn from_bytes(bytes: [u8; 28]) -> Option<Self> {
        let rows = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let cols = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let du = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let dv = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let unit = LengthUnitEnum::from(bytes[16]);
        // TODO: check if sample_data_size is valid
        let sample_data_size = bytes[17];
        let encoding = DataEncoding::from(bytes[18]);
        let compression = DataCompression::from(bytes[19]);
        Some(Self {
            rows,
            cols,
            du,
            dv,
            unit,
            sample_data_size,
            encoding,
            compression,
        })
    }
}

pub struct Vgms<'a> {
    pub header: VgmsHeader,
    pub body: Cow<'a, [f32]>,
}

impl<'a> Vgms<'a> {
    pub const MAGIC: &'static [u8] = b"VGMS";

    pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> std::io::Result<()> {
        self.header.write(writer)?;
        write_data_samples(
            writer,
            self.header.encoding,
            self.header.compression,
            &self.body,
            self.header.cols,
        )
    }
}

fn write_data_samples(
    writer: &mut impl Write,
    encoding: DataEncoding,
    compression: DataCompression,
    samples: &[f32],
    cols: u32,
) -> std::io::Result<()> {
    match encoding {
        DataEncoding::Ascii => match compression {
            DataCompression::None => {
                for (i, s) in samples.iter().enumerate() {
                    let val = if i as u32 % cols == cols - 1 {
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
                for (i, s) in samples.iter().enumerate() {
                    let val = if i as u32 % cols == cols - 1 {
                        format!("{s}\n")
                    } else {
                        format!("{s} ")
                    };
                    encoder.write_all(val.as_bytes())?;
                }
                encoder.finish()?;
            }
        },
        DataEncoding::Binary => match compression {
            DataCompression::None => {
                for s in samples.iter() {
                    writer.write_all(&s.to_le_bytes())?;
                }
            }
            DataCompression::Zlib => {
                let mut encoder =
                    flate2::write::ZlibEncoder::new(writer, flate2::Compression::default());
                for s in samples.iter() {
                    encoder.write_all(&s.to_le_bytes())?;
                }
                encoder.finish()?;
            }
        },
    }

    Ok(())
}

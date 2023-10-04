//! IO related types and functions.

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::Display,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

/// Error type when reading a file.
#[derive(Debug)]
pub struct ReadFileError {
    /// Path to the file that caused the error.
    pub path: Box<Path>,
    /// Kind of error that occurred.
    pub kind: ReadFileErrorKind,
}

impl Display for ReadFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error while reading file: {}", self.path.display())
    }
}

impl std::error::Error for ReadFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ReadFileErrorKind::Read(err) => Some(err),
            ReadFileErrorKind::Parse(err) => Some(err),
            ReadFileErrorKind::InvalidFileFormat => None,
        }
    }
}

impl ReadFileError {
    /// Creates a new `ReadFileError` from a `ParseError`.
    pub fn from_parse_error(path: impl AsRef<Path>, err: ParseError) -> Self {
        Self {
            path: path.as_ref().to_path_buf().into_boxed_path(),
            kind: ReadFileErrorKind::Parse(err),
        }
    }

    /// Creates a new `ReadFileError` from a `std::io::Error`.
    pub fn from_std_io_error(path: impl AsRef<Path>, err: std::io::Error) -> Self {
        Self {
            path: path.as_ref().to_path_buf().into_boxed_path(),
            kind: ReadFileErrorKind::Read(err),
        }
    }
}

/// Kind of error that occurred while reading a file.
#[derive(Debug)]
pub enum ReadFileErrorKind {
    /// Error caused by a `std::io::Error`.
    Read(std::io::Error),
    /// Error caused by a `ParseError`.
    Parse(ParseError),
    /// Error caused by an invalid file format.
    InvalidFileFormat,
}

impl From<std::io::Error> for ReadFileErrorKind {
    fn from(value: std::io::Error) -> Self { Self::Read(value) }
}

impl From<ParseError> for ReadFileErrorKind {
    fn from(value: ParseError) -> Self { Self::Parse(value) }
}

/// Error type when parsing a file.
#[derive(Debug)]
pub struct ParseError {
    /// Line number where the error occurred.
    pub line: u32,
    /// Position in the line where the error occurred.
    pub position: u32,
    /// Kind of error that occurred.
    pub kind: ParseErrorKind,
    /// Encoding of the file.
    pub encoding: FileEncoding,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.encoding {
            FileEncoding::Ascii => {
                write!(
                    f,
                    "error while parsing line {} at position {}: ",
                    self.line, self.position
                )?;
            }
            FileEncoding::Binary => {
                write!(
                    f,
                    "error while parsing byte at position {}: ",
                    self.position
                )?;
            }
        }
        match &self.kind {
            ParseErrorKind::InvalidMagicNumber => write!(f, "invalid magic number"),
            ParseErrorKind::InvalidEncoding => write!(f, "invalid encoding"),
            ParseErrorKind::InvalidCompression => write!(f, "invalid compression"),
            ParseErrorKind::InvalidLine => write!(f, "invalid line"),
            ParseErrorKind::ParseFloat => write!(f, "invalid float"),
            ParseErrorKind::NotEnoughData => write!(f, "not enough data"),
            ParseErrorKind::InvalidContent => write!(f, "invalid content"),
            ParseErrorKind::InvalidUft8 => write!(f, "invalid utf8"),
        }
    }
}

impl std::error::Error for ParseError {}

/// All possible errors while parsing a file.
#[derive(Debug)]
pub enum ParseErrorKind {
    /// The parsed content is not expected.
    InvalidContent,
    /// The parsed content is not valid UTF-8.
    InvalidUft8,
    /// The magic number is not valid.
    InvalidMagicNumber,
    /// The encoding is not valid.
    InvalidEncoding,
    /// The compression is not valid.
    InvalidCompression,
    /// The line is not valid.
    InvalidLine,
    /// The float is not valid.
    ParseFloat,
    /// There is not enough data to parse.
    NotEnoughData,
}

/// Possible errors while writing a file.
#[derive(Debug)]
pub struct WriteFileError {
    /// The path of the file.
    pub path: Box<Path>,
    /// The kind of the error.
    pub kind: WriteFileErrorKind,
}

impl Display for WriteFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error while writing file: {}", self.path.display())
    }
}

impl std::error::Error for WriteFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            WriteFileErrorKind::Write(err) => Some(err),
        }
    }
}

impl WriteFileError {
    /// Creates a new `WriteFileError` from a `std::io::Error`.
    pub fn from_std_io_error(path: impl AsRef<Path>, err: std::io::Error) -> Self {
        Self {
            path: path.as_ref().to_path_buf().into_boxed_path(),
            kind: WriteFileErrorKind::Write(err),
        }
    }
}

/// The kind of the error while writing a file.
#[derive(Debug)]
pub enum WriteFileErrorKind {
    /// The error is a `std::io::Error`.
    Write(std::io::Error),
}

impl From<std::io::Error> for WriteFileErrorKind {
    fn from(value: std::io::Error) -> Self { Self::Write(value) }
}

/// Data encoding while storing the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
#[repr(u8)]
pub enum FileEncoding {
    /// The data is encoded as ascii text (plain text).
    Ascii = 0x23, // '#'
    /// The data is encoded as binary data.
    Binary = 0x21, // '!'
}

impl From<u8> for FileEncoding {
    fn from(value: u8) -> Self {
        match value {
            0x23 => FileEncoding::Ascii,
            0x21 => FileEncoding::Binary,
            _ => panic!("Invalid data encoding: {}", value),
        }
    }
}

impl Display for FileEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileEncoding::Ascii => write!(f, "ascii"),
            FileEncoding::Binary => write!(f, "binary"),
        }
    }
}

impl FileEncoding {
    /// Returns true if the data is encoded as ascii text.
    pub fn is_ascii(&self) -> bool {
        match self {
            FileEncoding::Ascii => true,
            FileEncoding::Binary => false,
        }
    }

    /// Returns true if the data is encoded as binary data.
    pub fn is_binary(&self) -> bool {
        match self {
            FileEncoding::Ascii => false,
            FileEncoding::Binary => true,
        }
    }
}

/// Data compression scheme while storing the data.
#[repr(u8)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
pub enum CompressionScheme {
    /// No compression.
    None = 0x00,
    /// Zlib compression.
    Zlib = 0x01,
    /// Gzip compression.
    Gzip = 0x02,
}

impl From<u8> for CompressionScheme {
    fn from(value: u8) -> Self {
        match value {
            0x00 => CompressionScheme::None,
            0x01 => CompressionScheme::Zlib,
            0x02 => CompressionScheme::Gzip,
            _ => panic!("Invalid data compression: {}", value),
        }
    }
}

impl Display for CompressionScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionScheme::None => write!(f, "none"),
            CompressionScheme::Zlib => write!(f, "zlib"),
            CompressionScheme::Gzip => write!(f, "gzip"),
        }
    }
}

impl CompressionScheme {
    /// Returns true if the data is not compressed.
    pub fn is_none(&self) -> bool { matches!(self, CompressionScheme::None) }

    /// Returns true if the data is compressed with zlib.
    pub fn is_zlib(&self) -> bool { matches!(self, CompressionScheme::Zlib) }

    /// Returns true if the data is compressed with gzip.
    pub fn is_gzip(&self) -> bool { matches!(self, CompressionScheme::Gzip) }
}

/// Enum for different file variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VgonioFileVariant {
    /// A Measurement Output file.
    Vgmo,
    /// A Micro-surface profile file.
    Vgms,
}

/// Meta information of a VG file.
pub struct HeaderMeta {
    /// Version of the file.
    pub version: u32,
    /// Timestamp in RFC3339 and ISO 8601 format.
    pub timestamp: [u8; 32],
    /// Length of the whole file in bytes.
    pub length: u32,
    /// Size of a single sample in bytes.
    pub sample_size: u8,
    /// The file encoding.
    pub encoding: FileEncoding,
    /// The compression scheme.
    pub compression: CompressionScheme,
}

/// Trait for extra information of a file variant.
pub trait HeaderExt: Sized {
    /// The magic number of the file variant.
    const MAGIC: &'static [u8; 4];

    /// The size of the extra information in bytes in the memory.
    fn size() -> usize;

    /// The file variant.
    fn variant() -> VgonioFileVariant;

    /// Writes the extra information to a writer.
    fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> std::io::Result<()>;

    /// Reads the extra information from a reader.
    fn read<R: Read>(reader: &mut BufReader<R>) -> std::io::Result<Self>;
}

/// Header of a VG file.
pub struct Header<E: HeaderExt> {
    /// The meta information.
    pub meta: HeaderMeta,
    /// The file variant specific extra information.
    pub extra: E,
}

impl<E: HeaderExt> Header<E> {
    /// Writes the header to a writer.
    ///
    /// Note: the length of the whole file is not known at this point. Therefore
    /// the length field is set to 0. The length field is updated after the
    /// whole file is written.
    pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> Result<(), WriteFileErrorKind> {
        writer.write_all(E::MAGIC)?;
        writer.write_all(&self.meta.version.to_le_bytes())?;
        writer.write_all(&self.meta.length.to_le_bytes())?;
        writer.write_all(&self.meta.timestamp)?;
        writer.write_all(&[self.meta.sample_size])?;
        writer.write_all(&[self.meta.encoding as u8])?;
        writer.write_all(&[self.meta.compression as u8])?;
        writer.write_all(&[0u8; 1])?; // padding
        self.extra.write(writer)?;
        Ok(())
    }

    /// Returns the position of the length field in the serialized header.
    pub const fn length_pos() -> usize { E::MAGIC.len() + std::mem::size_of::<u32>() }

    /// Reads the header from the given reader.
    /// The reader must be positioned at the start of the header.
    pub fn read<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != *E::MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid magic number",
            ));
        } else {
            let mut buf = [0u8; 4];
            let version = {
                reader.read_exact(&mut buf)?;
                u32::from_le_bytes(buf)
            };
            let length = {
                reader.read_exact(&mut buf)?;
                u32::from_le_bytes(buf)
            };
            let timestamp = {
                let mut buf = [0u8; 32];
                reader.read_exact(&mut buf)?;
                buf
            };
            reader.read_exact(&mut buf)?;
            let sample_size = buf[0];
            let encoding = FileEncoding::from(buf[1]);
            let compression = CompressionScheme::from(buf[2]);
            let extra = E::read(reader)?;
            Ok(Self {
                meta: HeaderMeta {
                    version,
                    timestamp,
                    length,
                    sample_size,
                    encoding,
                    compression,
                },
                extra,
            })
        }
    }
}

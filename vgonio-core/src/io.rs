use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{fmt, fmt::Display, path::Path};

#[derive(Debug)]
pub struct ReadFileError {
    pub path: Box<Path>,
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

#[derive(Debug)]
pub enum ReadFileErrorKind {
    Read(std::io::Error),
    Parse(ParseError),
    InvalidFileFormat,
}

impl From<std::io::Error> for ReadFileErrorKind {
    fn from(value: std::io::Error) -> Self { Self::Read(value) }
}

impl From<ParseError> for ReadFileErrorKind {
    fn from(value: ParseError) -> Self { Self::Parse(value) }
}

#[derive(Debug)]
pub struct ParseError {
    pub line: u32,
    pub position: u32,
    pub kind: ParseErrorKind,
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

#[derive(Debug)]
pub enum ParseErrorKind {
    InvalidContent,
    InvalidUft8,
    InvalidMagicNumber,
    InvalidEncoding,
    InvalidCompression,
    InvalidLine,
    ParseFloat,
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

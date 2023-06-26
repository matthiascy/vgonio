//! Error type for vgonio.

use crate::io::{ReadFileError, WriteFileError};
use std::{
    error::Error,
    fmt::{Debug, Display, Formatter},
    str::Utf8Error,
};

/// Custom error type for vgonio.
#[derive(Debug)]
pub struct VgonioError {
    message: String,
    source: Option<Box<dyn Error>>,
}

impl Display for VgonioError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let cause = match &self.source {
            Some(cause) => format!("{}", cause),
            None => String::from("None"),
        };
        write!(f, "Error: {}, caused by {}", self.message, cause)
    }
}

impl Error for VgonioError {}

impl VgonioError {
    /// Create a new VgonioError.
    pub fn new<S>(message: S, source: Option<Box<dyn Error>>) -> Self
    where
        S: Into<String>,
    {
        Self {
            message: message.into(),
            source,
        }
    }

    /// Creates a new VgonioError from a Utf8Error.
    pub fn from_utf8_error<S>(err: Utf8Error, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(message, Some(Box::new(err)))
    }

    /// Creates a new VgonioError from a std::io::Error.
    pub fn from_io_error<S>(err: std::io::Error, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(message, Some(Box::new(err)))
    }

    /// Creates a new VgonioError from a ReadFileError.
    pub fn from_read_file_error<S>(err: ReadFileError, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(message, Some(Box::new(err)))
    }

    /// Creates a new VgonioError from a WriteFileError.
    pub fn from_write_file_error<S>(err: WriteFileError, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(message, Some(Box::new(err)))
    }
}

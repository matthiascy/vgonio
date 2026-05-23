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
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl Display for VgonioError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, source: {}",
            self.message,
            self.source
                .as_ref()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown".into())
        )
    }
}

// Override the default implementation of `source` to return the underlying error if it exists.
impl Error for VgonioError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|err| err as &(dyn Error + 'static))
    }
}

impl VgonioError {
    /// Create a new VgonioError.
    pub fn new<S>(message: S, source: Option<Box<dyn Error + Send + Sync>>) -> Self
    where
        S: Into<String>,
    {
        Self {
            message: message.into(),
            source,
        }
    }

    /// Creates a new `VgonioError` and stores the provided source error.
    pub fn with_source<S, E>(message: S, source: E) -> Self
    where
        S: Into<String>,
        E: Error + Send + Sync + 'static,
    {
        Self::new(message, Some(Box::new(source)))
    }

    /// Returns the error message.
    pub fn message(&self) -> &str { &self.message }

    /// Creates a new VgonioError from a Utf8Error.
    pub fn from_utf8_error<S>(err: Utf8Error, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::with_source(message, err)
    }

    /// Creates a new VgonioError from a std::io::Error.
    pub fn from_io_error<S>(err: std::io::Error, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::with_source(message, err)
    }

    /// Creates a new VgonioError from a ReadFileError.
    pub fn from_read_file_error<S>(err: ReadFileError, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::with_source(message, err)
    }

    /// Creates a new VgonioError from a WriteFileError.
    pub fn from_write_file_error<S>(err: WriteFileError, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::with_source(message, err)
    }
}

impl From<std::io::Error> for VgonioError {
    fn from(err: std::io::Error) -> Self { Self::with_source(err.to_string(), err) }
}

impl From<Utf8Error> for VgonioError {
    fn from(err: Utf8Error) -> Self { Self::with_source(err.to_string(), err) }
}

impl From<ReadFileError> for VgonioError {
    fn from(err: ReadFileError) -> Self { Self::with_source(err.to_string(), err) }
}

impl From<WriteFileError> for VgonioError {
    fn from(err: WriteFileError) -> Self { Self::with_source(err.to_string(), err) }
}

use crate::io::{ParseError, ReadFileError, WriteFileError};
use std::{
    error::Error,
    fmt::{Debug, Display, Formatter, Pointer},
    str::Utf8Error,
};

/// Custom error type for vgonio.
#[derive(Debug)]
pub struct VgonioError {
    message: String,
    cause: Option<Box<dyn Error>>,
}

impl Display for VgonioError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let cause = match &self.cause {
            Some(cause) => format!("{}", cause),
            None => String::from("None"),
        };
        write!(f, "[ERROR] {} caused by {}", self.message, cause)
    }
}

impl Error for VgonioError {}

impl VgonioError {
    /// Create a new VgonioError.
    pub fn new<S>(message: S, cause: Option<Box<dyn Error>>) -> Self
    where
        S: Into<String>,
    {
        Self {
            message: message.into(),
            cause,
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

    pub fn from_read_file_error<S>(err: ReadFileError, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(message, Some(Box::new(err)))
    }

    pub fn from_write_file_error<S>(err: WriteFileError, message: S) -> Self
    where
        S: Into<String>,
    {
        Self::new(message, Some(Box::new(err)))
    }
}

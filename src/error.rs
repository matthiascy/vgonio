use std::fmt::{Display, Formatter};
use std::str;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Rhi(wgpu::Error),
    Logger(log::SetLoggerError),
    BincodeError(bincode::Error),
    UnrecognizedFile,
    Utf8Error(str::Utf8Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(err) => {
                write!(f, "IO error: {}", err)
            }
            Error::Rhi(err) => {
                write!(f, "RHI error: {}", err)
            }
            Error::Logger(_) => {
                write!(f, "Set logger error.")
            }
            Error::UnrecognizedFile => {
                write!(f, "Open file failed:  unrecognized file type!")
            }
            Error::BincodeError(err) => {
                write!(f, "Bincode error: {}", err.to_string())
            }
            Error::Utf8Error(err) => {
                write!(f, "Utf8 error: {}", err)
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<wgpu::Error> for Error {
    fn from(err: wgpu::Error) -> Self {
        Error::Rhi(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<log::SetLoggerError> for Error {
    fn from(err: log::SetLoggerError) -> Self {
        Error::Logger(err)
    }
}

impl From<str::Utf8Error> for Error {
    fn from(err: str::Utf8Error) -> Self {
        Error::Utf8Error(err)
    }
}

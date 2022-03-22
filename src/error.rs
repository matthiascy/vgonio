use image::ImageError;
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
    YamlError(serde_yaml::Error),
    Any(String),
    FileError(&'static str),
    ImageError(image::ImageError),
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
                write!(f, "Bincode error: {}", err)
            }
            Error::Utf8Error(err) => {
                write!(f, "Utf8 error: {}", err)
            }
            Error::Any(err) => {
                write!(f, "Error: {}", err)
            }
            Error::FileError(err) => {
                write!(f, "File error: {}", err)
            }
            Error::YamlError(err) => {
                write!(f, "YAML error: {}", err)
            }
            Error::ImageError(err) => {
                write!(f, "Image error: {}", err)
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

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::BincodeError(err)
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(err: serde_yaml::Error) -> Self {
        Error::YamlError(err)
    }
}

impl From<image::ImageError> for Error {
    fn from(err: ImageError) -> Self {
        Error::ImageError(err)
    }
}

use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Rhi(wgpu::Error),
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

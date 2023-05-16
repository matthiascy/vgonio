use crate::io::{ReadFileError, WriteFileError};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
    str,
};

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Rhi(WgpuError),
    Logger(log::SetLoggerError),
    UnrecognizedFile,
    Utf8Error(str::Utf8Error),
    Any(String),
    FileError(&'static str),
    ImageError(image::ImageError),
    SerialisationError(SerialisationError),
    InvalidEmitter(&'static str),
    InvalidCollector(&'static str),
    SysConfigDirNotFound,
    SysCacheDirNotFound,
    SysDataDirNotFound,
    UserConfigNotFound,
    UserDataDirNotConfigured,
    InvalidOutputDir(PathBuf),
    InvalidParameter(&'static str),
    DirectoryOrFileNotFound(PathBuf),

    ReadFile(ReadFileError),
    WriteFile(WriteFileError),
}

#[derive(Debug)]
pub struct WgpuError {
    source: Box<dyn std::error::Error + Send + 'static>,
}

impl WgpuError {
    pub fn is_surface_error(&self) -> bool { self.source.is::<wgpu::SurfaceError>() }

    pub fn is_request_device_error(&self) -> bool { self.source.is::<wgpu::RequestDeviceError>() }

    pub fn is_create_surface_error(&self) -> bool { self.source.is::<wgpu::CreateSurfaceError>() }

    pub fn is_buffer_async_error(&self) -> bool { self.source.is::<wgpu::BufferAsyncError>() }

    pub fn get<T: std::error::Error + 'static>(&self) -> Option<&T> {
        self.source.downcast_ref::<T>()
    }
}

#[derive(Debug)]
pub enum SerialisationError {
    TomlSe(toml::ser::Error),
    TomlDe(toml::de::Error),
    Yaml(serde_yaml::Error),
    Bincode(bincode::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(err) => {
                write!(f, "IO error - {err}")
            }
            Error::Rhi(err) => {
                write!(f, "RHI error - {err:?}")
            }
            Error::Logger(_) => {
                write!(f, "Set logger error.")
            }
            Error::UnrecognizedFile => {
                write!(f, "Open file failed:  unrecognized file type!")
            }
            Error::Utf8Error(err) => {
                write!(f, "Utf8 error - {err}")
            }
            Error::Any(err) => {
                write!(f, "Error - {err}")
            }
            Error::FileError(err) => {
                write!(f, "File error - {err}")
            }
            Error::ImageError(err) => {
                write!(f, "Image error - {err}")
            }
            Error::SysConfigDirNotFound => {
                write!(f, "System configuration directory not found!")
            }
            Error::SysCacheDirNotFound => {
                write!(f, "System cache directory not found!")
            }
            Error::SysDataDirNotFound => {
                write!(f, "System cache directory not found!")
            }
            Error::UserConfigNotFound => {
                write!(f, "Specified user configuration file not found!")
            }
            Error::SerialisationError(err) => match err {
                SerialisationError::TomlSe(err) => {
                    write!(f, "Toml serialisation error: {err}")
                }
                SerialisationError::TomlDe(err) => {
                    write!(f, "Toml deserialization error: {err}")
                }
                SerialisationError::Yaml(err) => {
                    write!(f, "Yaml de/serialisation error: {err}")
                }
                SerialisationError::Bincode(err) => {
                    write!(f, "Bincode de/serialisation error: {err}")
                }
            },
            Error::InvalidEmitter(err) => {
                write!(f, "Invalid emitter: {err}")
            }
            Error::InvalidCollector(err) => {
                write!(f, "Invalid collector: {err}")
            }
            #[cfg(target_arch = "wasm32")]
            Error::WasmConsoleLogInitFailed => {
                write!(
                    f,
                    "Failed to initialise console log for vgonio wasm module!"
                )
            }
            Error::UserDataDirNotConfigured => {
                write!(
                    f,
                    "Try to load file from user data directory which doesn't exist!"
                )
            }
            Error::InvalidOutputDir(path) => {
                write!(f, "Invalid output path {}!", path.display())
            }
            Error::InvalidParameter(err) => {
                write!(f, "Invalid parameter: {err}")
            }
            Error::DirectoryOrFileNotFound(path) => {
                write!(f, "Directory or file not found: {}", path.display())
            }
            Error::ReadFile(err) => {
                write!(f, "Read file error: {}", err)
            }
            Error::WriteFile(err) => {
                write!(f, "Write file error: {}", err)
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self { Error::Io(err) }
}

impl From<log::SetLoggerError> for Error {
    fn from(err: log::SetLoggerError) -> Self { Error::Logger(err) }
}

impl From<str::Utf8Error> for Error {
    fn from(err: str::Utf8Error) -> Self { Error::Utf8Error(err) }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Self {
        Error::SerialisationError(SerialisationError::Bincode(err))
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(err: serde_yaml::Error) -> Self {
        Error::SerialisationError(SerialisationError::Yaml(err))
    }
}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self { Error::ImageError(err) }
}

impl From<toml::ser::Error> for Error {
    fn from(err: toml::ser::Error) -> Self {
        Error::SerialisationError(SerialisationError::TomlSe(err))
    }
}

impl From<toml::de::Error> for Error {
    fn from(err: toml::de::Error) -> Self {
        Error::SerialisationError(SerialisationError::TomlDe(err))
    }
}

impl From<WriteFileError> for Error {
    fn from(value: WriteFileError) -> Self { Self::WriteFile(value) }
}

impl From<ReadFileError> for Error {
    fn from(value: ReadFileError) -> Self { Self::ReadFile(value) }
}

macro impl_from_wgpu_errors($($err:ty),*) {
$(
        impl From<$err> for Error {
            fn from(source: $err) -> Self {
                Error::Rhi(WgpuError {
                    source: Box::new(source),
                })
            }
        }
    )*
}

impl_from_wgpu_errors!(
    wgpu::Error,
    wgpu::RequestDeviceError,
    wgpu::SurfaceError,
    wgpu::CreateSurfaceError,
    wgpu::BufferAsyncError
);

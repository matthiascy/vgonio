use crate::error::RuntimeError::Serialisation;
use std::fmt::{Debug, Display, Formatter};

/// Error type for runtime errors.
#[derive(Debug)]
pub enum RuntimeError {
    /// The output directory is invalid.
    InvalidOutputDir,
    /// The system configuration directory is not found.
    SysConfigDirNotFound,
    /// The system cache directory is not found.
    SysCacheDirNotFound,
    /// The system data directory is not found.
    SysDataDirNotFound,
    /// De/Serialisation error.
    Serialisation(SerialisationError),
    /// Logger error.
    Logger(log::SetLoggerError),
    /// Invalid parameters.
    InvalidParameters,
    /// Directory or file isn't found.
    DirectoryOrFileNotFound,
    /// Invalid emitter.
    InvalidEmitter,
    /// Invalid collector.
    InvalidDetector,
    /// User configuration not found.
    UserConfigNotFound,
    /// User data directory not configured.
    UserDataDirNotConfigured,
    /// Rendering Hardware Interface error.
    Rhi(WgpuError),
    /// Image error.
    Image(image::ImageError),
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOutputDir => write!(f, "Invalid output directory"),
            Self::SysConfigDirNotFound => write!(f, "System configuration directory not found"),
            Self::SysCacheDirNotFound => write!(f, "System cache directory not found"),
            Self::SysDataDirNotFound => write!(f, "System data directory not found"),
            Self::Serialisation(err) => write!(f, "Serialisation error: {}", err),
            Self::Logger(err) => write!(f, "Logger error: {}", err),
            Self::InvalidParameters => write!(f, "Invalid parameters"),
            Self::DirectoryOrFileNotFound => write!(f, "Directory or file not found"),
            Self::InvalidEmitter => write!(f, "Invalid emitter"),
            Self::InvalidDetector => write!(f, "Invalid collector"),
            Self::UserConfigNotFound => write!(f, "User configuration not found"),
            Self::UserDataDirNotConfigured => write!(f, "User data directory not configured"),
            Self::Rhi(err) => write!(f, "Rendering Hardware Interface error: {}", err),
            Self::Image(err) => write!(f, "Image error: {}", err),
        }
    }
}

impl std::error::Error for RuntimeError {}

unsafe impl Send for RuntimeError {}
unsafe impl Sync for RuntimeError {}

#[derive(Debug)]
pub struct WgpuError {
    source: Box<dyn std::error::Error + Send + 'static>,
}

impl Display for WgpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.source) }
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

impl Display for SerialisationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TomlSe(err) => write!(f, "Toml serialisation error: {}", err),
            Self::TomlDe(err) => write!(f, "Toml deserialisation error: {}", err),
            Self::Yaml(err) => write!(f, "Yaml error: {}", err),
            Self::Bincode(err) => write!(f, "Bincode error: {}", err),
        }
    }
}

impl From<log::SetLoggerError> for RuntimeError {
    fn from(err: log::SetLoggerError) -> Self { RuntimeError::Logger(err) }
}

impl From<image::ImageError> for RuntimeError {
    fn from(err: image::ImageError) -> Self { RuntimeError::Image(err) }
}

impl From<bincode::Error> for RuntimeError {
    fn from(err: bincode::Error) -> Self { Serialisation(SerialisationError::Bincode(err)) }
}

impl From<serde_yaml::Error> for RuntimeError {
    fn from(err: serde_yaml::Error) -> Self { Serialisation(SerialisationError::Yaml(err)) }
}

impl From<toml::ser::Error> for RuntimeError {
    fn from(err: toml::ser::Error) -> Self { Serialisation(SerialisationError::TomlSe(err)) }
}

impl From<toml::de::Error> for RuntimeError {
    fn from(err: toml::de::Error) -> Self { Serialisation(SerialisationError::TomlDe(err)) }
}

macro impl_from_wgpu_errors($($err:ty),*) {
$(
        impl From<$err> for RuntimeError {
            fn from(source: $err) -> Self {
                RuntimeError::Rhi(WgpuError {
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

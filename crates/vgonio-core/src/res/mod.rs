//! Application Asset Management

use std::{fmt::Debug, path::Path};

mod asset;
mod handle;
mod loader;
mod registry;
mod store;

use crate::optics::{IorFileError, IorReg, IorRegLoader};

pub use asset::*;
pub use handle::*;
pub use loader::*;
pub use registry::*;
pub use store::*;

/// Errors that can occur during asset management operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Asset type not found in the storage.
    #[error("Asset type {0} not found in the storage")]
    AssetTypeNotFound(&'static str),

    /// Asset with the given handle not found in the storage.
    #[error("Asset with ID {0} not found in the storage")]
    AssetNotFound(Handle),

    /// Type mismatch when retrieving an asset.
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// Expected type name.
        expected: &'static str,
        /// Actual type name.
        actual: &'static str,
    },

    /// I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Asset loader not found for the specified asset type.
    #[error("No loader found for asset of type {0}")]
    LoaderNotFound(&'static str),

    /// Unknown asset type ID.
    #[error("Unknown asset type ID {0}")]
    UnknownAssetTypeId(u8),

    /// Provided path is not a valid directory.
    #[error("Provided path '{0}' is not a valid directory")]
    InvalidDirectory(String),

    /// A loader that depends on the medium identity registry was invoked before
    /// `medium::bootstrap()` had run.
    #[error(
        "cannot load this asset before medium::bootstrap() has run: the medium identity registry \
         is the authority on which media exist"
    )]
    SpineNotBootstrapped,

    /// A medium declared in the built-in identity registry (`builtin.toml`) has no
    /// IOR dataset in any layer. The IOR registry must cover every baseline
    /// medium — Vacuum is the sole exception (it's mathematically privileged and
    /// never needs a dataset).
    #[error(
        "built-in medium {0:?} has no IOR dataset in the embedded/system/user layers. \
         Add `datafiles/ior/{0}_<source>.ior.ron` and a matching `sources.toml` entry, \
         or remove the medium from `crates/vgonio-core/src/utils/medium/builtin.toml`."
    )]
    MissingBuiltinIor(String),
}

impl From<IorFileError> for Error {
    fn from(e: IorFileError) -> Self {
        match e {
            IorFileError::Io { path, source } => Error::IoError(source),
            IorFileError::Ron { path, source } => Error::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse RON file: {source}"),
            )),
            IorFileError::Toml(_) => todo!(),
            IorFileError::Csv { path, msg } => Error::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to parse CSV file: {msg}"),
            )),
            IorFileError::SchemaVersion {
                path,
                found,
                expected,
            } => Error::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported schema version: found {found}, expected {expected}"),
            )),
            IorFileError::UnknownMedium(path, medium) => Error::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown medium '{medium}' in file: {path}"),
            )),
            IorFileError::Inconsistent(path, msg) => Error::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Inconsistent asset data in file {path}: {msg}"),
            )),
        }
    }
}

/// Structure for caching intermediate results and data.
/// Also used for managing assets.
pub struct RawDataStore {
    /// Generic storage for assets that don't need specific handling.
    pub assets: AssetsStorage,
    /// Handle to the refractive index database.
    pub ior_db: Option<Handle>,
}

impl RawDataStore {
    /// Creates a new `RawDataStore` with the given cache directory.
    ///
    /// If `load_ior_db` is true, the refractive index database is loaded from
    /// the cache directory.
    ///
    /// # Arguments
    ///
    /// * `load_ior_db` - Whether to load the refractive index database.
    /// * `sys_data_dir` - Path to the system data directory; can be obtained from
    ///   `Config::sys_data_dir()`.
    /// * `user_data_dir` - Path to the user data directory; can be obtained from
    ///   `Config::user_data_dir()`.
    pub fn new(
        load_ior_db: bool,
        sys_data_dir: Option<&Path>,
        user_data_dir: Option<&Path>,
        exclude_ior_files: Option<Vec<String>>,
    ) -> Self {
        let mut assets = AssetsStorage::default();

        let loader = IorRegLoader::new(sys_data_dir, user_data_dir, exclude_ior_files);
        assets.register_loader::<IorReg>(Box::new(loader));

        if load_ior_db {
            let ior_db = assets.load_asset::<IorReg>(None).unwrap();
            Self {
                assets,
                ior_db: Some(ior_db),
            }
        } else {
            Self {
                assets,
                ior_db: None,
            }
        }
    }

    /// Inserts an asset into the storage, returning a handle to it.
    pub fn insert<T: Asset>(&mut self, asset: T) -> Result<Handle, Error> {
        self.assets.insert_asset(Box::new(asset))
    }

    /// Gets a reference to the asset with the given handle.
    pub fn get<T: Asset>(&self, handle: Handle) -> Result<&T, Error> {
        self.assets
            .get_store::<T>()
            .ok_or(Error::AssetTypeNotFound(T::asset_type()))
            .and_then(|store| store.get(&handle).ok_or(Error::AssetNotFound(handle)))
            .and_then(|asset| {
                asset
                    .as_any()
                    .downcast_ref::<T>()
                    .ok_or(Error::TypeMismatch {
                        expected: T::asset_type(),
                        actual: asset.own_type_name(),
                    })
            })
    }

    /// Gets a mutable reference to the asset with the given handle.
    pub fn get_mut<T: Asset>(&mut self, handle: Handle) -> Result<&mut T, Error> {
        self.assets
            .get_store_mut::<T>()
            .ok_or(Error::AssetTypeNotFound(T::asset_type()))
            .and_then(|store| store.get_mut(&handle).ok_or(Error::AssetNotFound(handle)))
            .and_then(|asset| {
                let asset_type_name = asset.own_type_name();
                asset
                    .as_any_mut()
                    .downcast_mut::<T>()
                    .ok_or(Error::TypeMismatch {
                        expected: T::asset_type(),
                        actual: asset_type_name,
                    })
            })
    }

    /// Removes an asset from the storage without returning it.
    pub fn remove<T: Asset>(&mut self, handle: Handle) -> Result<(), Error> {
        if T::asset_type_id() != handle.asset_type_id() {
            let actual = handle.asset_type_name();
            return Err(Error::TypeMismatch {
                expected: T::asset_type(),
                actual,
            });
        }
        let _ = self.assets.remove_asset(handle);

        Ok(())
    }

    /// Iterates over all assets of a specific type.
    pub fn iter<T: Asset>(&self) -> impl Iterator<Item = &T> {
        self.assets
            .get_store::<T>()
            .into_iter()
            .flat_map(|store| store.values())
            .filter_map(|asset| asset.as_any().downcast_ref::<T>())
    }

    /// Iterates over all assets of a specific type mutably.
    pub fn iter_mut<T: Asset>(&mut self) -> impl Iterator<Item = &mut T> {
        self.assets
            .get_store_mut::<T>()
            .into_iter()
            .flat_map(|store| store.values_mut())
            .filter_map(|asset| asset.as_any_mut().downcast_mut::<T>())
    }
}

/// A thread-safe runtime data store that provides synchronized access to the
/// underlying `RawDataStore`.
#[derive(Clone)]
pub struct DataStore(std::sync::Arc<std::sync::RwLock<RawDataStore>>);

impl DataStore {
    /// Creates a new `DataStore` from the given `RawDataStore`.
    pub fn from_raw(inner: RawDataStore) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(inner)))
    }

    /// Creates a new `DataStore` with the given configuration.
    ///
    /// This is a convenience method that allows you to create a `DataStore` by
    /// specifying whether to load the refractive index database and providing
    /// the necessary paths and settings directly, without needing to
    /// construct a `RawDataStore` first.
    ///
    /// # Arguments
    ///
    /// - `load_ior_db`: Whether to load the refractive index database. If `true`, the database will
    ///   be loaded from the specified system and user data directories. If `false`, the database
    ///   will not be loaded, and the `ior_db` field in the `RawDataStore` will be set to `None`.
    /// - `sys_data_dir`: Optional path to the system data directory. This is where the refractive
    ///   index database will be loaded from if `load_ior_db` is `true`. If `None`, the database
    ///   will not be loaded from the system data directory.
    /// - `user_data_dir`: Optional path to the user data directory. This is where the refractive
    ///   index database will be loaded from if `load_ior_db` is `true`. If `None`, the database
    ///   will not be loaded from the user data directory.
    /// - `excluded_ior_files`: Optional list of file names to exclude when loading the refractive
    ///   index database. This allows you to specify certain files that should not be loaded, even
    ///   if they are present in the specified directories. If `None`, no files will be excluded.
    ///
    /// # Returns
    /// A new `DataStore` instance initialized with the specified configuration.
    pub fn new(
        load_ior_db: bool,
        sys_data_dir: Option<&Path>,
        user_data_dir: Option<&Path>,
        excluded_ior_files: Option<Vec<String>>,
    ) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(
            RawDataStore::new(load_ior_db, sys_data_dir, user_data_dir, excluded_ior_files),
        )))
    }

    /// Creates a new `DataStore` from the given configuration.
    ///
    /// This is a convenience method that extracts the necessary paths and
    /// settings from the provided `Config` object.
    ///
    /// # Arguments
    ///
    /// - `load_ior_db`: Whether to load the refractive index database. If `true`, the database will
    ///   be loaded from the system and user data directories specified in the configuration. If
    ///   `false`, the database will not be loaded, and the `ior_db` field in the `RawDataStore`
    ///   will be set to `None`.
    /// - `config`: The configuration object containing the necessary paths and settings.
    #[cfg(feature = "config")]
    pub fn new_from_config(load_ior_db: bool, config: &crate::config::Config) -> Self {
        Self::new(
            load_ior_db,
            Some(config.sys_data_dir()),
            config.user_data_dir(),
            config.user.excluded_ior_files.clone(),
        )
    }

    /// Provides read-only access to the cache for the duration of the provided
    /// closure.
    ///
    /// This method locks the cache for reading, allowing the closure to access
    /// its contents safely. The lock is released automatically when the
    /// closure finishes executing.
    ///
    /// # Arguments
    ///
    /// - `reader`: A closure that takes a reference to the `RawDataStore` and returns a value of
    ///   type `R`. This closure will be executed with shared access to the cache.
    pub fn read<R>(&self, reader: impl FnOnce(&RawDataStore) -> R) -> R {
        let cache = self.0.read().unwrap();
        reader(&cache)
    }

    /// Provides mutable access to the cache for the duration of the provided
    /// closure.
    ///
    /// This method locks the cache for writing, allowing the closure to modify
    /// its contents safely. The lock is released automatically when the closure
    /// finishes executing.
    ///
    /// # Arguments
    ///
    /// - `writer`: A closure that takes a mutable reference to the `RawDataStore` and returns a
    ///   value of type `R`. This closure will be executed with exclusive access to the cache.
    pub fn write<R>(&self, writer: impl FnOnce(&mut RawDataStore) -> R) -> R {
        let mut cache = self.0.write().unwrap();
        writer(&mut cache)
    }
}

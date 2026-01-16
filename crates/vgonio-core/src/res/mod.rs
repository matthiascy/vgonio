//! Application Asset Management

use std::{fmt::Debug, path::Path};

mod asset;
mod handle;
mod loader;
mod registry;
mod store;

use crate::optics::{IorReg, IorRegLoader};

pub use asset::*;
pub use handle::*;
pub use loader::*;
pub use registry::*;
pub use store::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Asset type {0} not found in the storage")]
    AssetTypeNotFound(&'static str),

    #[error("Asset with ID {0} not found in the storage")]
    AssetNotFound(Handle),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("No loader found for asset of type {0}")]
    LoaderNotFound(&'static str),

    #[error("Unknown asset type ID {0}")]
    UnknownAssetTypeId(u8),

    #[error("Provided path '{0}' is not a valid directory")]
    InvalidDirectory(String),
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
    /// * `sys_data_dir` - Path to the system data directory; can be obtained
    ///   from `Config::sys_data_dir()`.
    /// * `user_data_dir` - Path to the user data directory; can be obtained
    ///   from `Config::user_data_dir()`.
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

/// A thread-safe cache. This is a wrapper around `RawCache`.
#[derive(Clone)]
pub struct DataStore(std::sync::Arc<std::sync::RwLock<RawDataStore>>);

impl DataStore {
    pub fn from_raw(inner: RawDataStore) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(inner)))
    }

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

    #[cfg(feature = "config")]
    pub fn new_from_config(load_ior_db: bool, config: &crate::config::Config) -> Self {
        Self::new(
            load_ior_db,
            Some(config.sys_data_dir()),
            config.user_data_dir(),
            config.user.excluded_ior_files.clone(),
        )
    }

    pub fn read<R>(&self, reader: impl FnOnce(&RawDataStore) -> R) -> R {
        let cache = self.0.read().unwrap();
        reader(&cache)
    }

    pub fn write<R>(&self, writer: impl FnOnce(&mut RawDataStore) -> R) -> R {
        let mut cache = self.0.write().unwrap();
        writer(&mut cache)
    }
}

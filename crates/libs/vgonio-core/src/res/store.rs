use crate::res::{loader::AssetLoader, Asset, AssetTypeId, AssetTypeRegistry, Error, Handle};
use std::{collections::HashMap, path::Path};

/// Container for assets of the same type.
pub type AssetsContainer = HashMap<Handle, Box<dyn Asset>>;

/// A generic asset storage that can store any type of asset.
///
/// The storage is generic over the asset type, and the asset type id is used
/// as the key in the storage. No access to the underlying storage is provided,
/// but user can retrieve/insert/update/remove assets by their handle.
#[derive(Default)]
pub struct AssetsStorage {
    /// Stores assets by type.
    assets: HashMap<AssetTypeId, AssetsContainer>,
    /// List of asset loaders.
    loaders: HashMap<AssetTypeId, Box<dyn AssetLoader>>,
    // TODO: Asset dependencies.
    // dependencies: HashMap<Handle, AssetDeps>,
}

impl AssetsStorage {
    /// Registers an asset loader for a specific asset type.
    pub fn register_loader<T: Asset>(&mut self, loader: Box<dyn AssetLoader>) {
        let asset_type_id = T::asset_type_id();
        self.loaders.insert(asset_type_id, loader);
    }

    /// Loads asset of a specific type.
    pub fn load_asset<T: Asset>(&mut self, path: Option<&Path>) -> Result<Handle, Error> {
        let asset_type_id = T::asset_type_id();
        let loader = self
            .loaders
            .get(&asset_type_id)
            .ok_or(Error::LoaderNotFound(T::asset_type()))?;
        let asset = loader.load(path)?;
        self.insert_asset(asset)
    }

    /// Loads assets from a directory recursively.
    pub fn load_directory<T: Asset>(&mut self, path: &Path) -> Result<(), Error> {
        if !path.is_dir() {
            return Err(Error::InvalidDirectory(
                path.to_path_buf().to_str().unwrap().to_string(),
            ));
        }

        for entry in path.read_dir()? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                self.load_directory::<T>(&entry.path())?;
            } else {
                self.load_asset::<T>(Some(&entry.path()))?;
            }
        }
        Ok(())
    }

    /// Inserts an asset into the store.
    pub fn insert_asset(&mut self, asset: Box<dyn Asset>) -> Result<Handle, Error> {
        let asset_type_id = asset.own_type_id();
        let handle = Handle::new_with_asset_id(asset_type_id);
        let store = self
            .assets
            .entry(asset_type_id)
            .or_insert_with(HashMap::new);
        store.insert(handle, asset);
        Ok(handle)
    }

    /// Updates an asset in the store.
    ///
    /// # Errors
    ///
    /// Returns an error if the asset type is not found or the asset is not
    /// found.
    pub fn update_asset(&mut self, handle: Handle, asset: Box<dyn Asset>) -> Result<(), Error> {
        let store =
            self.assets
                .get_mut(&handle.asset_type_id())
                .ok_or(Error::AssetTypeNotFound(
                    AssetTypeRegistry::asset_type_name(handle.asset_type_id().0).unwrap(),
                ))?;
        if store.contains_key(&handle) {
            store.insert(handle, asset);
            Ok(())
        } else {
            Err(Error::AssetNotFound(handle))
        }
    }

    /// Removes an asset from the storage.
    pub fn remove_asset(&mut self, handle: Handle) -> Option<Box<dyn Asset>> {
        self.assets
            .get_mut(&handle.asset_type_id())
            .and_then(|store| store.remove(&handle))
    }

    /// Clears all assets of a specific type from the storage.
    pub fn clear_assets<T: Asset>(&mut self) { self.assets.remove(&T::asset_type_id()); }

    /// Clears all assets from the storage.
    pub fn clear_all(&mut self) { self.assets.clear(); }

    /// Checks if the storage contains an asset of a specific type.
    pub fn contains_assets<T: Asset>(&self) -> bool {
        self.assets.contains_key(&T::asset_type_id())
    }

    /// Gets an asset from the storage by handle.
    pub fn get_asset(&self, handle: Handle) -> Option<&Box<dyn Asset>> {
        self.assets
            .get(&handle.asset_type_id())
            .and_then(|store| store.get(&handle))
    }

    /// Gets an asset from the storage by handle and type.
    pub fn get_asset_with_type<T: Asset>(&self, handle: Handle) -> Option<&T> {
        self.assets
            .get(&T::asset_type_id())
            .and_then(|store| store.get(&handle))
            .and_then(|asset| asset.as_any().downcast_ref::<T>())
    }

    /// Gets a mutable asset from the storage by handle.
    pub fn get_asset_mut(&mut self, handle: Handle) -> Option<&mut Box<dyn Asset>> {
        self.assets
            .get_mut(&handle.asset_type_id())
            .and_then(|store| store.get_mut(&handle))
    }

    /// Gets a mutable asset from the storage by handle and type.
    pub fn get_asset_with_type_mut<T: Asset>(&mut self, handle: Handle) -> Option<&mut T> {
        self.assets
            .get_mut(&T::asset_type_id())
            .and_then(|store| store.get_mut(&handle))
            .and_then(|asset| asset.as_any_mut().downcast_mut::<T>())
    }

    /// Gets a store for a specific asset type if it exists.
    pub fn get_store<T: Asset>(&self) -> Option<&HashMap<Handle, Box<dyn Asset>>> {
        self.assets.get(&T::asset_type_id())
    }

    /// Gets a loader for a specific asset type if it exists.
    pub fn get_loader<T: Asset>(&self) -> Option<&Box<dyn AssetLoader>> {
        self.loaders.get(&T::asset_type_id())
    }

    /// Gets a mutable store for a specific asset type if it exists.
    pub(crate) fn get_store_mut<T: Asset>(
        &mut self,
    ) -> Option<&mut HashMap<Handle, Box<dyn Asset>>> {
        self.assets.get_mut(&T::asset_type_id())
    }

    /// Gets a mutable loader for a specific asset type if it exists.
    pub(crate) fn get_loader_mut<T: Asset>(&mut self) -> Option<&mut Box<dyn AssetLoader>> {
        self.loaders.get_mut(&T::asset_type_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset;

    #[derive(Debug, PartialEq)]
    struct TestAsset {
        value: i32,
    }

    asset!(TestAsset, "TestAsset");

    #[test]
    fn test_asset_store() {
        let mut store = AssetsStorage::default();

        // Test insert and get
        let asset = TestAsset { value: 42 };
        let handle = store.insert_asset(Box::new(asset)).unwrap();
        assert_eq!(
            store
                .get_asset_with_type::<TestAsset>(handle)
                .unwrap()
                .value,
            42
        );

        // Test update
        store
            .update_asset(handle, Box::new(TestAsset { value: 43 }))
            .unwrap();
        assert_eq!(
            store
                .get_asset_with_type::<TestAsset>(handle)
                .unwrap()
                .value,
            43
        );

        // Test remove
        let removed = store.remove_asset(handle).unwrap();
        assert_eq!(
            removed.as_any().downcast_ref::<TestAsset>().unwrap().value,
            43
        );
        assert!(store.get_asset(handle).is_none());

        // Test clear
        let _ = store
            .insert_asset(Box::new(TestAsset { value: 1 }))
            .unwrap();
        let _ = store
            .insert_asset(Box::new(TestAsset { value: 2 }))
            .unwrap();
        assert_eq!(store.get_store::<TestAsset>().unwrap().len(), 2);
        store.clear_assets::<TestAsset>();
        assert_eq!(store.get_store::<TestAsset>().unwrap().len(), 0);
    }
}

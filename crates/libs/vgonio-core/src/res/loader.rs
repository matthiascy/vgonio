use crate::res::{Asset, Error};
use std::path::Path;

use super::{AssetTypeId, AssetTypeRegistry};

/// Trait for loading assets from raw data.
pub trait AssetLoader: Send + Sync {
    /// Returns the type of asset that this loader can load. This should be the
    /// same as what the `asset_type` method returns.
    fn asset_type(&self) -> &'static str {
        AssetTypeRegistry::asset_type_name(self.asset_type_id().0).unwrap()
    }

    /// Returns the type of asset that this loader can load. This should be the
    /// same as what the `asset_type_id` method returns.
    fn asset_type_id(&self) -> AssetTypeId;

    /// Loads from a path.
    ///
    /// If `path` is `None`, the loader will load from the default path defined
    /// by the actual implementation.
    fn load(&self, path: Option<&Path>) -> Result<Box<dyn Asset>, Error>;
}

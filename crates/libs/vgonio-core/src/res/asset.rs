use crate::res::Handle;
use std::{
    any::Any,
    sync::atomic::{AtomicU8, Ordering},
};

/// Unique identifier for asset types.
static ASSET_TYPE_ID: AtomicU8 = AtomicU8::new(0);

pub struct AssetTypeIdAllocator;

impl AssetTypeIdAllocator {
    pub fn next() -> AssetTypeId {
        let id = ASSET_TYPE_ID.fetch_add(1, Ordering::Relaxed);
        if id == u8::MAX {
            panic!("Too many asset types registered!");
        }
        AssetTypeId(id)
    }
}

/// Unique identifier for asset types within the application.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AssetTypeId(pub u8);

impl From<u8> for AssetTypeId {
    fn from(value: u8) -> Self { Self(value) }
}

impl From<AssetTypeId> for u8 {
    fn from(value: AssetTypeId) -> Self { value.0 }
}

impl std::fmt::Display for AssetTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AssetTypeId#{}", self.0)
    }
}

/// Trait for assets that can be stored.
pub trait Asset: Any + Send + Sync + 'static {
    /// Unique identifier for the asset type.
    fn asset_type() -> &'static str
    where
        Self: Sized;

    /// Auto-generated unique asset type identifier inside the application
    /// within the range of 0-255.
    fn asset_type_id() -> AssetTypeId
    where
        Self: Sized;

    /// Returns the asset type identifier of the asset object.
    fn own_type_id(&self) -> AssetTypeId;

    /// Returns the asset type name of the asset object.
    fn own_type_name(&self) -> &'static str;

    /// Enables downcasting to the actual asset type using `Any`.
    fn as_any(&self) -> &dyn Any;

    /// Enables mutable downcasting to the actual asset type using `Any`.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Creates a new handle for the asset.
    fn new_handle() -> Handle
    where
        Self: Sized,
    {
        Handle::new::<Self>()
    }
}

#[macro_export]
macro_rules! asset {
    ($t:ty, $n:expr) => {
        impl $crate::res::Asset for $t {
            fn asset_type() -> &'static str { $n }

            fn asset_type_id() -> $crate::res::AssetTypeId {
                use $crate::res::{AssetTypeId, AssetTypeIdAllocator, AssetTypeRegistry};

                static ONCE: std::sync::Once = std::sync::Once::new();
                static mut TYPE_ID: AssetTypeId = AssetTypeId(0);

                ONCE.call_once(|| {
                    let id = AssetTypeIdAllocator::next();
                    unsafe {
                        TYPE_ID = id;
                    }
                    AssetTypeRegistry::register_asset_type(id.0, $n);
                });

                unsafe { TYPE_ID }
            }

            fn own_type_id(&self) -> AssetTypeId { Self::asset_type_id() }

            fn own_type_name(&self) -> &'static str { $n }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }
    };
}

/// Structure for tracking asset dependencies.
#[derive(Debug, Default)]
pub struct AssetDeps {
    depends_on: Vec<Handle>,
    used_by: Vec<Handle>,
}

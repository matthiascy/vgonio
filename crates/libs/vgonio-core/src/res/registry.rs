use lazy_static::lazy_static;
use std::{collections::HashMap, sync::Mutex};

lazy_static! {
    static ref ASSET_TYPE_REGISTRY: Mutex<HashMap<u8, &'static str>> = Mutex::new(HashMap::new());
}

/// Asset registry.
pub struct AssetTypeRegistry;

impl AssetTypeRegistry {
    /// Registers a new asset type with the given id.
    pub fn register_asset_type(id: u8, name: &'static str) {
        let mut registry = ASSET_TYPE_REGISTRY.lock().unwrap();
        if let Some(existing) = registry.get(&id) {
            panic!(
                "Asset type id {} already registered for type {}",
                id, existing
            );
        }
        log::info!("Registering asset type {} with id {}", name, id);
        registry.insert(id, name);
    }

    /// Returns the name of the asset type with the given id.
    pub fn asset_type_name(id: u8) -> Option<&'static str> {
        ASSET_TYPE_REGISTRY.lock().unwrap().get(&id).copied()
    }

    /// Returns the id of the asset type with the given name.
    pub fn asset_type_id(name: &str) -> Option<u8> {
        ASSET_TYPE_REGISTRY
            .lock()
            .unwrap()
            .iter()
            .find_map(|(id, n)| if n == &name { Some(*id) } else { None })
    }

    /// Returns whether a type id is registered.
    pub fn is_registered(id: u8) -> bool {
        ASSET_TYPE_REGISTRY
            .lock()
            .unwrap()
            .iter()
            .find(|(i, _)| i == &&id)
            .is_some()
    }
}

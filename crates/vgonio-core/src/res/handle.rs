//! Strongly typed handle referencing loaded assets.
use crate::res::{Asset, AssetTypeId, AssetTypeRegistry};
use std::{
    fmt::{Debug, Display, Formatter},
    hash::Hash,
};
use uuid::Uuid;

/// Generic handle to an asset referencing assets.
///
/// The id follows the UUID Version 4 in RFC 9562 which is a randomly
/// generated UUID then stored as two 64-bit integers to avoid alignment
/// issues.
///
/// According to the RFC, the 128-bit UUID is laid out as follows (bytes):
///
/// `xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx`
///
/// where 4 bits are used for the version number (4 for version 4) which is
/// encoded in the 13th byte of the UUID (position M). The variant is encoded in
/// the upper two or three bits of the 17th byte (10 for variants 1 and 110 for
/// variant 2). For variant 1, there are 122 bits for the randomly
/// generated part, and for variant 2, there are 121 bits.
///
/// To be safe, we only use the first byte of the UUID to store the variant id.
/// This supports up to 256 variants of different asset types.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle(u64, u64);

impl Handle {
    /// Creates a new handle of the given type.
    ///
    /// `T` here is to constrain the handle to only be used with types that
    /// implement the `Asset` trait.
    pub fn new<T: Asset>() -> Self { Self::new_inner(T::asset_type_id().0) }

    /// Creates an invalid handle.
    pub fn invalid() -> Self { Self(0, 0) }

    /// Creates a new handle with the given asset type id.
    pub fn new_with_asset_id(id: AssetTypeId) -> Self {
        if !AssetTypeRegistry::is_registered(id.0) {
            panic!("Asset with id = {id} hasn't been registered!")
        }
        Self::new_inner(id.0)
    }

    /// Creates a new handle with the given asset type id and variant.
    ///
    /// Useful to encode the asset type and variant in the handle. User is
    /// responsible for ensuring that the variant is unique for the asset
    /// type.
    pub fn new_with_variant<T: Asset>(variant: u8) -> Self {
        let type_id = T::asset_type_id().0;
        let mut id = Uuid::new_v4().into_bytes();
        id[0] = type_id;
        id[1] = variant;
        let (a, b) = id.split_at(8);
        Self(
            u64::from_be_bytes(a.try_into().unwrap()),
            u64::from_be_bytes(b.try_into().unwrap()),
        )
    }

    /// Creates a new handle with the given UUID and asset type.
    #[deprecated(note = "Handle should be created when the asset store loads the asset.")]
    pub fn with_id<T: Asset>(id: Uuid) -> Self {
        let bytes = id.into_bytes();
        if bytes[0] != T::asset_type_id().0 {
            panic!("UUID asset type id doesn't match the type of the asset!")
        }
        let (a, b) = bytes.split_at(8);
        Self(
            u64::from_be_bytes(a.try_into().unwrap()),
            u64::from_be_bytes(b.try_into().unwrap()),
        )
    }

    /// Internal function to create a new handle.
    fn new_inner(type_id: u8) -> Self {
        let mut id = Uuid::new_v4().into_bytes();
        id[0] = type_id;
        let (a, b) = id.split_at(8);
        Self(
            u64::from_be_bytes(a.try_into().unwrap()),
            u64::from_be_bytes(b.try_into().unwrap()),
        )
    }

    /// Gets the embedded asset type id from the handle.
    ///
    /// This is the first byte of the UUID stored in the handle.
    pub fn asset_type_id(&self) -> AssetTypeId { AssetTypeId(self.0.to_be_bytes()[0]) }

    /// Gets the embedded asset type name from the handle.
    pub fn asset_type_name(&self) -> &'static str {
        AssetTypeRegistry::asset_type_name(self.asset_type_id().0).unwrap_or("Unknown")
    }

    /// Checks if the handle is of the same type as the given asset type.
    pub fn same_type_id_as<T: Asset>(&self) -> bool { self.asset_type_id() == T::asset_type_id() }

    /// Checks if the handle is of the same type as another handle.
    pub fn eq_type_id(&self, other: &Handle) -> bool {
        self.asset_type_id() == other.asset_type_id()
    }

    /// Converts the handle to a UUID.
    pub fn into_uuid(self) -> Uuid { Uuid::from_u64_pair(self.0, self.1) }

    /// Converts a UUID to a handle.
    ///
    /// This function assumes that the UUID is valid and the asset type id
    /// matches the type of the asset.
    pub fn from_uuid<T: Asset>(uuid: Uuid) -> Self {
        let bytes = uuid.as_bytes();
        if bytes[0] != T::asset_type_id().0 {
            panic!(
                "UUID asset type id #{} doesn't match the type of the asset #{}!",
                bytes[0],
                T::asset_type_id().0
            )
        }
        let (a, b) = bytes.split_at(8);
        Self(
            u64::from_be_bytes(a.try_into().unwrap()),
            u64::from_be_bytes(b.try_into().unwrap()),
        )
    }

    /// Checks if the handle is valid.
    pub fn is_valid(&self) -> bool { self.0 | self.1 != 0 }

    /// Gets the variant of the handle.
    pub fn variant(&self) -> u8 { self.into_uuid().as_bytes()[1] }
}

impl Debug for Handle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Handle<{}>#{}",
            AssetTypeRegistry::asset_type_name(self.asset_type_id().0).unwrap_or("Unknown"),
            self.into_uuid()
        )
    }
}

impl Display for Handle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) }
}

impl Into<Uuid> for Handle {
    fn into(self) -> Uuid { self.into_uuid() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset;

    #[derive(Debug, PartialEq)]
    struct TestAsset;

    asset!(TestAsset, "TestAsset");

    #[test]
    fn test_handle() {
        let _ = env_logger::try_init().ok();

        let handle = Handle::new::<TestAsset>();
        assert_eq!(handle.asset_type_id(), AssetTypeId(0));
        assert_eq!(handle.asset_type_name(), "TestAsset");
        assert!(handle.same_type_id_as::<TestAsset>());
        assert_eq!(
            handle.to_string(),
            format!("Handle<TestAsset>#{}", handle.into_uuid())
        );
    }

    #[test]
    fn test_handle_uuid_conversion() {
        let handle = Handle::new::<TestAsset>();
        let uuid: Uuid = handle.into();
        let handle2: Handle = Handle::from_uuid::<TestAsset>(uuid);
        assert_eq!(handle, handle2);
    }

    #[test]
    fn test_handle_invalid() {
        let invalid_handle = Handle::invalid();
        assert!(!invalid_handle.is_valid());
    }

    #[test]
    fn test_handle_new_with_asset_id() {
        let asset_id = AssetTypeId(0);
        let handle = Handle::new_with_asset_id(asset_id);
        assert_eq!(handle.asset_type_id(), asset_id);
    }

    #[test]
    fn test_handle_new_with_variant() {
        let variant: u8 = 1;
        let handle = Handle::new_with_variant::<TestAsset>(variant);
        assert_eq!(handle.variant(), variant);
    }

    #[test]
    fn test_handle_eq_type_id() {
        let handle1 = Handle::new::<TestAsset>();
        let handle2 = Handle::new::<TestAsset>();
        assert!(handle1.eq_type_id(&handle2));
    }

    #[test]
    fn test_handle_from_uuid_invalid_type() {
        let uuid = Uuid::new_v4();
        let result = std::panic::catch_unwind(|| {
            Handle::from_uuid::<TestAsset>(uuid);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_with_id_invalid_type() {
        let uuid = Uuid::new_v4();
        let result = std::panic::catch_unwind(|| {
            Handle::with_id::<TestAsset>(uuid);
        });
        assert!(result.is_err());
    }
}

//! Strongly typed handle referencing loaded assets.

use crate::utils::Asset;
use std::{
    any::TypeId,
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
};
use uuid::Uuid;

/// Handle referencing loaded assets.
pub struct Handle<T>
where
    T: Asset,
{
    /// Id of the handle.
    pub id: Uuid,
    _phantom: std::marker::PhantomData<fn() -> T>,
}

impl<T: Asset> Clone for Handle<T> {
    fn clone(&self) -> Self { *self }
}

impl<T: Asset> Copy for Handle<T> {}

impl<T: Asset> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}

impl<T: Asset> Eq for Handle<T> {}

impl<T: Asset> Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.id.hash(state) }
}

impl<T: Asset> Debug for Handle<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}<{}>",
            std::any::type_name::<T>().split(&"::").last().unwrap(),
            self.id
        )
    }
}

impl<T: Asset> Display for Handle<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) }
}

impl<T> Handle<T>
where
    T: Asset,
{
    /// Creates a new handle with a random id.
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new handle with type id of `T` and a possibly variant id.
    ///
    /// This is useful for embedding the type information in the handle.
    ///
    /// - The type id starts from the 9th byte up to the 16th byte of the
    ///   handle's id, stored in little endian.
    /// - The variant id is the 8th byte of the handle's id.
    pub fn with_type_id(variant: u8) -> Self {
        let mut id = Uuid::new_v4().into_bytes();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write_u128(unsafe { std::mem::transmute::<TypeId, u128>(TypeId::of::<T>()) });
        let type_id = hasher.finish();
        id[8..].copy_from_slice(&type_id.to_le_bytes());
        id[7] = variant;
        Self::with_id(Uuid::from_bytes(id))
    }

    /// Creates a new handle with the given id.
    pub fn with_id(id: Uuid) -> Self {
        Self {
            id,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns an invalid handle.
    pub fn invalid() -> Self {
        Self {
            id: Uuid::nil(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the id of the handle.
    pub fn id(&self) -> Uuid { self.id }

    /// Returns true if the handle is valid.
    pub fn is_valid(&self) -> bool { !self.id.is_nil() }

    /// Returns if the embedded type id is the same as `T`.
    pub fn same_type_id_as<U: 'static>(&self) -> bool {
        let embedded = u64::from_le_bytes(self.id.as_bytes()[8..].try_into().unwrap());
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write_u128(unsafe { std::mem::transmute::<TypeId, u128>(TypeId::of::<U>()) });
        let type_id_hash = hasher.finish();
        embedded == type_id_hash
    }

    /// Returns the variant id embedded in the handle.
    pub fn variant_id(&self) -> u8 { self.id.as_bytes()[7] }
}

/// Default handle is an invalid handle.
impl<T: Asset> Default for Handle<T> {
    fn default() -> Self { Self::with_id(Uuid::nil()) }
}

impl<T: Asset> From<Uuid> for Handle<T> {
    fn from(id: Uuid) -> Self { Self::with_id(id) }
}

#[test]
fn test_handle_type_id_embedding() {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq)]
    enum OwnType {
        Variant1 = 0,
        Variant2 = 1,
    }
    impl Asset for OwnType {}

    let hdl1 = Handle::<OwnType>::with_type_id(OwnType::Variant1 as u8);
    let hdl2 = Handle::<OwnType>::with_type_id(OwnType::Variant2 as u8);
    assert!(hdl1.same_type_id_as::<OwnType>());
    assert!(hdl2.same_type_id_as::<OwnType>());
    assert_eq!(hdl1.variant_id(), OwnType::Variant1 as u8);
    assert_eq!(hdl2.variant_id(), OwnType::Variant2 as u8);
}

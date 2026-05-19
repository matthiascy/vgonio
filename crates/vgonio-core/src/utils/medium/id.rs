//! `MediumId` -- the runtime handle. A `Copy` wrapper around the canonical
//! medium name as a `&'static str`. Equality and hashing are content-based.

use core::fmt;

/// A medium handle. See module docs.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct MediumId(pub(super) &'static str);

impl MediumId {
    /// Vacuum: a perfect vacuum. n = 1, k = 0 at every λ.
    pub const VACUUM: MediumId = MediumId("vac");

    /// Shipped built-in: Air.
    pub const AIR: MediumId = MediumId("air");
    /// Shipped built-in: Aluminium.
    pub const AL: MediumId = MediumId("al");
    /// Shipped built-in: Copper.
    pub const CU: MediumId = MediumId("cu");
    /// Shipped built-in: Nickel.
    pub const NI: MediumId = MediumId("ni");
    /// Shipped built-in: Polyvinyl chloride.
    pub const PVC: MediumId = MediumId("pvc");
    /// Shipped built-in: Chromium.
    pub const CR: MediumId = MediumId("cr");

    /// The canonical name as &'static str. Stable identity for this medium.
    #[inline]
    pub fn name(self) -> &'static str { self.0 }
}

impl fmt::Debug for MediumId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "MediumId({:?})", self.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consts_have_expected_names() {
        assert_eq!(MediumId::VACUUM.name(), "vac");
        assert_eq!(MediumId::AIR.name(), "air");
        assert_eq!(MediumId::AL.name(), "al");
        assert_eq!(MediumId::CU.name(), "cu");
        assert_eq!(MediumId::NI.name(), "ni");
        assert_eq!(MediumId::PVC.name(), "pvc");
        assert_eq!(MediumId::CR.name(), "cr");
    }

    #[test]
    fn equality_is_content_based() {
        let a = MediumId::AL;
        let b = MediumId("al"); // a `const` value built from a different &'static str literal
        assert_eq!(a, b, "content equality should make these equal");
    }

    #[test]
    fn debug_format_includes_name() {
        assert_eq!(format!("{:?}", MediumId::AIR), r#"MediumId("air")"#);
    }
}

//! Legacy 3-byte BSDF medium codec — layer A (pure mapping).
//!
//! Closed over the 7-symbol set written by pre-0.2.0 vgonio. Used by:
//!   - the in-tree BSDF reader/writer (via layer B, below)
//!   - Plan 2's `cargo x bsdf migrate` xtask (its own byte-for-byte duplicate of this table; will
//!     be verified by a shared round-trip fixture test)

use std::fmt;

#[derive(Debug)]
pub enum LegacySymbolError {
    /// The 3-byte sequence does not match any of the 7 known legacy symbols.
    UnknownSymbol([u8; 3]),
    /// The canonical name is not one of the 7 names that pre-0.2.0 vgonio knew
    /// how to write. Includes user-added names like `au`/`si` even if they
    /// would fit in 3 bytes.
    NotInLegacyTable { name: String },
}

impl fmt::Display for LegacySymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSymbol(b) => write!(
                f,
                "legacy BSDF medium symbol {b:?} is not one of the 7 known pre-0.2.0 symbols"
            ),
            Self::NotInLegacyTable { name } => write!(
                f,
                "medium {name:?} is not in the closed 7-symbol legacy table; BSDF format 0.2.0 \
                 (Plan 2) is required to write this medium",
            ),
        }
    }
}

impl std::error::Error for LegacySymbolError {}

const LEGACY_TABLE: &[(&[u8; 3], &str)] = &[
    (b"vac", "vac"),
    (b"air", "air"),
    (b"al\0", "al"),
    (b"cu\0", "cu"),
    (b"ni\0", "ni"),
    (b"pvc", "pvc"),
    (b"cr\0", "cr"),
];

/// Map a legacy 3-byte BSDF symbol to its canonical medium name. Pure — does
/// not consult the registry.
pub fn canonical_name_for_legacy_symbol(src: &[u8; 3]) -> Option<&'static str> {
    LEGACY_TABLE
        .iter()
        .find(|(s, _)| *s == src)
        .map(|(_, n)| *n)
}

/// Pack a canonical medium name into the legacy 3-byte slot. Errors for any
/// name not in the closed 7-symbol table.
pub fn legacy_symbol_for_canonical_name(
    name: &str,
    dst: &mut [u8; 3],
) -> Result<(), LegacySymbolError> {
    let entry = LEGACY_TABLE
        .iter()
        .find(|(_, n)| *n == name)
        .ok_or_else(|| LegacySymbolError::NotInLegacyTable {
            name: name.to_owned(),
        })?;
    *dst = *entry.0;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_all_seven_symbols() {
        for (sym, name) in LEGACY_TABLE {
            assert_eq!(canonical_name_for_legacy_symbol(sym), Some(*name));
            let mut dst = [0u8; 3];
            legacy_symbol_for_canonical_name(name, &mut dst).unwrap();
            assert_eq!(&dst, *sym);
        }
    }

    #[test]
    fn unknown_symbol_returns_none() {
        assert!(canonical_name_for_legacy_symbol(b"xxx").is_none());
        assert!(canonical_name_for_legacy_symbol(b"au\0").is_none()); // user-added, not in table
    }

    #[test]
    fn user_added_name_errors_on_write() {
        let mut dst = [0u8; 3];
        let err = legacy_symbol_for_canonical_name("au", &mut dst).unwrap_err();
        assert!(matches!(err, LegacySymbolError::NotInLegacyTable { ref name } if name == "au"));
    }
}

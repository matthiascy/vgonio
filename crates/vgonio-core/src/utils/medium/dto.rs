//! On-disk TOML schema for `media.toml` files (all four layers share this).
//! Loaded once at bootstrap; not used on hot paths.

use crate::utils::medium::MediumLoadError;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MediumTomlFile {
    pub schema_version: u32,
    #[serde(default)]
    pub medium: Vec<MediumDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MediumDto {
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// Read + parse a layer file. Returns the deserialized DTO; does NOT validate
/// the contents against the registry rules (that's `validate_dto` below).
pub(super) fn read_file(path: &Path) -> Result<MediumTomlFile, MediumLoadError> {
    let bytes = std::fs::read(path).map_err(|e| MediumLoadError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let text = std::str::from_utf8(&bytes).map_err(|e| MediumLoadError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
    })?;
    parse_str(text, path)
}

/// Parse a TOML string to a `MediumTomlFile`.
pub(super) fn parse_str(text: &str, path: &Path) -> Result<MediumTomlFile, MediumLoadError> {
    toml::from_str::<MediumTomlFile>(text).map_err(|e| MediumLoadError::Parse {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Canonical-name grammar: `^[a-z][a-z0-9_]{0,31}$`.
pub(super) fn is_valid_name(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes.len() > 32 {
        return false;
    }
    let first = bytes[0];
    if !first.is_ascii_lowercase() {
        return false;
    }
    bytes[1..]
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_')
}

/// Alias grammar: `^[\x21-\x7E]{1,32}$` (ASCII printable, no whitespace).
pub(super) fn is_valid_alias(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes.len() > 32 {
        return false;
    }
    bytes.iter().all(|b| (0x21..=0x7E).contains(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn name_grammar_accepts_canonical_built_ins() {
        for n in ["air", "al", "cu", "ni", "pvc", "cr"] {
            assert!(is_valid_name(n), "{n} should pass");
        }
    }

    #[test]
    fn name_grammar_rejects_bad_inputs() {
        for n in [
            "",
            "Al",
            "AL",
            "1al",
            "_al",
            "al-x",
            "al x",
            "alβ",
            &"x".repeat(33),
        ] {
            assert!(!is_valid_name(n), "{n:?} should fail");
        }
    }

    #[test]
    fn alias_grammar_accepts_mixed_case_and_digits() {
        for n in ["Al", "AL", "SiO2", "Cr2O3", "PVC"] {
            assert!(is_valid_alias(n), "{n} should pass");
        }
    }

    #[test]
    fn alias_grammar_rejects_whitespace_and_non_ascii() {
        for n in ["", "Al ", "Al x", "alβ", "α-Fe", &"x".repeat(33)] {
            assert!(!is_valid_alias(n), "{n:?} should fail");
        }
    }

    #[test]
    fn parse_minimal_valid_file() {
        let text = r#"
schema_version = 1

[[medium]]
name = "al"
display_name = "Aluminium"
aliases = ["aluminium", "Al"]
"#;
        let file = parse_str(text, &PathBuf::from("<test>")).unwrap();
        assert_eq!(file.schema_version, 1);
        assert_eq!(file.medium.len(), 1);
        assert_eq!(file.medium[0].name, "al");
        assert_eq!(file.medium[0].aliases, vec!["aluminium", "Al"]);
    }

    #[test]
    fn parse_rejects_unknown_field() {
        let text = r#"
schema_version = 1

[[medium]]
name = "al"
display_name = "Aluminium"
bogus = "not a field"
"#;
        let result = parse_str(text, &PathBuf::from("<test>"));
        assert!(matches!(result, Err(MediumLoadError::Parse { .. })));
    }
}

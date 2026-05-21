//! In-memory medium registry. Built once at startup from layered TOML inputs.

use crate::utils::medium::{
    dto::{self, MediumDto, MediumTomlFile},
    error::MediumLoadError,
    intern::intern,
    MediumId, Provenance,
};
use std::{collections::HashMap, path::Path};

/// The compile-time embedded fallback (`builtin.toml`).
pub(super) const BUILTIN_TOML: &str = include_str!("builtin.toml");

/// A single medium entry in the registry.
#[derive(Debug, Clone)]
pub struct MediumEntry {
    /// Canonical medium handle.
    pub id: MediumId,
    /// Human-readable name of the medium (e.g. "Aluminium").
    pub display_name: &'static str,
    /// Alternative names that resolve to the same medium (e.g. "al" for "Aluminium").
    pub aliases: &'static [&'static str],
    /// Where this entry came from.
    pub source: Provenance,
}

/// The frozen registry of media.
#[derive(Debug)]
pub struct MediumRegistry {
    by_id: HashMap<&'static str, MediumEntry>,
    by_alias: HashMap<&'static str, &'static str>,
}

impl MediumRegistry {
    /// Build a registry from the embedded fallback + (optional) system, user files.
    /// Vacuum is seeded as a synthetic built-in before any layer is processed.
    pub fn build(sys: Option<&Path>, user: Option<&Path>) -> Result<Self, MediumLoadError> {
        let mut reg = Self::empty();
        Self::seed_vacuum(&mut reg);
        Self::merge_str(
            &mut reg,
            BUILTIN_TOML,
            Path::new("<buildin.toml>"),
            Provenance::Builtin,
        )?;

        if let Some(p) = sys {
            Self::merge_file_optional(&mut reg, p, Provenance::System)?;
        }

        if let Some(p) = user {
            Self::merge_file_optional(&mut reg, p, Provenance::User)?;
        }

        Ok(reg)
    }

    pub(super) fn empty() -> Self {
        Self {
            by_id: HashMap::new(),
            by_alias: HashMap::new(),
        }
    }

    /// Look up an entry by its canonical name.
    pub fn by_name(&self, name: &str) -> Option<&MediumEntry> {
        if let Some(e) = self.by_id.get(name) {
            return Some(e);
        }
        let canonical = self.by_alias.get(name)?;
        self.by_id.get(canonical)
    }

    /// Iterate all entries (insertion order is not guaranteed; for stable UI
    /// ordering, callers should sort by `id.name()` or `display_name`).
    pub fn iter(&self) -> impl Iterator<Item = &MediumEntry> { self.by_id.values() }

    /// Number of entries (including the synthetic Vacuum).
    pub fn len(&self) -> usize { self.by_id.len() }

    /// Insert a new entry. Caller is responsible for collision checks; this
    /// method assumes the entry is valid against the cumulative state.
    pub(super) fn insert(&mut self, entry: MediumEntry) {
        for alias in entry.aliases {
            self.by_alias.insert(alias, entry.id.name());
        }
        self.by_id.insert(entry.id.name(), entry);
    }

    fn seed_vacuum(reg: &mut Self) {
        reg.insert(MediumEntry {
            id: MediumId::VACUUM,
            display_name: "Vacuum",
            aliases: &["vacuum"],
            source: Provenance::Builtin,
        });
    }

    fn merge_file_optional(
        reg: &mut Self,
        path: &Path,
        source: Provenance,
    ) -> Result<(), MediumLoadError> {
        if !path.exists() {
            return Ok(()); // missing layer is OK
        }

        let file = dto::read_file(path)?;
        Self::validate_and_merge(reg, file, path, source)
    }

    fn merge_str(
        reg: &mut Self,
        text: &str,
        path: &Path,
        source: Provenance,
    ) -> Result<(), MediumLoadError> {
        let file = dto::parse_str(text, path)?;
        Self::validate_and_merge(reg, file, path, source)
    }

    fn validate_and_merge(
        reg: &mut Self,
        file: MediumTomlFile,
        path: &Path,
        source: Provenance,
    ) -> Result<(), MediumLoadError> {
        if file.schema_version != 1 {
            return Err(MediumLoadError::UnsupportedSchemaVersion {
                found: file.schema_version,
            });
        }
        // Per-file: within-file duplicate detection. Owned `String` keys so
        // the borrow doesn't conflict with the subsequent `for m in file.medium`
        // which consumes `file.medium`.
        let mut seen_names: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (idx, m) in file.medium.iter().enumerate() {
            if !dto::is_valid_name(&m.name) {
                return Err(MediumLoadError::InvalidName {
                    name: m.name.clone(),
                });
            }
            if m.name == "vac" {
                return Err(MediumLoadError::ReservedName {
                    name: m.name.clone(),
                    path: path.to_path_buf(),
                });
            }
            for alias in &m.aliases {
                if !dto::is_valid_alias(alias) {
                    return Err(MediumLoadError::InvalidAlias {
                        alias: alias.clone(),
                        entry: m.name.clone(),
                        path: path.to_path_buf(),
                    });
                }
                if alias == "vac" {
                    return Err(MediumLoadError::ReservedName {
                        name: alias.clone(),
                        path: path.to_path_buf(),
                    });
                }
            }
            if let Some(&first) = seen_names.get(&m.name) {
                return Err(MediumLoadError::DuplicateName {
                    name: m.name.clone(),
                    path: path.to_path_buf(),
                    first,
                    second: idx,
                });
            }
            seen_names.insert(m.name.clone(), idx);
        }
        drop(seen_names); // explicit — readers see the borrow story clearly
                          // Cumulative: cross-layer collisions.
        for m in file.medium {
            Self::check_and_insert(reg, m, path, source)?;
        }
        Ok(())
    }

    fn check_and_insert(
        reg: &mut Self,
        m: MediumDto,
        path: &Path,
        source: Provenance,
    ) -> Result<(), MediumLoadError> {
        // Layer-collision: does this name (or any alias) already exist?
        if let Some(existing) = reg.by_id.get(m.name.as_str()) {
            return Err(MediumLoadError::LayerCollision {
                name: m.name.clone(),
                path: path.to_path_buf(),
                earlier_layer: existing.source.label(),
            });
        }
        if let Some(canon) = reg.by_alias.get(m.name.as_str()) {
            let other_source = reg.by_id[canon].source.label();
            return Err(MediumLoadError::AliasCollision {
                alias: m.name.clone(),
                entry: m.name.clone(),
                path: path.to_path_buf(),
                kind: "alias",
                other: canon.to_string(),
                other_source,
            });
        }
        for alias in &m.aliases {
            if let Some(existing) = reg.by_id.get(alias.as_str()) {
                return Err(MediumLoadError::AliasCollision {
                    alias: alias.clone(),
                    entry: m.name.clone(),
                    path: path.to_path_buf(),
                    kind: "name",
                    other: existing.id.name().to_string(),
                    other_source: existing.source.label(),
                });
            }
            if let Some(canon) = reg.by_alias.get(alias.as_str()) {
                let other_source = reg.by_id[canon].source.label();
                return Err(MediumLoadError::AliasCollision {
                    alias: alias.clone(),
                    entry: m.name.clone(),
                    path: path.to_path_buf(),
                    kind: "alias",
                    other: canon.to_string(),
                    other_source,
                });
            }
        }
        // All checks passed; intern strings and insert.
        let name_static: &'static str = intern(&m.name);
        let display_static: &'static str = intern(&m.display_name);
        let aliases_static: &'static [&'static str] = {
            let v: Vec<&'static str> = m.aliases.iter().map(|a| intern(a)).collect();
            Box::leak(v.into_boxed_slice())
        };
        reg.insert(MediumEntry {
            id: MediumId(name_static),
            display_name: display_static,
            aliases: aliases_static,
            source,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_has_no_entries() {
        let reg = MediumRegistry::empty();
        assert_eq!(reg.len(), 0);
        assert!(reg.by_name("al").is_none());
    }

    #[test]
    fn insert_and_lookup_by_canonical_name() {
        let mut reg = MediumRegistry::empty();
        reg.insert(MediumEntry {
            id: MediumId::AL,
            display_name: "Aluminium",
            aliases: &["aluminium", "Al"],
            source: Provenance::Builtin,
        });
        assert_eq!(reg.by_name("al").unwrap().display_name, "Aluminium");
    }

    #[test]
    fn lookup_resolves_via_alias() {
        let mut reg = MediumRegistry::empty();
        reg.insert(MediumEntry {
            id: MediumId::AL,
            display_name: "Aluminium",
            aliases: &["aluminium", "Al"],
            source: Provenance::Builtin,
        });
        assert_eq!(reg.by_name("Al").unwrap().id, MediumId::AL);
        assert_eq!(reg.by_name("aluminium").unwrap().id, MediumId::AL);
    }
}

#[cfg(test)]
mod embedded_tests {
    use super::*;
    use crate::utils::medium::dto;

    #[test]
    fn builtin_toml_parses_and_has_six_entries() {
        let file = dto::parse_str(BUILTIN_TOML, std::path::Path::new("<builtin.toml>")).unwrap();
        assert_eq!(file.schema_version, 1);
        assert_eq!(file.medium.len(), 6);
        let names: Vec<_> = file.medium.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["air", "al", "cu", "ni", "pvc", "cr"]);
    }

    #[test]
    fn builtin_toml_does_not_list_vac() {
        let file = dto::parse_str(BUILTIN_TOML, std::path::Path::new("<builtin.toml>")).unwrap();
        for m in &file.medium {
            assert_ne!(m.name, "vac");
            assert!(!m.aliases.iter().any(|a| a == "vac"));
        }
    }
}

#[cfg(test)]
mod build_tests {
    use super::*;

    #[test]
    fn build_with_no_layers_yields_seven_entries() {
        let reg = MediumRegistry::build(None, None).unwrap();
        assert_eq!(reg.len(), 7); // 6 built-ins + vacuum
        assert_eq!(reg.by_name("vac").unwrap().id, MediumId::VACUUM);
        assert_eq!(reg.by_name("al").unwrap().display_name, "Aluminium");
        assert_eq!(reg.by_name("aluminium").unwrap().id, MediumId::AL);
        assert_eq!(reg.by_name("chrome").unwrap().id, MediumId::CR);
    }

    #[test]
    fn reserved_vac_in_layer_errors() {
        let bad = r#"
schema_version = 1
[[medium]]
name = "vac"
display_name = "Bogus"
"#;
        let mut reg = MediumRegistry::empty();
        MediumRegistry::seed_vacuum(&mut reg);
        let err = MediumRegistry::merge_str(&mut reg, bad, Path::new("<bad>"), Provenance::User)
            .unwrap_err();
        assert!(matches!(err, MediumLoadError::ReservedName { .. }));
    }

    #[test]
    fn invalid_name_errors() {
        let bad = r#"
schema_version = 1
[[medium]]
name = "Al"
display_name = "x"
"#;
        let err = MediumRegistry::merge_str(
            &mut MediumRegistry::empty(),
            bad,
            Path::new("<bad>"),
            Provenance::User,
        )
        .unwrap_err();
        assert!(matches!(err, MediumLoadError::InvalidName { .. }));
    }

    #[test]
    fn schema_version_mismatch_errors() {
        let bad = "schema_version = 999\n";
        let err = MediumRegistry::merge_str(
            &mut MediumRegistry::empty(),
            bad,
            Path::new("<bad>"),
            Provenance::User,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MediumLoadError::UnsupportedSchemaVersion { found: 999 }
        ));
    }

    #[test]
    fn user_layer_redefining_builtin_errors() {
        let bad = r#"
schema_version = 1
[[medium]]
name = "al"
display_name = "Custom Al"
"#;
        let mut reg = MediumRegistry::build(None, None).unwrap();
        let err = MediumRegistry::merge_str(&mut reg, bad, Path::new("<user>"), Provenance::User)
            .unwrap_err();
        assert!(matches!(err, MediumLoadError::LayerCollision { .. }));
    }

    #[test]
    fn user_layer_adding_new_name_succeeds() {
        let ok = r#"
schema_version = 1
[[medium]]
name = "au"
display_name = "Gold"
aliases = ["gold", "Au"]
"#;
        let mut reg = MediumRegistry::build(None, None).unwrap();
        MediumRegistry::merge_str(&mut reg, ok, Path::new("<user>"), Provenance::User).unwrap();
        assert_eq!(reg.by_name("au").unwrap().display_name, "Gold");
        assert_eq!(reg.by_name("Au").unwrap().id.name(), "au");
    }
}

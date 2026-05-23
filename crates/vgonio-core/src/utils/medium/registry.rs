//! In-memory medium registry. Built once at startup from layered TOML inputs.

use ron::de;

use crate::utils::medium::{
    dto::{self, MediumDto, MediumTomlFile},
    error::MediumLoadError,
    intern::intern,
    merge_layers, MediumId, MergePolicy, Provenance,
};
use std::{collections::HashMap, path::Path};

/// One medium parsed from a layer file, before string interning.
#[derive(Debug, Clone)]
struct ParsedMedium {
    display_name: String,
    aliases: Vec<String>,
}

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
    pub provenance: Provenance,
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
        let builtin = Self::validate_layer(
            dto::parse_str(BUILTIN_TOML, Path::new("<builtin.toml>"))?,
            Path::new("<builtin.toml>"),
        )?;
        let system = Self::read_optional_layer(sys)?;
        let user = Self::read_optional_layer(user)?;

        let merged = merge_layers(
            [
                (Provenance::Builtin, builtin),
                (Provenance::System, system),
                (Provenance::User, user),
            ],
            MergePolicy::AdditiveOnly,
        )
        .map_err(|c| MediumLoadError::LayerCollision {
            name: c.key,
            earlier: c.earlier,
            later: c.later,
        })?;

        let mut reg = Self::empty();
        Self::seed_vacuum(&mut reg);
        Self::insert_merged(&mut reg, merged)?;

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
            provenance: Provenance::Builtin,
        });
    }

    /// Parse + validate an optional layer file. A `None` path, or a path that
    /// does not exist, yields an empty layer.
    fn read_optional_layer(
        path: Option<&Path>,
    ) -> Result<HashMap<String, ParsedMedium>, MediumLoadError> {
        match path {
            Some(p) if p.exists() => Self::validate_layer(dto::read_file(p)?, p),
            _ => Ok(HashMap::new()),
        }
    }

    /// Parse-time validation of one layer: schema version, the canonical-name
    /// and alias grammars, the `vac` reserved name, and within-layer duplicate
    /// names. Returns the layer as a `canonical-name -> ParsedMedium` map.
    fn validate_layer(
        file: MediumTomlFile,
        path: &Path,
    ) -> Result<HashMap<String, ParsedMedium>, MediumLoadError> {
        if file.schema_version != 1 {
            return Err(MediumLoadError::UnsupportedSchemaVersion {
                found: file.schema_version,
            });
        }
        let mut map: HashMap<String, ParsedMedium> = HashMap::new();
        let mut order: HashMap<String, usize> = HashMap::new();
        for (idx, m) in file.medium.into_iter().enumerate() {
            if !dto::is_valid_name(&m.name) {
                return Err(MediumLoadError::InvalidName { name: m.name });
            }
            if m.name == "vac" {
                return Err(MediumLoadError::ReservedName {
                    name: m.name,
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
            if let Some(&first) = order.get(&m.name) {
                return Err(MediumLoadError::DuplicateName {
                    name: m.name,
                    path: path.to_path_buf(),
                    first,
                    second: idx,
                });
            }
            order.insert(m.name.clone(), idx);
            map.insert(
                m.name,
                ParsedMedium {
                    display_name: m.display_name,
                    aliases: m.aliases,
                },
            );
        }
        Ok(map)
    }

    /// Validate the alias graph over `merged` (Vacuum is already seeded into
    /// `reg`), then intern strings and insert every entry. Canonical names are
    /// already unique — `merge_layers` guaranteed it — so only alias clashes
    /// can occur here.
    fn insert_merged(
        reg: &mut Self,
        merged: HashMap<String, (ParsedMedium, Provenance)>,
    ) -> Result<(), MediumLoadError> {
        // Deterministic order so a collision is always reported the same way.
        let mut entries: Vec<_> = merged.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        for (name, (parsed, provenance)) in entries {
            // The canonical name must not already exist as an alias (Vacuum's
            // "vacuum", or an alias of an earlier-inserted entry).
            if let Some(canon) = reg.by_alias.get(name.as_str()) {
                return Err(MediumLoadError::AliasCollision {
                    alias: name.clone(),
                    entry: name.clone(),
                    kind: "alias",
                    other: (*canon).to_string(),
                    other_source: reg.by_id[canon].provenance.label(),
                });
            }
            // Each alias must not clash with a canonical name or another alias.
            for alias in &parsed.aliases {
                if let Some(existing) = reg.by_id.get(alias.as_str()) {
                    return Err(MediumLoadError::AliasCollision {
                        alias: alias.clone(),
                        entry: name.clone(),
                        kind: "name",
                        other: existing.id.name().to_string(),
                        other_source: existing.provenance.label(),
                    });
                }
                if let Some(canon) = reg.by_alias.get(alias.as_str()) {
                    return Err(MediumLoadError::AliasCollision {
                        alias: alias.clone(),
                        entry: name.clone(),
                        kind: "alias",
                        other: (*canon).to_string(),
                        other_source: reg.by_id[canon].provenance.label(),
                    });
                }
            }
            let name_static: &'static str = intern(&name);
            let display_static: &'static str = intern(&parsed.display_name);
            // `Box::leak` is bounded: production builds invoke `build` exactly once
            // via the `OnceLock` in `medium::bootstrap`, so the leaked alias slice
            // lives for the program's lifetime by design. Direct calls to `build`
            // from tests do leak per call; the leak is small and bounded by test
            // count. If a future caller needs to rebuild the registry repeatedly
            // (hot reload, multi-tenant), this needs to switch to an arena.
            let aliases_static: &'static [&'static str] = {
                let v: Vec<&'static str> = parsed.aliases.iter().map(|a| intern(a)).collect();
                Box::leak(v.into_boxed_slice())
            };
            reg.insert(MediumEntry {
                id: MediumId(name_static),
                display_name: display_static,
                aliases: aliases_static,
                provenance,
            });
        }
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
            provenance: Provenance::Builtin,
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
            provenance: Provenance::Builtin,
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
        let err = MediumRegistry::validate_layer(
            dto::parse_str(bad, Path::new("<bad>")).unwrap(),
            Path::new("<bad>"),
        )
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
        let err = MediumRegistry::validate_layer(
            dto::parse_str(bad, Path::new("<bad>")).unwrap(),
            Path::new("<bad>"),
        )
        .unwrap_err();
        assert!(matches!(err, MediumLoadError::InvalidName { .. }));
    }

    #[test]
    fn schema_version_mismatch_errors() {
        let err = MediumRegistry::validate_layer(
            dto::parse_str("schema_version = 999\n", Path::new("<bad>")).unwrap(),
            Path::new("<bad>"),
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
        let dir = std::env::temp_dir().join(format!("vgn-medium-redef-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let user = dir.join("media.toml");
        std::fs::write(&user, bad).unwrap();
        let err = MediumRegistry::build(None, Some(&user)).unwrap_err();
        assert!(matches!(err, MediumLoadError::LayerCollision { .. }));
        std::fs::remove_dir_all(&dir).ok();
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
        let dir = std::env::temp_dir().join(format!("vgn-medium-add-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let user = dir.join("media.toml");
        std::fs::write(&user, ok).unwrap();
        let reg = MediumRegistry::build(None, Some(&user)).unwrap();
        assert_eq!(reg.by_name("au").unwrap().display_name, "Gold");
        assert_eq!(reg.by_name("Au").unwrap().id.name(), "au");
        assert_eq!(reg.by_name("au").unwrap().provenance, Provenance::User);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn build_records_builtin_provenance() {
        let reg = MediumRegistry::build(None, None).unwrap();
        assert_eq!(reg.by_name("al").unwrap().provenance, Provenance::Builtin);
        assert_eq!(reg.by_name("vac").unwrap().provenance, Provenance::Builtin);
    }
}

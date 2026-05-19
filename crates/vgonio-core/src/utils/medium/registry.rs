//! In-memory medium registry. Built once at startup from layered TOML inputs.

use crate::utils::medium::MediumId;
use std::collections::HashMap;

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
    pub source: MediumSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediumSource {
    Builtin,
    Embedded,
    System,
    User,
}

impl MediumSource {
    pub fn label(self) -> &'static str {
        match self {
            MediumSource::Builtin => "builtin",
            MediumSource::Embedded => "embedded",
            MediumSource::System => "system",
            MediumSource::User => "user",
        }
    }
}

/// The frozen registry of media.
#[derive(Debug)]
pub struct MediumRegistry {
    by_id: HashMap<&'static str, MediumEntry>,
    by_alias: HashMap<&'static str, &'static str>,
}

impl MediumRegistry {
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
            source: MediumSource::Embedded,
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
            source: MediumSource::Embedded,
        });
        assert_eq!(reg.by_name("Al").unwrap().id, MediumId::AL);
        assert_eq!(reg.by_name("aluminium").unwrap().id, MediumId::AL);
    }
}

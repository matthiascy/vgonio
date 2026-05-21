//! Layered-registry support shared by the medium identity registry and every
//! per-medium *facet* (IOR ...). Owns layer provenance and the cross-layer
//! merge logic.
use std::{collections::HashMap, fmt, hash::Hash};

/// Which layer a registry entry came from.
///
/// Ordered low-to-high precedence: Builtin < System < User. A later layer either
/// adds to or overrides an earlier one, depending on the merge policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Compiled into the binary, the embedded baseline and synthetic entries.
    Builtin,
    /// From the system data directory.
    System,
    /// From the user data directory.
    User,
}

impl Provenance {
    /// Lower-case label, used in log lines and error messages.
    pub fn label(&self) -> &'static str {
        match self {
            Provenance::Builtin => "builtin",
            Provenance::System => "system",
            Provenance::User => "user",
        }
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.label()) }
}

/// How [`merge_layers`] resolves a key present in more than one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MergePolicy {
    /// A later layer with a duplicate key causes a [`Collision`] error.
    AdditiveOnly,
    /// A later layer with a duplicate key silently overrides an earlier one.
    LastWins,
}

/// A key supplied by two layers under `MergePolicy::AdditiveOnly`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collision<K> {
    /// The key provided by both layers.
    pub key: K,
    /// The layer that first provided it.
    pub earlier: Provenance,
    /// The layer that redefined it.
    pub later: Provenance,
}

/// Merge per-layer maps into one, tagging every surviving entry with the layer
/// it came from.
///
/// `layers` must be supplied in ascending-precedence order -- `Builtin`, then
/// `System`, then `User` -- each item being that layer's already resolved
/// `key -> value` map. Within-layer duplicate detection is the caller's
/// responsibility: a `HashMap` cannot represent a duplicate key.
///
/// Under [`MergePolicy::LastWins`] this never returns `Err`.
pub fn merge_layers<K, V>(
    layers: impl IntoIterator<Item = (Provenance, HashMap<K, V>)>,
    policy: MergePolicy,
) -> Result<HashMap<K, (V, Provenance)>, Collision<K>>
where
    K: Eq + Hash,
{
    let mut merged = HashMap::new();
    for (prov, layer) in layers {
        for (k, v) in layer {
            match merged.get(&k) {
                Some(&(_, earlier)) => match policy {
                    MergePolicy::AdditiveOnly => {
                        return Err(Collision {
                            key: k,
                            earlier,
                            later: prov,
                        });
                    },
                    MergePolicy::LastWins => {
                        merged.insert(k, (v, prov));
                    },
                },
                None => {
                    merged.insert(k, (v, prov));
                },
            }
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashMap, hash::Hash};

    fn layer(prov: Provenance, kvs: &[(&str, i32)]) -> (Provenance, HashMap<String, i32>) {
        (prov, kvs.iter().map(|(k, v)| (k.to_string(), *v)).collect())
    }

    #[test]
    fn additive_disjoint_layers_merge() {
        let merged = merge_layers(
            [
                layer(Provenance::Builtin, &[("a", 1)]),
                layer(Provenance::User, &[("b", 2)]),
            ],
            MergePolicy::AdditiveOnly,
        )
        .unwrap();
        assert_eq!(merged.get("a"), Some(&(1, Provenance::Builtin)));
        assert_eq!(merged.get("b"), Some(&(2, Provenance::User)));
    }

    #[test]
    fn additive_redefinition_is_a_collision() {
        let err = merge_layers(
            [
                layer(Provenance::Builtin, &[("a", 1)]),
                layer(Provenance::User, &[("a", 9)]),
            ],
            MergePolicy::AdditiveOnly,
        )
        .unwrap_err();
        assert_eq!(
            err,
            Collision {
                key: "a".to_string(),
                earlier: Provenance::Builtin,
                later: Provenance::User,
            }
        );
    }

    #[test]
    fn last_wins_lets_later_layers_override() {
        let merged = merge_layers(
            [
                layer(Provenance::Builtin, &[("a", 1)]),
                layer(Provenance::System, &[("a", 2)]),
                layer(Provenance::User, &[("a", 3)]),
            ],
            MergePolicy::LastWins,
        )
        .unwrap();
        assert_eq!(merged.get("a"), Some(&(3, Provenance::User)));
    }

    #[test]
    fn label_and_display_agree() {
        assert_eq!(Provenance::Builtin.label(), "builtin");
        assert_eq!(Provenance::System.label(), "system");
        assert_eq!(Provenance::User.label(), "user");
        assert_eq!(format!("{}", Provenance::User), "user");
    }
}

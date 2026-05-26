//! Errors produced by the medium registry loader.

use crate::utils::medium::Provenance;
use std::path::PathBuf;
use thiserror::Error;

/// Errors returned while building or installing a [`MediumRegistry`].
///
/// [`MediumRegistry`]: crate::utils::medium::MediumRegistry
#[derive(Debug, Error)]
pub enum MediumLoadError {
    /// The process-wide registry has already been installed by a previous
    /// `bootstrap` call.
    #[error("medium registry is already initialized")]
    AlreadyInitialized,

    /// Failed to read a layer file from disk.
    #[error("failed to read {path}: {source}")]
    Io {
        /// Path of the file that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse a layer file as TOML.
    #[error("failed to parse {path}: {source}")]
    Parse {
        /// Path of the file that could not be parsed.
        path: PathBuf,
        /// Underlying TOML deserialization error.
        #[source]
        source: toml::de::Error,
    },

    /// A layer file declares a `schema_version` this build does not understand.
    #[error("unsupported media.toml schema_version: {found} (this build understands 1)")]
    UnsupportedSchemaVersion {
        /// `schema_version` field as found in the file.
        found: u32,
    },

    /// A medium name does not match the canonical-name grammar.
    #[error(
        "medium name {name:?} violates the canonical-name grammar [a-z][a-z0-9_]{{0,31}}; rename \
         it or use --alias"
    )]
    InvalidName {
        /// Offending medium name.
        name: String,
    },

    /// An alias does not match the alias grammar.
    #[error(
        "alias {alias:?} in entry {entry:?} ({path}) violates the alias grammar (ASCII printable, \
         no whitespace, 1-32 chars)"
    )]
    InvalidAlias {
        /// Offending alias string.
        alias: String,
        /// Canonical name of the entry that declared the alias.
        entry: String,
        /// Path of the file the entry was loaded from.
        path: PathBuf,
    },

    /// Two entries in the same layer file share the same canonical name.
    #[error("duplicate medium name {name:?} in {path} (entries #{first} and #{second})")]
    DuplicateName {
        /// Canonical name that appears twice.
        name: String,
        /// Path of the file containing the duplicate.
        path: PathBuf,
        /// Index of the first occurrence.
        first: usize,
        /// Index of the second occurrence.
        second: usize,
    },

    /// A layer file uses a name reserved for built-in media (e.g. `vacuum`).
    #[error(
        "medium name {name:?} is reserved for the built-in Vacuum and cannot appear in {path}"
    )]
    ReservedName {
        /// Reserved name that was used.
        name: String,
        /// Path of the file containing the reserved name.
        path: PathBuf,
    },

    /// An alias collides with an existing canonical name or alias in another
    /// entry.
    #[error(
        "alias {alias:?} in entry {entry:?} collides with {kind} {other:?} (from the \
         {other_source} layer)"
    )]
    AliasCollision {
        /// Alias being introduced.
        alias: String,
        /// Canonical name of the entry that declared the alias.
        entry: String,
        /// Kind of the colliding identifier: `"name"` or `"alias"`.
        kind: &'static str, // "name" | "alias"
        /// The existing identifier the alias collides with.
        other: String,
        /// [`Provenance`] label of the entry that owns `other`.
        other_source: &'static str, // the Provenance label of the existing entry
    },

    /// A later layer redefines a medium already provided by an earlier layer.
    /// Layers are additive — they may extend but not override prior layers.
    #[error(
        "medium {name:?} from the {later} layer redefines one already provided by the {earlier} \
         layer; layers may only add new media"
    )]
    LayerCollision {
        /// Canonical name being redefined.
        name: String,
        /// Layer that originally defined the medium.
        earlier: Provenance,
        /// Layer attempting to redefine it.
        later: Provenance,
    },
}

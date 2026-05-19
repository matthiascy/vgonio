//! Errors produced by the medium registry loader.

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MediumLoadError {
    #[error("medium registry is already initialized")]
    AlreadyInitialized,

    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("unsupported media.toml schema_version: {found} (this build understands 1)")]
    UnsupportedSchemaVersion { found: u32 },

    #[error(
        "medium name {name:?} violates the canonical-name grammar [a-z][a-z0-9_]{{0,31}}; rename \
         it or use --alias"
    )]
    InvalidName { name: String },

    #[error(
        "alias {alias:?} in entry {entry:?} ({path}) violates the alias grammar (ASCII printable, \
         no whitespace, 1-32 chars)"
    )]
    InvalidAlias {
        alias: String,
        entry: String,
        path: PathBuf,
    },

    #[error("duplicate medium name {name:?} in {path} (entries #{first} and #{second})")]
    DuplicateName {
        name: String,
        path: PathBuf,
        first: usize,
        second: usize,
    },

    #[error(
        "medium name {name:?} is reserved for the built-in Vacuum and cannot appear in {path}"
    )]
    ReservedName { name: String, path: PathBuf },

    #[error(
        "alias {alias:?} in entry {entry:?} ({path}) collides with {kind} {other:?} (from the \
         {other_source} layer)"
    )]
    AliasCollision {
        alias: String,
        entry: String,
        path: PathBuf,
        kind: &'static str,         // "name" | "alias"
        other: String,              // the canonical name on the other side
        other_source: &'static str, // the MediumSource label of the existing entry
    },

    #[error(
        "media.toml entry {name:?} ({path}) redefines a medium from {earlier_layer}; layers may \
         only add new media"
    )]
    LayerCollision {
        name: String,
        path: PathBuf,
        earlier_layer: &'static str,
    },
}

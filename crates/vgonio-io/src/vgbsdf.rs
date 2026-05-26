//! `.vgbsdf` archival container - a JAR-style zip wrapping per-bounce-level OpenEXR
//! files plus structured TOML metadata.
//!
//! # Why a container
//!
//! Vgonio's measurement pipeline produces hemisphere-carried scalar/spectral data:
//! a BRDF is a function of (incident dir, outgoing dir, λ); an NDF is a function
//! of normal direction; an SDF is a histogram on normal directions; a heightfield
//! is a 2D field with elevation. Historically each of these wrote bespoke `.vgmo`
//! / `.vgms` cache files plus ad-hoc `.exr` exports with no documented conventions.
//! Cache files stay (see [`vgn_core::io::CompressionScheme`] - LZ4 lives there);
//! this module is the **conventions-governed archival** layer used as the
//! long-term interchange artifact for downstream tooling and 10-year readers.
//!
//! # What's inside one archive
//!
//! ```text
//! my_measurement.vgbsdf            (zip, Store method, no container compression)
//! ├── manifest.toml                (top-level metadata + provenance)
//! ├── partition.toml               (SphericalPartition descriptor, lossless)
//! ├── incident_grid.toml           (BSDF only; per-(θᵢ,φᵢ) measurement order)
//! ├── spectrum.toml                (wavelengths in nm, or "scalar")
//! └── l0/                          (single-bounce; siblings: l1/, l1+/)
//!     ├── disc.exr                 (omitted for SDF)
//!     ├── thetaphi.exr
//!     └── patches.exr
//! ```
//!
//! Per-bounce-level subdirectories (`l0/`, `l1/`, `l1+/`) live under a single
//! container (*not* one container per level), so a multi-bounce measurement
//! ships as one file with one manifest.
//!
//! # Naming convention
//!
//! All measurement archive kinds use the same container shape; only the file
//! extension and the manifest's `output_kind` differ. See the
//! `EXTENSION_*` constants below: `.vgbsdf` for BSDF, `.vgndf` for NDF,
//! `.vgmsf` for MSF, `.vgsdf` for SDF, `.vgsurf` for heightfields. The
//! capability that produced each archive maps one-to-one to a
//! `CapabilityId` in `vgonio-job-api` (see that crate's docs for the
//! `measure-bsdf` / `measure-ndf` / `measure-msf` / `measure-sdf` table).
//!
//! # Conventions governing the EXR payloads
//!
//! Every `.exr` inside the container follows the **VGONIO EXR Conventions v1**
//! (see [`conventions`] and `design/INTERNAL_FORMATS_REDESIGN_SPEC.md` Part 3):
//! standardized layer/channel names, partition-aware attributes, three
//! projection encodings (disc / θφ / patches) of the same underlying patch data.
//!
//! Container compression is always `Store` (ZIP method 0); the EXR members
//! already use their own internal compression, so a second pass would just
//! cost CPU.

pub mod conventions;
pub mod incident_grid;
pub mod manifest;
pub mod partition_toml;
pub mod reader;
pub mod spectrum;
pub mod writer;

pub use conventions::{ConventionsVersion, OutgoingEncoding, OutputKind};
pub use incident_grid::IncidentGridToml;
pub use manifest::Manifest;
pub use partition_toml::PartitionToml;
pub use reader::VgbsdfReader;
pub use spectrum::SpectrumToml;
pub use writer::VgbsdfWriter;

/// File extension for vgonio BSDF archives (without leading dot).
pub const EXTENSION_BSDF: &str = "vgbsdf";
/// File extension for vgonio NDF archives.
pub const EXTENSION_NDF: &str = "vgndf";
/// File extension for vgonio MSF archives.
pub const EXTENSION_MSF: &str = "vgmsf";
/// File extension for vgonio SDF archives.
pub const EXTENSION_SDF: &str = "vgsdf";
/// File extension for vgonio heightfield/microsurface archives.
pub const EXTENSION_SURF: &str = "vgsurf";

/// Internal manifest filename inside the container.
pub const MANIFEST_FILENAME: &str = "manifest.toml";
/// Internal partition descriptor filename.
pub const PARTITION_FILENAME: &str = "partition.toml";
/// Internal incident-grid filename (BSDF only).
pub const INCIDENT_GRID_FILENAME: &str = "incident_grid.toml";
/// Internal spectrum filename.
pub const SPECTRUM_FILENAME: &str = "spectrum.toml";

//! `.vgbsdf` archival container - a JAR-style zip wrapping per-bounce-level OpenEXR
//! files plus structured TOML metadata.
//!
//! Container compression is always `Store` (ZIP method 0); the EXR members handle
//! their own internal compression.

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

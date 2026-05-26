//! Manifest of the VG-BSDF format.
//!
//! Example content:
//!
//! ```toml
//! [vgonio]
//! version = "0.2.0"                     # vgonio version that produced this archive
//! conventions = "v1"                    # VGONIO EXR Conventions version — see Part 3
//!
//! [archive]
//! created = "2026-05-14T12:34:56Z"      # RFC 3339
//! type = "bsdf"                         # discriminator — future: "ndf", "msf", "sdf", "surface"
//! description = "Aluminium BSDF measured at 256 incident directions, 4 wavelengths"
//!
//! [material]
//! # Free-form material identity. Refers to vgonio's Medium registry by canonical name.
//! # Under MEDIUM_DATA_DRIVEN_SPEC.md, medium names are arbitrary strings, not enum tags.
//! incident_medium = "air"
//! transmitted_medium = "al"
//!
//! [bsdf]
//! # Which BrdfLevel directories are present in this archive.
//! levels = ["l0", "l1", "l1+"]
//! # Which outgoing-domain encodings each level provides. Always all three under v1.
//! encodings = ["disc", "thetaphi", "patches"]
//!
//! [provenance]
//! # Optional but encouraged. Free-form for citation / reproducibility.
//! software = "vgonio 0.2.0"
//! git_commit = "574fcaf3"               # optional
//! input_surface = "samples/aluminium_rough_001.vgms"  # optional, path or hash
//! ```
use serde::{Deserialize, Serialize};

/// Top-level archive manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    /// Identifies the writing vgonio version and conventions revision.
    pub vgonio: VgonioBlock,
    /// Archive creation metadata.
    pub archive: ArchiveBlock,
    /// Optional material identity (incident/transmitted media).
    pub material: Option<MaterialBlock>,
    /// Optional BSDF-specific layout block.
    #[serde(default)]
    pub bsdf: Option<BsdfBlock>,
    /// Optional provenance for citation / reproducibility.
    #[serde(default)]
    pub provenance: Option<ProvenanceBlock>,
}

/// Identifies the writing vgonio version and the conventions revision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VgonioBlock {
    /// Vgonio version that produced this archive (e.g. "0.2.0").
    pub version: String,
    /// VGONIO EXR Conventions version. Only "v1" is valid under this spec.
    pub conventions: String,
}

/// Archive creation metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArchiveBlock {
    /// RFC 3339 timestamp of archive creation.
    pub created: String,
    /// "bsdf" | "ndf" | "msf" | "sdf" | "heightfield".
    #[serde(rename = "type")]
    pub kind: String,
    /// Free-form human-readable description.
    pub description: Option<String>,
}

/// Material identity — the media on either side of the surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MaterialBlock {
    /// Canonical name of the incident-side medium in vgonio's registry.
    pub incident_medium: String,
    /// Canonical name of the transmitted-side medium in vgonio's registry.
    pub transmitted_medium: String,
}

/// BSDF-specific archive layout — which levels and encodings are present.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BsdfBlock {
    /// Which BrdfLevel directories are present, e.g. ["l0", "l1", "l1+"].
    pub levels: Vec<String>,
    /// Which outgoing-domain encodings are emitted per level.
    /// Always ["disc", "thetaphi", "patches"] under v1.
    pub encodings: Vec<String>,
}

/// Optional provenance information for citation and reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ProvenanceBlock {
    /// Name and version of the software that produced this archive.
    pub software: Option<String>,
    /// Git commit hash of the producing software, if available.
    pub git_commit: Option<String>,
    /// Identifier (path or hash) of the input surface used.
    pub input_surface: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            vgonio: VgonioBlock {
                version: "0.2.0".into(),
                conventions: "v1".into(),
            },
            archive: ArchiveBlock {
                created: "2026-05-14T12:34:56Z".into(),
                kind: "bsdf".into(),
                description: Some("Aluminium BSDF, 256 wi, 4 wavelengths".into()),
            },
            material: Some(MaterialBlock {
                incident_medium: "air".into(),
                transmitted_medium: "al".into(),
            }),
            bsdf: Some(BsdfBlock {
                levels: vec!["l0".into(), "l1".into(), "l1+".into()],
                encodings: vec!["disc".into(), "thetaphi".into(), "patches".into()],
            }),
            provenance: Some(ProvenanceBlock {
                software: Some("vgonio 0.2.0".into()),
                git_commit: Some("574fcaf3".into()),
                input_surface: None,
            }),
        }
    }

    #[test]
    fn manifest_round_trip_full() {
        let m = sample();
        let s = toml::to_string_pretty(&m).unwrap();
        let r: Manifest = toml::from_str(&s).unwrap();
        assert_eq!(m, r);
    }

    #[test]
    fn manifest_round_trip_minimal() {
        // Spec says material/bsdf/provenance are optional.
        let m = Manifest {
            vgonio: VgonioBlock {
                version: "0.2.0".into(),
                conventions: "v1".into(),
            },
            archive: ArchiveBlock {
                created: "2026-05-14T12:34:56Z".into(),
                kind: "ndf".into(),
                description: None,
            },
            material: None,
            bsdf: None,
            provenance: None,
        };
        let s = toml::to_string_pretty(&m).unwrap();
        let r: Manifest = toml::from_str(&s).unwrap();
        assert_eq!(m, r);
    }

    #[test]
    fn manifest_serialised_keys_match_spec() {
        let s = toml::to_string_pretty(&sample()).unwrap();
        assert!(s.contains("[vgonio]"));
        assert!(s.contains("[archive]"));
        assert!(s.contains("[material]"));
        assert!(s.contains("[bsdf]"));
        assert!(s.contains("[provenance]"));
        assert!(s.contains("type =")); // ArchiveBlock.kind renamed
    }
}

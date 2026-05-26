//! Artifact references: the wire-side handle to an immutable byte blob.
use serde::{Deserialize, Serialize};

use crate::ids::ArtifactId;

/// Wire reference to one artifact blob, carried in
/// [`crate::envelope::JobEnvelope::inputs`] and returned in capability results.
///
/// Pairs an opaque routing [`ArtifactId`] with a content-derived [`Checksum`]
/// so consumers can both fetch the blob via [`origin`](Self::origin) and
/// verify its bytes on receipt. See the `ArtifactId` docs in
/// [`crate::ids`] for the routing-vs-integrity split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub kind: ArtifactKind,
    pub origin: ArtifactOrigin,
    pub checksum: Checksum,
    pub size_bytes: u64,
}

/// The concrete on-disk format of the artifact's bytes.
///
/// `#[non_exhaustive]` so new variants land without a protocol bump (the wire
/// is the snake-case discriminant string, so an older reader sees an unknown
/// kind and can reject cleanly).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArtifactKind {
    /// `.vgbsdf` archival container (BSDF / NDF / MSF / SDF, governed by
    /// VGONIO EXR Conventions v1).
    Vgbsdf,
    /// `.vgms` bespoke local cache file (heightfields). Workers exchange these uncompressed across
    /// the artifact boundary even when stored LZ4'd locally.
    Vgms,
    /// `.vgmo` legacy measurement output.
    Vgmo,
    /// `.ior.ron` refractive-index data for a single medium, keyed by
    /// `MediumId`.
    IorRon,
    /// EXR image.
    Exr,
    /// Generic blob with a MIME-style content type hint (e.g. `application/octet-stream` for
    /// opaque binary).
    Raw { mime: String },
}

/// Where the artifact's bytes can be fetched from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scheme", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArtifactOrigin {
    /// Path on the local filesystem of whichever process is reading the ref.
    /// Only valid within a single host (the local executor / single-machine
    /// worker).
    LocalPath(String),
    /// HTTP(S) URL the consumer can fetch.
    HttpUrl(String),
    /// The bytes live in an in-process [`crate::context::JobContext`] artifact
    /// store. Lookup by [`ArtifactId`].
    Inline,
}

/// Content-derived integrity tag.
///
/// Wire shape: `{"algo": "sha256", "value": "<hex>"}`, letting future
/// algorithms (Blake3, ...) land as new variants without re-encoding existing
/// values. `#[non_exhaustive]` for the same reason.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "algo", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Checksum {
    Sha256(String),
}

impl Checksum {
    /// Wraps a pre-computed sha256 hex string. The string is **not** validated
    /// here; Task 1.7 will add a `Checksum::compute(&[u8])` helper and a
    /// validating parser.
    pub fn sha256_hex(s: impl Into<String>) -> Self { Self::Sha256(s.into()) }

    /// Returns the hex-encoded digest, regardless of algorithm.
    pub fn hex(&self) -> &str {
        match self {
            Checksum::Sha256(h) => h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ref() -> ArtifactRef {
        ArtifactRef {
            id: ArtifactId::new(),
            kind: ArtifactKind::Vgbsdf,
            origin: ArtifactOrigin::LocalPath("/tmp/sample.vgbsdf".into()),
            checksum: Checksum::sha256_hex("deadbeef"),
            size_bytes: 123456,
        }
    }

    #[test]
    fn artifact_ref_roundtrips_json() {
        let r = sample_ref();
        let json = serde_json::to_string(&r).unwrap();
        let back: ArtifactRef = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn artifact_kind_wire_uses_snake_case() {
        let json = serde_json::to_string(&ArtifactKind::Vgbsdf).unwrap();
        assert_eq!(json, "\"vgbsdf\"");
        let json = serde_json::to_string(&ArtifactKind::IorRon).unwrap();
        assert_eq!(json, "\"ior_ron\"");
    }

    #[test]
    fn checksum_wire_is_tagged() {
        let c = Checksum::sha256_hex("abc");
        let json = serde_json::to_string(&c).unwrap();
        // Tag/content form: algorithm-agnostic, future Blake3 won't re-encode.
        assert_eq!(json, r#"{"algo":"sha256","value":"abc"}"#);
    }

    #[test]
    fn checksum_hex_accessor() {
        assert_eq!(Checksum::sha256_hex("0123").hex(), "0123");
    }

    #[test]
    fn artifact_origin_local_path_roundtrips() {
        let o = ArtifactOrigin::LocalPath("/tmp/x".into());
        let json = serde_json::to_string(&o).unwrap();
        assert_eq!(json, r#"{"scheme":"local_path","value":"/tmp/x"}"#);
        let back: ArtifactOrigin = serde_json::from_str(&json).unwrap();
        assert_eq!(o, back);
    }
}

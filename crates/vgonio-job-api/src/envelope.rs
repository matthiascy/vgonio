//! Versioned wire envelope for a single submitted job.
//!
//! [`JobEnvelope`] is the *one struct that crosses the wire* per submission;
//! everything else in this crate (IDs, resources, handshake, progress, errors)
//! either nests inside it or describes the steady-state around it.
//!
//! # Versioning
//!
//! Two independent counters travel with every envelope:
//!
//! - **`protocol_version`**: bumped when *this struct* (or any of its nested schema) changes.
//!   Workers compare it against [`crate::handshake::WorkerCapabilities::protocol_version`] during
//!   routing and reject mismatches loudly. Producers must always set it from
//!   [`crate::PROTOCOL_VERSION`] (see [`JobEnvelope::new`]).
//! - **`capability_version`**: per-capability payload schema version. A worker may accept several
//!   payload versions for one capability (declared in
//!   [`crate::handshake::CapabilityDescriptor::accepted_versions`]); the capability owner controls
//!   when to bump.
//!
//! The two are deliberately decoupled so a payload-only change (e.g. new
//! optional field in `FitOptions`) doesn't force a protocol-wide reroll.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    artifact::ArtifactRef,
    ids::{CapabilityId, IdempotencyKey, JobId},
    resources::ResourceHints,
};

/// Wire envelope for a single job submission.
///
/// Construct via [`JobEnvelope::new`] so `protocol_version` is always seeded
/// from [`crate::PROTOCOL_VERSION`]; building the struct literally is allowed
/// but easy to get wrong.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEnvelope {
    /// Server-assigned ID for *this attempt*. See [`crate::ids::JobId`] for
    /// the JobId-vs-IdempotencyKey distinction.
    pub job_id: JobId,
    /// Wire protocol version: always [`crate::PROTOCOL_VERSION`] for envelopes
    /// produced by this crate.
    pub protocol_version: u32,
    /// Which capability is being invoked. See [`crate::ids::CapabilityId`]
    /// for the canonical set.
    pub capability_id: CapabilityId,
    /// Per-capability payload schema version (independent of
    /// `protocol_version`).
    pub capability_version: u32,
    /// How `payload` is encoded; the receiver picks the matching decoder.
    pub payload_encoding: PayloadEncoding,
    /// Capability-specific request bytes. Opaque to the executor; only the
    /// matching capability handler deserializes it.
    #[serde(with = "bytes_serde")]
    pub payload: Bytes,
    /// Resource / feature requirements the scheduler matches against worker
    /// handshakes.
    pub resources: ResourceHints,
    /// Input artifacts (heightfields, previous measurement outputs, IOR data,
    /// ...). Order is preserved and may be semantically significant.
    pub inputs: Vec<ArtifactRef>,
    /// Caller-chosen key that deduplicates retried submissions. See
    /// [`crate::ids::IdempotencyKey`] for the recommended seed composition.
    pub idempotency_key: IdempotencyKey,
    /// Distributed-tracing context, threaded through to capability handlers
    /// for observability.
    pub trace: TraceContext,
}

impl JobEnvelope {
    /// Builds a new envelope with `protocol_version` correctly seeded from
    /// [`crate::PROTOCOL_VERSION`] and `id` minted fresh.
    pub fn new(
        capability_id: CapabilityId,
        capability_version: u32,
        payload_encoding: PayloadEncoding,
        payload: Bytes,
        resources: ResourceHints,
        inputs: Vec<ArtifactRef>,
        idempotency_key: IdempotencyKey,
        trace: TraceContext,
    ) -> Self {
        Self {
            job_id: JobId::new(),
            protocol_version: crate::PROTOCOL_VERSION,
            capability_id,
            capability_version,
            payload_encoding,
            payload,
            resources,
            inputs,
            idempotency_key,
            trace,
        }
    }
}

/// How the capability payload bytes are encoded on the wire.
///
/// `#[non_exhaustive]` is mandatory: adding `MessagePack` later must not break
/// existing consumers that wildcard-match. LZ4 is **not** valid here; it's a
/// cache-layer-only scheme (see [`vgn_core::io::CompressionScheme`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PayloadEncoding {
    /// JSON: the v1 default; human-readable, debuggable, widely supported.
    Json,
    /// Bincode: compact binary; chosen per-payload when JSON's overhead hurts
    /// (large fit-residual arrays, dense MeasurementParams).
    Bincode,
    /// CBOR: reserved for future use; not produced by any current capability.
    Cbor,
}

/// Minimal distributed-tracing context propagated end-to-end through the job.
///
/// Both fields are `None` for un-traced submissions (default). When integrated
/// with `tracing`, set `trace_id` to the W3C trace-context ID and
/// `parent_span_id` to the submitter's current span ID; worker-side spans
/// chain off it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: Option<String>,
    pub parent_span_id: Option<String>,
}

mod bytes_serde {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(b: &Bytes, s: S) -> Result<S::Ok, S::Error> {
        // base64 string for JSON-friendliness; consumers using bincode can override.
        STANDARD.encode(b).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Bytes, D::Error> {
        let s = String::deserialize(d)?;
        let v = STANDARD.decode(&s).map_err(serde::de::Error::custom)?;
        Ok(Bytes::from(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{ArtifactKind, ArtifactOrigin, ArtifactRef, Checksum},
        ids::{ArtifactId, CapabilityId, IdempotencyKey},
        resources::ResourceHints,
    };

    fn sample_envelope() -> JobEnvelope {
        JobEnvelope::new(
            CapabilityId::measure_bsdf(),
            1,
            PayloadEncoding::Json,
            Bytes::from_static(b"{\"params\":\"...\"}"),
            ResourceHints::default(),
            vec![ArtifactRef {
                id: ArtifactId::new(),
                kind: ArtifactKind::Vgms,
                origin: ArtifactOrigin::LocalPath("/tmp/surface.vgms".into()),
                checksum: Checksum::sha256_hex("cafebabe"),
                size_bytes: 42,
            }],
            IdempotencyKey::from_hash(b"seed"),
            TraceContext::default(),
        )
    }

    #[test]
    fn new_seeds_protocol_version_from_crate_const() {
        let env = sample_envelope();
        assert_eq!(env.protocol_version, crate::PROTOCOL_VERSION);
    }

    #[test]
    fn envelope_roundtrips_json() {
        let env = sample_envelope();
        let json = serde_json::to_string(&env).unwrap();
        let back: JobEnvelope = serde_json::from_str(&json).unwrap();

        assert_eq!(env.job_id, back.job_id);
        assert_eq!(env.protocol_version, back.protocol_version);
        assert_eq!(env.capability_id, back.capability_id);
        assert_eq!(env.capability_version, back.capability_version);
        assert_eq!(env.payload_encoding, back.payload_encoding);
        assert_eq!(env.payload, back.payload);
        assert_eq!(env.inputs, back.inputs);
        assert_eq!(env.idempotency_key, back.idempotency_key);
    }

    #[test]
    fn payload_bytes_survive_base64_roundtrip() {
        let env = JobEnvelope::new(
            CapabilityId::fit(),
            1,
            PayloadEncoding::Bincode,
            // Non-UTF-8 bytes; proves we're not accidentally treating the
            // payload as a string somewhere.
            Bytes::from_static(&[0x00, 0xff, 0x7f, 0x80, 0xfe]),
            ResourceHints::default(),
            vec![],
            IdempotencyKey::from_hash(b"x"),
            TraceContext::default(),
        );
        let json = serde_json::to_string(&env).unwrap();
        let back: JobEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(env.payload, back.payload);
    }

    #[test]
    fn payload_encoding_wire_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&PayloadEncoding::Json).unwrap(),
            "\"json\""
        );
        assert_eq!(
            serde_json::to_string(&PayloadEncoding::Bincode).unwrap(),
            "\"bincode\""
        );
    }

    #[test]
    fn trace_context_defaults_omit_cleanly() {
        let env = sample_envelope();
        let json = serde_json::to_string(&env).unwrap();
        // Default trace context fields are null; deserialize must accept them.
        let back: JobEnvelope = serde_json::from_str(&json).unwrap();
        assert!(back.trace.trace_id.is_none());
        assert!(back.trace.parent_span_id.is_none());
    }
}

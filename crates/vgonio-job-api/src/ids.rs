//! Typed identifiers used throughout the job API.
//!
//! Every ID here is `#[serde(transparent)]` so it serializes as its inner
//! value (a UUID string or a plain string), keeping the wire shape stable and
//! free of newtype envelopes. The newtypes exist purely to prevent
//! `JobId` / `WorkerId` / `ArtifactId` from being mistaken for each other at
//! the type level.
//!
//! # Relationship to other identifier systems
//!
//! This crate is the *job/transport* identity layer. Two other identity systems
//! live elsewhere in the workspace and intentionally don't go through here:
//!
//! - **`vgn_core::utils::medium::MediumId`** — identifies a physical medium (`"vac"`, `"air"`,
//!   `"al"`, …). Backed by a leak-pooled `Box<str>` intern table for the process lifetime, *not* a
//!   UUID — content equality is the key invariant so that `MediumId::AIR` from two crates compare
//!   equal. Used to key IOR data (`.ior.ron`) and to tag measurement file headers. The 3-byte
//!   medium field at the BSDF file-header offset is forward-compatible, which is why the BSDF
//!   format stayed at v0.1.0 across the data-driven Medium refactor — see
//!   `design/MEDIUM_DATA_DRIVEN_SPEC.md`.
//!
//!   On the wire, a `MediumId` serializes as its `&'static str` *content* (the canonical short
//!   name); the intern pool is a local-process address-equality optimization, never a wire
//!   concept. Receivers deserialize via `MediumId::try_from_name(&str)`; there is no public
//!   `MediumId::new`, so a payload carrying an unknown name fails resolution at the worker
//!   boundary rather than fabricating an unregistered ID. Every worker is guaranteed to recognize
//!   the *built-in* `MediumId` constants (`VACUUM`, `AIR`, `AL`, …); custom media added at runtime
//!   require either shipping the corresponding `.ior.ron` blob to the worker as an input
//!   artifact, or a future feature flag like `ior-extended-database` advertised in the worker's
//!   [`crate::handshake::WorkerCapabilities`].
//!
//! - **`Checksum`** (defined in [`crate::artifact`]) — content-derived integrity tag (sha256).
//!   Pairs with [`ArtifactId`] on every [`crate::artifact::ArtifactRef`] so consumers can both
//!   *route* to a blob (by ID) and *verify* it (by checksum).
//!
//! When in doubt: use UUID-based IDs from this module for "what job/attempt/
//! worker/blob is this", `MediumId` for medium identity, and `Checksum` for
//! content integrity.
//!
//! # Worked lifecycle: a `measure-bsdf` submission
//!
//! Walking the IDs in the order they appear:
//!
//! ```text
//! Client (CLI adapter)
//!   heightfield_ref = ArtifactRef {
//!       id:       ArtifactId::new(),                    // routing handle
//!       kind:     Vgms,                                 // .vgms heightfield cache
//!       origin:   <local fs path or remote URL>,
//!       checksum: Sha256(...),                          // content integrity
//!   }
//!   key_seed = b"measure-bsdf" ++ canonical(params) ++ checksum.bytes
//!            ++ sorted([MediumId::AL.as_str(), MediumId::AIR.as_str()]).bytes
//!   envelope = JobEnvelope {
//!       id:                 JobId::new(),               // this attempt
//!       capability_id:      CapabilityId::measure_bsdf(),
//!       capability_version: 1,
//!       payload:            canonical(MeasurementParams { incident, output, ... }),
//!       inputs:             vec![heightfield_ref],
//!       resources:          ResourceHints {
//!                               required_features: vec![FeatureId::embree()], ... },
//!       idempotency_key:    IdempotencyKey::from_hash(&sha256(key_seed)),
//!   }
//!
//! Executor (local or remote)
//!   if dedup_cache.contains((capability_id, idempotency_key)) -> return prior JobId
//!   else route to a worker advertising FeatureId::embree() in its handshake
//!
//! Worker
//!   resolves the MediumIds against its local registry (failure = JobError)
//!   resolves heightfield_ref via the ArtifactStore (failure = JobError)
//!   runs the measurement; scratch goes through the local LZ4 .vgms cache layer
//!   publishes the result as a single .vgbsdf zip — per-bounce-level subdirs
//!     l0/, l1/, l1+/, each holding disc.exr + thetaphi.exr + patches.exr plus
//!     manifest.toml (the manifest carries provenance: producing JobId, the
//!     MediumId list, the heightfield's Checksum, capability_version)
//!   returns ArtifactRef { id: ArtifactId::new(), kind: Vgbsdf, ... }
//! ```
//!
//! Note the layering: `JobId` lives for one attempt; `IdempotencyKey` lives
//! across retries of the same logical request; `ArtifactId` lives for the life
//! of the blob in the store; `MediumId` and `FeatureId` live across the entire
//! deployment. The LZ4 file cache and the executor-level `IdempotencyKey` dedup
//! sit at *different layers* and never see each other — the file cache lives
//! inside the worker, below the artifact store, and is invisible to the wire
//! protocol.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Generates a `#[serde(transparent)]` newtype around `Uuid` with the
/// standard constructor / `Default` / `Display` boilerplate.
macro_rules! uuid_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Mints a fresh random (v4) identifier.
            pub fn new() -> Self { Self(Uuid::new_v4()) }
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
        }
    };
}

uuid_id!(
    JobId,
    "Identifies one *attempt* to run a job - minted by the executor when an envelope is accepted. \
     Distinct from [`IdempotencyKey`], which identifies a logical request and may map to the same \
     `JobId` across retries."
);
uuid_id!(
    ArtifactId,
    "Identifies a single artifact blob (job input or output). Carried inside \
     [`crate::artifact::ArtifactRef`] alongside the origin/checksum needed to fetch and verify \
     it. \n\nThe ID itself is opaque and *not* content-derived — content integrity lives on the \
     sibling `Checksum` field. This split lets two physically distinct blobs that happen to have \
     the same bytes (e.g. the same `.vgbsdf` archive re-emitted by two runs) keep separate \
     provenance while still being deduplicable by their checksums. \n\nConcrete artifact formats \
     this crate is designed to carry (see `design/INTERNAL_FORMATS_REDESIGN_SPEC.md`): \n- \
     `.vgbsdf` — JAR-style zip container (`Store` method, **no container compression** — the EXR \
     payloads inside are already compressed), the canonical archival format for \
     hemisphere-carried measurement outputs (BSDF / NDF / MSF / SDF). One container per \
     measurement; per-bounce-level subdirectories `l0/` (single-bounce), `l1/` (first-order \
     multiple), `l1+/` (full multi-bounce), each holding up to three projections (`disc.exr`, \
     `thetaphi.exr`, `patches.exr`) of the same patch data under the **VGONIO EXR Conventions \
     v1**, plus a `manifest.toml` recording the partition, incident grid, spectrum, and \
     provenance (producing `JobId`, `CapabilityId`, `capability_version`, MediumId list, input \
     heightfield's `Checksum`). \n- `.vgms` / `.vgmo` — bespoke local cache files for \
     heightfields and measurement scratch, opt-in LZ4-compressible via `CompressionScheme::Lz4` \
     (cache layer only; never crosses the artifact-store boundary as LZ4 — workers exchange the \
     uncompressed payload). \n- `.ior.ron` — refractive-index data for one medium, keyed by \
     [`vgn_core::utils::medium::MediumId`]. Produced by the `cargo x ior` fetch tool against a \
     refractiveindex.info catalog snapshot; a worker that lacks a given `MediumId` in its \
     baseline registry can be sent the matching `.ior.ron` as an input artifact instead of \
     needing a code-level registry update."
);
uuid_id!(
    WorkerId,
    "Stable identity of a worker *installation* - chosen by the worker once and persisted across \
     restarts, so the scheduler can recognize a returning worker as the same one. Pair with \
     [`WorkerSessionId`] to distinguish individual handshakes."
);
uuid_id!(
    WorkerSessionId,
    "Fresh-per-handshake identity for a worker connection. Lets the scheduler tell \"same worker \
     reconnected\" (new session, same [`WorkerId`]) from \"new worker registered\" (new session \
     *and* new [`WorkerId`])."
);

/// Client-chosen key that deduplicates retried submissions.
///
/// Two envelopes with the same `(capability_id, idempotency_key)` must produce
/// the same result: on the second submission the executor returns the existing
/// [`JobId`] / its status instead of running the work again. The key is the
/// caller's promise that "this is the same logical request as before"; the
/// executor caches the mapping with a TTL.
///
/// Two ways to populate it:
/// - **Auto-derive** via [`Self::from_hash`] over a canonical serialization of the request (payload
///   bytes + inputs + capability). Same inputs => same key => free dedup of retries.
/// - **User-supplied**: pass an explicit UUID stored alongside the caller's own request record,
///   when "same inputs, force a re-run" must be possible.
///
/// [`JobId`] identifies an *attempt*; this identifies the *logical request*.
///
/// # Recommended seed composition (per canonical capability)
///
/// The seed fed into [`Self::from_hash`] should be a *canonical* (byte-stable)
/// serialization. For the capabilities defined in [`CapabilityId`] below:
///
/// - **`measure-bsdf` / `measure-ndf` / `measure-msf` / `measure-sdf`**: `capability_id ++
///   canonical(MeasurementParams) ++ input_heightfield.checksum ++ sorted(MediumId list).bytes`.
///   Including the input heightfield's `Checksum` (the one carried alongside its [`ArtifactId`])
///   means re-submitting the same measurement on the same surface is automatically deduped;
///   including the sorted `MediumId` list (from `vgn_core::utils::medium`) means swapping `air` and
///   `vacuum` in the params bumps the key as it should.
/// - **`fit`**: `capability_id ++ canonical(FitOptions) ++ input_brdf_artifact.checksum`.
///
/// # Layering versus the local cache
///
/// This is *executor-level* dedup (avoid re-running the *job*). It's distinct
/// from (and complementary to) the file-level LZ4 cache for `.vgmo` / `.vgms`
/// scratch files described in the internal-formats cache plan, which dedups at
/// the `read/write to disk` layer regardless of which job produced the bytes.
/// Both layers can coexist; neither subsumes the other.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IdempotencyKey(pub String);

impl IdempotencyKey {
    // TODO: replace with sha256(seed) once `Checksum` is available
    // and rename to make the hashing direction explicit.
    /// Builds a key from an already-hashed seed.
    ///
    /// The seed is hex-encoded verbatim — this function does **not** hash for
    /// you. Callers must feed in a cryptographic digest (sha256 of the
    /// canonical request bytes is the intended choice once the `Checksum`
    /// helper lands); a non-cryptographic hash risks collisions
    /// that would return the wrong cached result.
    ///
    /// # Canonicalization pitfalls
    ///
    /// "Same logical request" is a *bytes-level* claim, so the seed has to be
    /// byte-stable across runs. Common foot-guns when assembling the seed:
    ///
    /// - **Floats in `MeasurementParams` / `FitOptions`**: serialize via a canonical encoder (RON
    ///   with explicit precision, or hash the IEEE-754 bit pattern) — `Debug`/`Display` formatting
    ///   is locale- and Rust-version- sensitive.
    /// - **`MediumId` lists**: sort by `.as_str()` before hashing. The intern- pool insertion order
    ///   is not stable across processes, so the in-memory order of a `Vec<MediumId>` may differ
    ///   between client and worker; the string content is what's invariant.
    /// - **`Vec<ArtifactRef>` for `inputs`**: hash by `(kind, checksum)` tuples, not by
    ///   `ArtifactId` — two clients holding the same heightfield blob under different `ArtifactId`s
    ///   should still dedup. Preserve insertion order if it carries meaning (e.g. positional
    ///   capability arguments); sort it if it doesn't.
    /// - **`HashMap` iteration**: never. Use `BTreeMap`, or sort entries before serializing.
    pub fn from_hash(seed: &[u8]) -> Self { Self(hex::encode(seed)) }
}

/// Names a capability — the kind of work a worker can perform.
///
/// Kept as a free-form string (not an enum) so new capabilities can be added
/// without a `vgonio-job-api` version bump: workers advertise the IDs they
/// handle in their [`crate::handshake::WorkerCapabilities`], and the scheduler
/// routes by string match. The constructors below are the canonical IDs known
/// to this crate; downstream crates may mint their own.
///
/// # Canonical IDs (mirrors the measurement / fitting capability split)
///
/// | ID | What it does | Output artifact kind |
/// |---|---|---|
/// | `"fit"` | Fits an analytical BxDF model to measured data. | `.vgbsdf` (fitted model layer) + report |
/// | `"measure-bsdf"` | Runs a BSDF measurement on a heightfield. | `.vgbsdf` (per-bounce-level subdirs) |
/// | `"measure-ndf"` | Computes a normal-distribution function. | `.vgbsdf` (NDF projection) |
/// | `"measure-msf"` | Computes the masking-shadowing function. | `.vgbsdf` (MSF projection) |
/// | `"measure-sdf"` | Computes the slope-distribution function. | `.vgbsdf` (SDF projection, no `disc.exr`) |
///
/// All four `measure-*` capabilities write the same `.vgbsdf` container shape
/// (one zip per measurement, governed by the **VGONIO EXR Conventions v1**);
/// the differences live in which projections are emitted and what the manifest
/// records.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilityId(pub String);

impl CapabilityId {
    pub fn fit() -> Self { Self("fit".into()) }
    pub fn measure_bsdf() -> Self { Self("measure-bsdf".into()) }
    pub fn measure_ndf() -> Self { Self("measure-ndf".into()) }
    pub fn measure_msf() -> Self { Self("measure-msf".into()) }
    pub fn measure_sdf() -> Self { Self("measure-sdf".into()) }
}

/// Names a compile-time backend feature a worker was built with
/// (`"embree"`, `"cuda"`, `"optix"`, `"wgpu"`, ...).
///
/// Workers advertise their enabled features in the handshake; submitters list
/// the features they require via
/// [`crate::resources::ResourceHints::required_features`]; the scheduler
/// matches the two sets when picking a worker. Like [`CapabilityId`], kept as
/// a free-form string so new backends don't force a protocol bump.
///
/// # Canonical IDs and the planned backend-crate split
///
/// The four built-in values map one-to-one to the optional measurement-backend
/// crates introduced in Phase 2 of the distributed-capability plan: each
/// `vgonio-measurement-{embree,cuda,wgpu}` (and OptiX, folded in or split later)
/// is a *worker-side* crate; a worker only enables the `FeatureId` for backends
/// it was compiled with. This is what lets one CLI binary stay light while a
/// heavy GPU worker pulls in CUDA/OptiX without polluting every consumer.
///
/// # What is *not* a `FeatureId`
///
/// Feature flags are for compile-time *backend* capability — code paths that
/// either exist in the worker binary or don't. Things that vary at runtime and
/// can be supplied per-job belong elsewhere:
///
/// - **IOR / medium data availability** is *not* a feature flag. Every worker carries the built-in
///   `MediumId` set (`VACUUM`, `AIR`, `AL`, …) and its baseline IOR database; jobs that need a
///   medium outside that baseline ship the matching `.ior.ron` blob in `JobEnvelope.inputs` as an
///   artifact. If a bulk preloaded-database distinction becomes useful later, the convention is a
///   separately-named feature like `ior-extended-database` — but baseline media never need one.
/// - **Heightfields, measured BRDFs, fitted-model archives** are artifacts ([`ArtifactId`] +
///   [`crate::artifact::ArtifactRef`]), not features.
/// - **Per-job resource hints** (cpu cores, memory, gpu requirement) live on
///   [`crate::resources::ResourceHints`], alongside (not inside) the `required_features` list.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FeatureId(pub String);

impl FeatureId {
    pub fn embree() -> Self { Self("embree".into()) }
    pub fn cuda() -> Self { Self("cuda".into()) }
    pub fn optix() -> Self { Self("optix".into()) }
    pub fn wgpu() -> Self { Self("wgpu".into()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_id_roundtrips_json() {
        let id = JobId::new();
        let json = serde_json::to_string(&id).unwrap();
        let back: JobId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn feature_id_has_known_values() {
        assert_eq!(FeatureId::embree().0, "embree");
    }
}

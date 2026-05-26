//! Worker capability handshake DTOs.
//!
//! When a worker connects to a scheduler, it sends a [`WorkerCapabilities`]
//! describing what it can do; the scheduler indexes the advertisement and
//! later matches submitted [`crate::envelope::JobEnvelope`]s against it.
//!
//! Match logic the scheduler runs on each submission:
//!
//! 1. `envelope.protocol_version == worker.protocol_version` (hard).
//! 2. There exists a `CapabilityDescriptor` in `worker.capabilities` with matching `id` and
//!    `envelope.capability_version ∈ accepted_versions` (hard).
//! 3. Every `envelope.resources.required_features` is in `worker.enabled_features` (hard).
//! 4. `worker.resources` are *sufficient* for `envelope.resources` hints (soft / ranking).
//!
//! A worker is *not* committing to atomic identity across reconnects via
//! this struct; that's [`crate::ids::WorkerId`] (stable across restarts) paired
//! with [`crate::ids::WorkerSessionId`] (fresh per handshake).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{CapabilityId, FeatureId, WorkerId};

/// The full advertisement a worker sends to the scheduler on handshake.
///
/// Wire-stable: receivers should treat unknown fields tolerantly (serde does
/// by default) so a newer worker can talk to an older scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Stable installation ID of this worker. See [`WorkerId`].
    pub worker_id: WorkerId,
    /// Free-form version string of the worker binary (e.g.
    /// `"vgonio-worker 0.3.2 (commit abcd123)"`). Informational; not parsed by
    /// the scheduler.
    pub worker_version: String,
    /// Wire protocol version this worker speaks — must equal
    /// [`crate::PROTOCOL_VERSION`] in *its* dependency on this crate. The
    /// scheduler rejects envelopes whose `protocol_version` doesn't match.
    pub protocol_version: u32,
    /// Every capability this worker can serve, with the per-capability payload
    /// versions it accepts.
    pub capabilities: Vec<CapabilityDescriptor>,
    /// Backend features this worker was compiled with (embree, cuda, ...).
    /// Matched against [`crate::resources::ResourceHints::required_features`].
    pub enabled_features: Vec<FeatureId>,
    /// What the worker *has* on hand — used for hint matching / ranking.
    pub resources: WorkerResources,
    /// UTC start time of this handshake. Lets the scheduler distinguish a
    /// reconnect (new `started_at`, same `worker_id`) from a stale snapshot.
    pub started_at: DateTime<Utc>,
}

/// One capability a worker advertises, plus the payload schema versions it
/// understands for that capability.
///
/// A worker may accept several `accepted_versions` (e.g. `[1, 2]` during a
/// payload-format transition), letting the scheduler keep routing old envelopes
/// while clients migrate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityDescriptor {
    pub id: CapabilityId,
    pub accepted_versions: Vec<u32>,
}

/// Concrete hardware the worker brings to the table.
///
/// All counts are *available* (post-OS), not theoretical maxima.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerResources {
    pub cpu_cores: u16,
    /// Worker memory budget in **mebibytes** (2²⁰ bytes), matching the unit on
    /// [`crate::resources::ResourceHints::memory_mib`].
    pub memory_mib: u32,
    pub gpus: Vec<GpuInfo>,
}

/// One physical GPU attached to a worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    /// VRAM in mebibytes if known; `None` for adapters whose VRAM can't be
    /// queried (some integrated GPUs).
    pub memory_mib: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_capabilities() -> WorkerCapabilities {
        WorkerCapabilities {
            worker_id: WorkerId::new(),
            worker_version: "vgonio-worker 0.3.2".into(),
            protocol_version: crate::PROTOCOL_VERSION,
            capabilities: vec![
                CapabilityDescriptor {
                    id: CapabilityId::measure_bsdf(),
                    accepted_versions: vec![1],
                },
                CapabilityDescriptor {
                    id: CapabilityId::fit(),
                    accepted_versions: vec![1, 2],
                },
            ],
            enabled_features: vec![FeatureId::embree(), FeatureId::wgpu()],
            resources: WorkerResources {
                cpu_cores: 16,
                memory_mib: 32_768,
                gpus: vec![GpuInfo {
                    vendor: "NVIDIA".into(),
                    model: "RTX 4090".into(),
                    memory_mib: Some(24_576),
                }],
            },
            started_at: Utc::now(),
        }
    }

    #[test]
    fn worker_capabilities_roundtrip_json() {
        let h = sample_capabilities();
        let json = serde_json::to_string(&h).unwrap();
        let back: WorkerCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn capability_descriptor_supports_multiple_versions() {
        // A worker in the middle of a payload format transition can accept
        // both versions: this is what `accepted_versions` is for.
        let d = CapabilityDescriptor {
            id: CapabilityId::fit(),
            accepted_versions: vec![1, 2, 3],
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: CapabilityDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(back.accepted_versions, vec![1, 2, 3]);
    }

    #[test]
    fn gpu_info_memory_optional() {
        let g = GpuInfo {
            vendor: "Intel".into(),
            model: "Arc A770".into(),
            memory_mib: None,
        };
        let json = serde_json::to_string(&g).unwrap();
        let back: GpuInfo = serde_json::from_str(&json).unwrap();
        assert!(back.memory_mib.is_none());
    }

    #[test]
    fn worker_resources_default_is_empty_box() {
        let r = WorkerResources::default();
        assert_eq!(r.cpu_cores, 0);
        assert_eq!(r.memory_mib, 0);
        assert!(r.gpus.is_empty());
    }
}

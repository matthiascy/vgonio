//! Resource and feature hints attached to a job envelope.
//!
//! [`ResourceHints`] is the submitter's request for *what kind of worker*
//! should run the job. The scheduler matches these against worker handshake
//! advertisements ([`crate::handshake::WorkerCapabilities`]) before dispatch.
//!
//! Distinction to remember:
//!
//! - **`required_features`** are *hard* constraints; a worker without every feature in this list is
//!   ineligible. Use for backend-presence checks (`embree`, `cuda`, ...). See
//!   [`crate::ids::FeatureId`] for what does and does *not* belong here (IOR/medium data is shipped
//!   as artifacts, not gated by features).
//! - **`cpu_cores` / `memory_mib` / `gpu` / `max_runtime_secs`** are *hints*. They influence
//!   ranking and may be enforced as job-level quotas by the scheduler, but a worker satisfying the
//!   hard features won't be rejected for failing a hint alone.

use serde::{Deserialize, Serialize};

use crate::ids::FeatureId;

/// Submitter-declared resource expectations for a job.
///
/// All fields are optional / defaultable. A `ResourceHints::default()` means
/// "no preferences, route anywhere with the required features (none)".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceHints {
    /// Preferred CPU core count. Schedulers may use this for ranking or to
    /// configure rayon pool sizes on the worker.
    pub cpu_cores: Option<u16>,
    /// Preferred memory budget, in **mebibytes** (2²⁰ bytes). Document this
    /// unit at call sites so a caller doesn't accidentally pass MB.
    pub memory_mib: Option<u32>,
    /// GPU preference. Defaults to [`GpuHint::None`] (no GPU needed).
    pub gpu: GpuHint,
    /// Hard feature requirements. A worker missing any of these is
    /// ineligible. See module-level doc for hint-vs-requirement.
    pub required_features: Vec<FeatureId>,
    /// Soft wall-clock budget in seconds. Past this, the executor may cancel
    /// the job and surface a timeout.
    pub max_runtime_secs: Option<u32>,
}

/// GPU requirement attached to a job.
///
/// `#[non_exhaustive]` so future variants (e.g. minimum-VRAM constraint) land
/// without a protocol bump.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GpuHint {
    /// No GPU needed; the default. CPU-only jobs (`fit`, NDF/MSF/SDF compute,
    /// embree-based BSDF) leave this alone.
    None,
    /// Any GPU is fine (the job just needs one). Suitable for wgpu jobs that
    /// portably target whatever adapter the worker exposes.
    Any,
    /// Specific vendor + model required (e.g. CUDA jobs that only run on a
    /// particular NVIDIA generation).
    Specific { vendor: String, model: String },
}

impl Default for GpuHint {
    fn default() -> Self { GpuHint::None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_no_gpu_no_features_no_caps() {
        let h = ResourceHints::default();
        assert!(h.cpu_cores.is_none());
        assert!(h.memory_mib.is_none());
        assert_eq!(h.gpu, GpuHint::None);
        assert!(h.required_features.is_empty());
        assert!(h.max_runtime_secs.is_none());
    }

    #[test]
    fn resource_hints_roundtrip_json() {
        let h = ResourceHints {
            cpu_cores: Some(8),
            memory_mib: Some(4096),
            gpu: GpuHint::Specific {
                vendor: "NVIDIA".into(),
                model: "A100".into(),
            },
            required_features: vec![FeatureId::embree(), FeatureId::cuda()],
            max_runtime_secs: Some(600),
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: ResourceHints = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn gpu_hint_specific_roundtrips() {
        let g = GpuHint::Specific {
            vendor: "NVIDIA".into(),
            model: "RTX 4090".into(),
        };
        let json = serde_json::to_string(&g).unwrap();
        let back: GpuHint = serde_json::from_str(&json).unwrap();
        assert_eq!(g, back);
    }

    #[test]
    fn gpu_hint_wire_uses_snake_case() {
        assert_eq!(serde_json::to_string(&GpuHint::None).unwrap(), "\"none\"");
        assert_eq!(serde_json::to_string(&GpuHint::Any).unwrap(), "\"any\"");
        // Tag-included form for the struct variant.
        let json = serde_json::to_string(&GpuHint::Specific {
            vendor: "v".into(),
            model: "m".into(),
        })
        .unwrap();
        assert!(json.contains("\"specific\""));
        assert!(json.contains("\"vendor\":\"v\""));
        assert!(json.contains("\"model\":\"m\""));
    }
}

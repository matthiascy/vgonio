//! Embree BSDF ray-tracing backend.
//!
//! Stub scaffold: the `todo!()` bodies are replaced with calls
//! into the moved `embr::*` kernels in later commits.

use std::sync::Arc;

use vgn_core::{math::Sph2, optics::Ior};
use vgn_io::MicroSurfaceMesh;
use vgn_measurement::{
    backend::{BsdfRtBackend, DeviceHandle, GeometryHandle, SceneHandle, SingleSimResult},
    bsdf::emitter::EmitterCircularSector,
};

/// Embree-backed BSDF ray tracer.
pub struct EmbreeBackend;

impl BsdfRtBackend for EmbreeBackend {
    fn create_resources(
        &self,
        _mesh: &MicroSurfaceMesh,
    ) -> (DeviceHandle, SceneHandle, GeometryHandle) {
        todo!("embree backend impl lands in Task 2.8")
    }

    #[allow(clippy::too_many_arguments)]
    fn simulate_single_point(
        &self,
        _wi: Sph2,
        _sector: &EmitterCircularSector<'_>,
        _mesh: &MicroSurfaceMesh,
        _geometry: &GeometryHandle,
        _scene: &SceneHandle,
        #[cfg(not(feature = "vdbg"))] _fresnel: bool,
        #[cfg(not(feature = "vdbg"))] _iors_i: &[Ior],
        #[cfg(not(feature = "vdbg"))] _iors_t: &[Ior],
    ) -> SingleSimResult {
        todo!("embree backend impl lands in Task 2.8")
    }
}

/// Constructs the embree BSDF backend (CLI default path).
pub fn provide() -> Arc<dyn BsdfRtBackend> { Arc::new(EmbreeBackend) }

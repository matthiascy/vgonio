//! Embree BSDF ray-tracing backend.

mod embr;

use std::sync::Arc;

use embree::{Geometry, Scene};
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
        mesh: &MicroSurfaceMesh,
    ) -> (DeviceHandle, SceneHandle, GeometryHandle) {
        let (device, scene, geometry) = embr::create_resources(mesh);
        (
            DeviceHandle::new(device),
            SceneHandle::new(scene),
            GeometryHandle::new(geometry),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn simulate_single_point(
        &self,
        wi: Sph2,
        sector: &EmitterCircularSector<'_>,
        mesh: &MicroSurfaceMesh,
        geometry: &GeometryHandle,
        scene: &SceneHandle,
        #[cfg(not(feature = "vdbg"))] fresnel: bool,
        #[cfg(not(feature = "vdbg"))] iors_i: &[Ior],
        #[cfg(not(feature = "vdbg"))] iors_t: &[Ior],
    ) -> SingleSimResult {
        let geometry: &Arc<Geometry<'static>> = geometry
            .downcast_ref()
            .expect("geometry handle is Arc<embree::Geometry>");
        let scene: &Scene<'static> = scene.downcast_ref().expect("scene handle is embree::Scene");
        embr::simulate_bsdf_measurement_single_point(
            wi,
            sector,
            mesh,
            Arc::clone(geometry),
            scene,
            #[cfg(not(feature = "vdbg"))]
            fresnel,
            #[cfg(not(feature = "vdbg"))]
            iors_i,
            #[cfg(not(feature = "vdbg"))]
            iors_t,
        )
    }
}

/// Constructs the embree BSDF backend (CLI default path).
pub fn provide() -> Arc<dyn BsdfRtBackend> { Arc::new(EmbreeBackend) }

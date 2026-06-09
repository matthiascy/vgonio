//! Backend dispatch traits for the measurement capability.
//!
//! `vgonio-measurement` defines the [`BsdfRtBackend`] / [`GafBackend`] traits
//! plus opaque, `'static` handle newtypes; the concrete backends live in
//! `vgonio-measurement-{embree,wgpu,cuda}`, are selected app-side per build
//! features, and passed into `register_handlers`. The traits never reference a
//! backend crate, which is what avoids the Cargo cycle.

use std::{any::Any, sync::Arc};

use vgn_core::{math::Sph2, optics::Ior, res::Handle};
use vgn_io::MicroSurfaceMesh;

use crate::{
    bsdf::emitter::EmitterCircularSector, cache::ComputeCache, measurement::Measurement,
    mfd::params::GafMeasurementParams,
};

/// Per-direction simulation result, defined in [`crate::bsdf`]; re-exported so
/// backend impls can name it as `backend::SingleSimResult`.
pub use crate::bsdf::SingleSimResult;

/// Opaque, `'static` device handle wrapping a backend-specific type (the embree
/// backend stores `embree::Device`). All embree handle types are `Send + Sync`
/// upstream, so `Box<dyn Any + Send + Sync>` needs no vgonio-side assertion.
pub struct DeviceHandle(Box<dyn Any + Send + Sync>);
/// Opaque, `'static` scene handle (embree stores `Scene<'static>`).
pub struct SceneHandle(Box<dyn Any + Send + Sync>);
/// Opaque, `'static` geometry handle (embree stores `Arc<Geometry<'static>>`,
/// cloned per direction to match today's loop).
pub struct GeometryHandle(Box<dyn Any + Send + Sync>);

macro_rules! impl_handle {
    ($ty:ident) => {
        impl $ty {
            /// Wraps a backend value as an opaque handle.
            pub fn new<T: Any + Send + Sync>(v: T) -> Self { Self(Box::new(v)) }
            /// Downcasts to the concrete backend type.
            pub fn downcast_ref<T: Any>(&self) -> Option<&T> { self.0.downcast_ref() }
        }
    };
}
impl_handle!(DeviceHandle);
impl_handle!(SceneHandle);
impl_handle!(GeometryHandle);

/// BSDF ray-tracing backend (Embree today; OptiX/CUDA later). A
/// [`NullBsdfBackend`] is registered when no backend is compiled in.
pub trait BsdfRtBackend: Send + Sync {
    /// Backend-owned scene construction. Returns `'static` opaque handles the
    /// orchestration passes back into [`Self::simulate_single_point`].
    fn create_resources(
        &self,
        mesh: &MicroSurfaceMesh,
    ) -> (DeviceHandle, SceneHandle, GeometryHandle);

    /// Per-incident-direction kernel. Mirrors today's
    /// `simulate_bsdf_measurement_single_point`.
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
    ) -> SingleSimResult;

    /// True for the null fallback backend; lets callers (e.g. the GUI) disable
    /// actions instead of calling and panicking. Default `false`.
    fn is_null(&self) -> bool { false }
}

/// GAF (masking-shadowing) backend (wgpu today).
pub trait GafBackend: Send + Sync {
    /// Computes the GAF for `handles`, mirroring today's
    /// `measure_masking_shadowing_function`.
    fn measure_masking_shadowing_function(
        &self,
        params: GafMeasurementParams,
        handles: &[Handle],
        cache: &ComputeCache,
    ) -> Box<[Measurement]>;

    /// True for the null fallback backend. Default `false`.
    fn is_null(&self) -> bool { false }
}

// ---- Null fallbacks: registered when the backend feature is off. They panic
// on first use; the executor's `catch_unwind` translates that into a JobError.

/// BSDF backend used when none is compiled in; panics on use.
pub struct NullBsdfBackend;
impl BsdfRtBackend for NullBsdfBackend {
    fn create_resources(
        &self,
        _mesh: &MicroSurfaceMesh,
    ) -> (DeviceHandle, SceneHandle, GeometryHandle) {
        panic!("no BSDF backend compiled in; rebuild vgonio-app --features embree")
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
        panic!("no BSDF backend compiled in; rebuild vgonio-app --features embree")
    }

    fn is_null(&self) -> bool { true }
}
/// Constructs the null BSDF backend.
pub fn null_bsdf_backend() -> Arc<dyn BsdfRtBackend> { Arc::new(NullBsdfBackend) }

/// GAF backend used when none is compiled in; panics on use.
pub struct NullGafBackend;
impl GafBackend for NullGafBackend {
    fn measure_masking_shadowing_function(
        &self,
        _params: GafMeasurementParams,
        _handles: &[Handle],
        _cache: &ComputeCache,
    ) -> Box<[Measurement]> {
        panic!("no GAF backend compiled in; rebuild vgonio-app --features wgpu")
    }

    fn is_null(&self) -> bool { true }
}
/// Constructs the null GAF backend.
pub fn null_gaf_backend() -> Arc<dyn GafBackend> { Arc::new(NullGafBackend) }

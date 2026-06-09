//! WGPU GAF (masking-shadowing) backend.
//!
//! Stub scaffold: real impl lands in later.

use std::sync::Arc;

use vgn_core::res::Handle;
use vgn_measurement::{
    backend::GafBackend, cache::ComputeCache, measurement::Measurement,
    params::GafMeasurementParams,
};

/// GAF backend holding its own GPU context. GAF's code uses `vgn_uxgx::gfx`
/// primitives (Camera/Projection/RenderPass/Texture all hang off `GpuContext`),
/// so the backend owns one. The GUI shares its renderer's context via
/// [`WgpuBackend::with_context`]; headless CLI runs use [`WgpuBackend::headless`].
/// (Q20: port GAF off `vgn_uxgx` so this can be `Arc<wgpu::Device>` instead.)
pub struct WgpuBackend {
    #[allow(dead_code)]
    ctx: Arc<vgn_uxgx::gfx::context::GpuContext>,
}

impl WgpuBackend {
    /// Constructs a compute-only context for headless CLI runs.
    pub fn headless() -> Self { todo!("construct a compute-only GpuContext in Task 2.8") }

    /// Reuses an existing GPU context (the GUI shares its renderer's).
    pub fn with_context(ctx: Arc<vgn_uxgx::gfx::context::GpuContext>) -> Self { Self { ctx } }
}

impl GafBackend for WgpuBackend {
    fn measure_masking_shadowing_function(
        &self,
        _params: GafMeasurementParams,
        _handles: &[Handle],
        _cache: &ComputeCache,
    ) -> Box<[Measurement]> {
        todo!("wgpu GAF backend impl lands in Task 2.8")
    }
}

/// Constructs the headless wgpu GAF backend (CLI default path). GUI callers use
/// [`WgpuBackend::with_context`] directly to share the renderer's `GpuContext`.
pub fn provide() -> Arc<dyn GafBackend> { Arc::new(WgpuBackend::headless()) }

#![feature(vec_push_within_capacity)]
//! WGPU GAF (masking-shadowing) backend.

mod gaf;

use std::sync::Arc;

use vgn_core::res::Handle;
use vgn_measurement::{
    backend::GafBackend, cache::ComputeCache, measurement::Measurement,
    params::GafMeasurementParams,
};

/// WGPU-backed GAF estimator. The rendering self-constructs an offscreen GPU
/// context per measurement (matching Phase 1 behavior), so the backend itself
/// holds no state.
pub struct WgpuBackend;

impl GafBackend for WgpuBackend {
    fn measure_masking_shadowing_function(
        &self,
        params: GafMeasurementParams,
        handles: &[Handle],
        cache: &ComputeCache,
    ) -> Box<[Measurement]> {
        gaf::measure_masking_shadowing_function(params, handles, cache)
    }
}

/// Constructs the wgpu GAF backend (CLI default path).
pub fn provide() -> Arc<dyn GafBackend> { Arc::new(WgpuBackend) }

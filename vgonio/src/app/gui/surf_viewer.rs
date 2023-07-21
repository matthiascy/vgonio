use vgcore::math::{Mat4, Vec4};

use crate::app::gui::{state::camera::CameraState, visual_grid::VisualGridState};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MicroSurfaceUniforms {
    /// The model matrix.
    pub model: Mat4,
    /// Extra information: lowest, highest, span, scale.
    pub info: Vec4,
}

impl MicroSurfaceUniforms {
    /// Returns the size of the uniforms in bytes, aligned to the device's
    /// `min_uniform_buffer_offset_alignment`.
    pub fn aligned_size(device: &wgpu::Device) -> u32 {
        let alignment = device.limits().min_uniform_buffer_offset_alignment;
        let size = std::mem::size_of::<MicroSurfaceUniforms>() as u32;
        let remainder = size % alignment;
        if remainder == 0 {
            size
        } else {
            size + alignment - remainder
        }
    }
}

/// Surface viewer.
pub struct SurfViewer {
    camera: CameraState,
    /// State of the visual grid rendering, including the pipeline, binding
    /// groups, and buffers.
    visual_grid_state: VisualGridState,
}

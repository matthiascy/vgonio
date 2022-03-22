pub mod camera;
mod context;
mod grid;
mod mesh;
mod shadow_pass;
mod texture;
mod vertex;

use crate::Error;
pub use context::GpuContext;
use glam::Mat4;
pub use grid::*;
pub use mesh::MeshView;
pub use shadow_pass::ShadowPass;
use std::num::NonZeroU32;
pub use texture::Texture;
pub use vertex::*;
use wgpu::util::DeviceExt;
use wgpu::{BufferSize, ShaderSource};

/// Represents a rendering pipeline and its associated bind group (shader
/// input).
pub struct RdrPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub uniform_buffer: Option<wgpu::Buffer>,
}

pub const DEFAULT_BIND_GROUP_LAYOUT_DESC: wgpu::BindGroupLayoutDescriptor =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("default_vertex_stage_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    };

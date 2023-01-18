mod buffers;
pub mod camera;
mod context;
mod grid;
mod mesh;
mod shadow_pass;
mod texture;
mod vertex;

pub use buffers::*;

pub use context::{GpuContext, WgpuConfig};
pub use grid::*;
pub use mesh::*;
pub use shadow_pass::ShadowPass;
pub use texture::Texture;
pub use vertex::*;

pub use mesh::RenderableMesh;

/// Enum for selecting the right buffer type.
#[derive(Debug)]
pub enum BufferType {
    Uniform,
    Index,
    Vertex,
}

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

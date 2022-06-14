pub mod camera;
mod context;
mod grid;
mod mesh;
mod shadow_pass;
mod texture;
mod vertex;

pub use context::GpuContext;
pub use grid::*;
pub use mesh::*;
pub use shadow_pass::ShadowPass;
pub use texture::Texture;
pub use vertex::*;
use wgpu::util::DeviceExt;

/// Wrap a [`wgpu::Buffer`] and include additional size information.
#[derive(Debug)]
pub struct SizedBuffer {
    pub raw: wgpu::Buffer,
    pub size: usize,
}

impl SizedBuffer {
    pub fn new(device: &wgpu::Device, size: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        Self { raw, size }
    }

    pub fn new_init(device: &wgpu::Device, contents: &[u8], usage: wgpu::BufferUsages, label: Option<&str>) -> Self {
        let size = contents.len();
        let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label, contents, usage });
        Self { raw, size }
    }
}

/// Represents a rendering pipeline and its associated bind group (shader
/// input).
pub struct RdrPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub uniform_buffer: Option<wgpu::Buffer>,
}

pub const DEFAULT_BIND_GROUP_LAYOUT_DESC: wgpu::BindGroupLayoutDescriptor = wgpu::BindGroupLayoutDescriptor {
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

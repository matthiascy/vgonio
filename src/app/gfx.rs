mod buffers;
pub mod camera;
mod context;
mod grid;
mod mesh;
mod shadow_pass;
mod texture;
mod vertex;

pub use buffers::*;
use wgpu::TextureFormat;

pub use context::{GpuContext, WgpuConfig, WindowSurface};
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
pub struct RenderPass {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub uniform_buffers: Option<Vec<wgpu::Buffer>>,
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

pub fn remap_depth(depth: f32, near: f32, far: f32) -> f32 {
    linearize_depth(depth, near, far) / (far - near)
}

pub fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    (2.0 * near * far) / (far + near - depth * (far - near))
}

/// Returns the texture format's pixel size in bytes.
pub const fn tex_fmt_bpp(format: wgpu::TextureFormat) -> u32 {
    match format {
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint
        | TextureFormat::Stencil8 => 1,
        TextureFormat::R16Uint
        | TextureFormat::R16Sint
        | TextureFormat::R16Unorm
        | TextureFormat::R16Snorm
        | TextureFormat::R16Float
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint => 2,
        TextureFormat::R32Uint
        | TextureFormat::R32Sint
        | TextureFormat::R32Float
        | TextureFormat::Rg16Uint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rg16Unorm
        | TextureFormat::Rg16Snorm
        | TextureFormat::Rg16Float
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rgb9e5Ufloat
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rg11b10Float => 4,
        TextureFormat::Rg32Uint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rg32Float
        | TextureFormat::Rgba16Uint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba16Unorm
        | TextureFormat::Rgba16Snorm
        | TextureFormat::Rgba16Float => 8,
        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => 16,
        TextureFormat::Depth16Unorm => 2,
        TextureFormat::Depth24Plus => 4,
        TextureFormat::Depth24PlusStencil8 => 4,
        TextureFormat::Depth32Float => 4,
        TextureFormat::Depth32FloatStencil8
        | TextureFormat::Bc1RgbaUnorm
        | TextureFormat::Bc1RgbaUnormSrgb
        | TextureFormat::Bc2RgbaUnorm
        | TextureFormat::Bc2RgbaUnormSrgb
        | TextureFormat::Bc3RgbaUnorm
        | TextureFormat::Bc3RgbaUnormSrgb => 16,
        TextureFormat::Bc4RUnorm => 8,
        TextureFormat::Bc4RSnorm => 8,
        TextureFormat::Bc5RgUnorm
        | TextureFormat::Bc5RgSnorm
        | TextureFormat::Bc6hRgbUfloat
        | TextureFormat::Bc6hRgbSfloat
        | TextureFormat::Bc7RgbaUnorm
        | TextureFormat::Bc7RgbaUnormSrgb => 16,
        TextureFormat::Etc2Rgb8Unorm
        | TextureFormat::Etc2Rgb8UnormSrgb
        | TextureFormat::Etc2Rgb8A1Unorm
        | TextureFormat::Etc2Rgb8A1UnormSrgb => 8,
        TextureFormat::Etc2Rgba8Unorm => 16,
        TextureFormat::Etc2Rgba8UnormSrgb => 16,
        TextureFormat::EacR11Unorm => 8,
        TextureFormat::EacR11Snorm => 8,
        TextureFormat::EacRg11Unorm => 16,
        TextureFormat::EacRg11Snorm => 16,
        TextureFormat::Astc { .. } => 16,
    }
}

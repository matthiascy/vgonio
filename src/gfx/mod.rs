pub mod camera;
mod context;
mod grid;
mod mesh;
mod texture;
mod vertex;

pub use context::GpuContext;
pub use grid::*;
pub use mesh::MeshView;
pub use texture::Texture;
pub use vertex::*;
use wgpu::ShaderSource;

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

pub struct DepthPass {
    pub pass: RdrPass,
    pub depth_texture: Texture,
}

pub struct DepthPassUniforms {
    pub model_matrix: [f32; 16],
    pub light_space_matrix: [f32; 16],
}

impl DepthPass {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let uniform_buffer_size = std::mem::size_of::<DepthPassUniforms>() as wgpu::BufferAddress;
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(uniform_buffer_size),
                        },
                        count: None,
                    }],
                });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("depth_pass_pipeline"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: uniform_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let shader_module = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../app/assets/shaders/wgsl/depth_map.wgsl").into(),
                ),
            });
        let depth_texture =
            Texture::create_depth_texture(&ctx.device, width, height, "depth_pass_depth_texture");
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth_pass_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: wgpu::VertexFormat::Float32x3.size(),
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2, // corresponds to bi-linear filtering
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                fragment: None,
                multiview: None,
            });
        let pass = RdrPass {
            pipeline,
            bind_groups: vec![bind_group],
            uniform_buffer: Some(uniform_buffer),
        };

        Self {
            pass,
            depth_texture,
        }
    }
}

use crate::app::{
    gfx::{
        camera::{Camera, Projection},
        GpuContext, Texture, Vertex,
    },
    gui::{
        state::{GuiContext, GuiRenderer},
        VgonioEvent, VgonioEventLoop,
    },
};
use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};
use wgpu::util::DeviceExt;
use winit::event_loop::EventLoopProxy;

pub const SHADER: &str = r#"
@group(0) @binding(0)
var<uniform> mvp: mat4x4<f32>;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return mvp * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}"#;

pub struct BsdfItem {
    pub id: egui::TextureId,
    pub texture: Texture,
    pub data: Option<wgpu::Buffer>,
}

pub struct BsdfViewer {
    gpu: Arc<GpuContext>,
    gui: Arc<RwLock<GuiRenderer>>,
    proj_view_model: glam::Mat4,
    sampler: Arc<wgpu::Sampler>,
    bsdf_items: Vec<BsdfItem>,
    depth_attachment: Texture,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    event_loop: VgonioEventLoop,
}

impl BsdfViewer {
    pub fn new(
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        format: wgpu::TextureFormat,
        event_loop: VgonioEventLoop,
    ) -> Self {
        let camera = Camera::new(
            glam::Vec3::new(2.0, 1.5, 2.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let projection = Projection::new(0.1, 100.0, 45.0, 256, 256);
        let proj_view_model = projection
            .matrix(crate::app::gfx::camera::ProjectionKind::Perspective)
            * camera.view_matrix();
        let sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bsdf-viewer-image-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        let depth_attachment = Texture::create_depth_texture(
            &gpu.device,
            256,
            256,
            None,
            Some(sampler.clone()),
            None,
            Some("sampling-debugger-depth-attachment"),
        );
        let uniform_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sampling-debugger-uniform-buffer"),
                contents: bytemuck::cast_slice(&proj_view_model.to_cols_array()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let uniform_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sampling-debugger-uniform-bind-group-layout"),
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
                });
        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bsdf-viewer-uniform-bind-group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bsdf-viewer-pipeline-layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bsdf-viewer-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bsdf-viewer-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::PointList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                    unclipped_depth: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        let vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bsdf-viewer-vertex-buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<Vertex>() as u64 * 1024,
            mapped_at_creation: false,
        });

        Self {
            gpu,
            gui,
            proj_view_model,
            sampler,
            bsdf_items: vec![],
            depth_attachment,
            uniform_buffer,
            uniform_bind_group,
            pipeline,
            vertex_buffer,
            event_loop,
        }
    }

    pub fn create_new_view(&mut self) -> egui::TextureId {
        let color_attachment = Texture::new(
            &self.gpu.device,
            &wgpu::TextureDescriptor {
                label: Some(&format!(
                    "bsdf-viewer-color-attachment-{}",
                    self.bsdf_items.len()
                )),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Some(self.sampler.clone()),
        );
        let color_attachment_id = self.gui.write().unwrap().register_native_texture(
            &self.gpu.device,
            &color_attachment.view,
            wgpu::FilterMode::Linear,
        );

        self.bsdf_items.push(BsdfItem {
            id: color_attachment_id,
            texture: color_attachment,
            data: None,
        });

        color_attachment_id
    }
}

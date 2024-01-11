use crate::app::{
    gfx::{
        camera::{Camera, Projection},
        GpuContext, Texture,
    },
    gui::state::GuiRenderer,
};
use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};
use vgcore::{
    math,
    math::{Mat4, Vec3},
};

use crate::app::gui::event::EventLoopProxy;

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

#[derive(Debug)]
pub struct BsdfRenderingRecord {
    pub enabled: bool,
    pub texture_id: egui::TextureId,
    pub texture: Texture,
    pub rotation: f32,
    pub data: Option<wgpu::Buffer>,
    pub vertex_count: u32,
}

pub struct BsdfViewer {
    gpu: Arc<GpuContext>,
    gui: Arc<RwLock<GuiRenderer>>,
    sampler: Arc<wgpu::Sampler>,
    records: Vec<BsdfRenderingRecord>,
    depth_attachment: Texture,
    /// The uniform buffer is used to pass the projection-view-model matrix to
    /// the shader. This contains the matrix for each view.
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    event_loop: EventLoopProxy,
    proj_view: Mat4,
    aligned_uniform_buffer_size: u32,
}

impl BsdfViewer {
    pub const OUTPUT_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub fn new(
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let camera = Camera::new(Vec3::new(2.0, 1.5, 2.0), Vec3::ZERO);
        let projection = Projection::new(0.1, 100.0, 45.0, 256, 256);
        let proj_view = projection.matrix(crate::app::gfx::camera::ProjectionKind::Perspective)
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
        let aligned_size = math::calc_aligned_size(
            std::mem::size_of::<Mat4>() as u32,
            gpu.device.limits().min_uniform_buffer_offset_alignment,
        );
        // Pre-allocate the uniform buffer for 8 views.
        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bsdf-viewer-uniform-buffer"),
            size: aligned_size as u64 * 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &uniform_buffer,
            0,
            bytemuck::cast_slice(&[proj_view.to_cols_array(); 8]),
        );
        let uniform_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sampling-debugger-uniform-bind-group-layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(aligned_size as u64),
                        },
                        count: None,
                    }],
                });
        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bsdf-viewer-uniform-bind-group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(aligned_size as u64),
                }),
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
                        array_stride: std::mem::size_of::<Vec3>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Self::OUTPUT_TEXTURE_FORMAT,
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

        Self {
            gpu,
            gui,
            proj_view,
            sampler,
            records: vec![],
            depth_attachment,
            uniform_buffer,
            uniform_bind_group,
            pipeline,
            event_loop,
            aligned_uniform_buffer_size: aligned_size,
        }
    }

    pub fn toggle_view(&mut self, id: egui::TextureId) {
        for record in self.records.iter_mut() {
            if record.texture_id == id {
                record.enabled = !record.enabled;
            }
        }
    }

    /// Update the buffer that contains the data for a specific view.
    pub fn update_bsdf_data_buffer(
        &mut self,
        id: egui::TextureId,
        buffer: Option<wgpu::Buffer>,
        count: u32,
    ) {
        match self.records.iter_mut().find(|v| v.texture_id == id) {
            None => {}
            Some(record) => {
                record.data = buffer;
                record.vertex_count = count;
            }
        }
    }

    pub fn rotate(&mut self, id: egui::TextureId, angle: f32) {
        match self.records.iter().position(|v| v.texture_id == id) {
            None => {}
            Some(idx) => {
                self.records[idx].rotation += angle;
                // Update the uniform buffer
                self.gpu.queue.write_buffer(
                    &self.uniform_buffer,
                    self.aligned_uniform_buffer_size as u64 * idx as u64,
                    bytemuck::cast_slice(&[(self.proj_view
                        * Mat4::from_rotation_y(self.records[idx].rotation))
                    .to_cols_array()]),
                );
            }
        };
    }

    pub fn create_new_view(&mut self) -> egui::TextureId {
        log::debug!("Creating new view for BSDF viewer");
        let color_attachment = Texture::new(
            &self.gpu.device,
            &wgpu::TextureDescriptor {
                label: Some(&format!(
                    "bsdf-viewer-color-attachment-{}",
                    self.records.len()
                )),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::OUTPUT_TEXTURE_FORMAT,
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

        if self.records.len() > 8 {
            // Ensure that our uniform buffer is large enough to hold all the data
            let new_size = self.aligned_uniform_buffer_size * (self.records.len() + 1) as u32;
            self.uniform_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bsdf-viewer-uniform-buffer"),
                size: new_size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.uniform_bind_group =
                self.gpu
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bsdf-viewer-uniform-bind-group"),
                        layout: &self.pipeline.get_bind_group_layout(0),
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.uniform_buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new(
                                    self.aligned_uniform_buffer_size as u64,
                                ),
                            }),
                        }],
                    });
            // Copy the old data into the new buffer
            let mut encoder =
                self.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bsdf-viewer-uniform-buffer-copy"),
                    });
            encoder.copy_buffer_to_buffer(
                &self.uniform_buffer,
                0,
                &self.uniform_buffer,
                0,
                self.aligned_uniform_buffer_size as u64 * self.records.len() as u64,
            );
        }

        self.records.push(BsdfRenderingRecord {
            enabled: false,
            texture_id: color_attachment_id,
            texture: color_attachment,
            rotation: 0.0,
            data: None,
            vertex_count: 0,
        });

        color_attachment_id
    }

    pub fn render(&mut self) {
        let to_be_rendered = self
            .records
            .iter()
            .enumerate()
            .filter(|(_, record)| record.data.is_some() && record.enabled)
            .collect::<Vec<_>>();

        if to_be_rendered.is_empty() {
            return;
        }

        let cmd_buffers = to_be_rendered.iter().map(|(i, record)| {
            let mut encoder =
                self.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!(
                            "bsdf-viewer-command-encoder-{:?}",
                            record.texture_id
                        )),
                    });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bsdf-viewer-render-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &record.texture.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_attachment.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(
                    0,
                    &self.uniform_bind_group,
                    &[*i as u32 * self.aligned_uniform_buffer_size],
                );
                render_pass.set_vertex_buffer(0, record.data.as_ref().unwrap().slice(..));
                render_pass.draw(0..record.vertex_count, 0..1);
            }
            encoder.finish()
        });
        self.gpu.queue.submit(cmd_buffers);
    }
}

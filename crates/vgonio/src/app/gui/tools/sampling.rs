use egui::PointerButton;
use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};

use base::{
    math::{Mat4, Vec3},
    units::deg,
};
use gfxkit::{
    camera::{Camera, Projection, ProjectionKind},
    context::GpuContext,
    texture::Texture,
};
use wgpu::util::DeviceExt;

use crate::{
    app::gui::{
        event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
        state::GuiRenderer,
    },
    measure,
};

use super::Tool;

// TODO: use paint callback in the future

/// A tool that allows to inspect different sampling patterns.
///
/// Generated samples will be rendered into a texture in offscreen mode, and
/// then displays it on the screen by using directly the texture as an egui
/// image.
pub struct SamplingInspector {
    gpu: Arc<GpuContext>,
    proj_view_model: Mat4,
    pub color_attachment_id: egui::TextureId,
    pub color_attachment: Texture,
    pub depth_attachment: Texture,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub event_loop: EventLoopProxy,
    samples: Box<[Vec3]>,
    sample_count: u32,
    azimuth_min: f32, // in degrees
    azimuth_max: f32, // in degrees
    zenith_min: f32,  // in degrees
    zenith_max: f32,  // in degrees
}

impl Tool for SamplingInspector {
    fn name(&self) -> &'static str { "Sampling" }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        self.event_loop.send_event(VgonioEvent::Debugging(
            DebuggingEvent::ToggleSamplingRendering(*open),
        ));
        egui::Window::new(self.name()).open(open).show(ctx, |ui| {
            self.ui(ui);
        });
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Sample count");
        ui.add(
            egui::DragValue::new(&mut self.sample_count)
                .speed(1)
                .clamp_range(1..=u32::MAX),
        );
        // TODO: strict bounds
        ui.horizontal(|ui| {
            ui.label("Zenith range");
            ui.add(
                egui::DragValue::new(&mut self.zenith_min)
                    .speed(0.1)
                    .clamp_range(0.0..=180.0)
                    .suffix("°")
                    .prefix("min: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.zenith_max)
                    .speed(0.1)
                    .clamp_range(0.0..=180.0)
                    .suffix("°")
                    .prefix("max: "),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Azimuth range");
            ui.add(
                egui::DragValue::new(&mut self.azimuth_min)
                    .speed(0.1)
                    .clamp_range(0.0..=360.0)
                    .suffix("°")
                    .prefix("min: "),
            );
            ui.add(
                egui::DragValue::new(&mut self.azimuth_max)
                    .speed(0.1)
                    .clamp_range(0.0..=360.0)
                    .suffix("°")
                    .prefix("max: "),
            );
        });
        if ui.button("Unit Sphere").clicked() {
            self.samples = measure::uniform_sampling_on_unit_sphere(
                self.sample_count as usize,
                deg!(self.zenith_min).into(),
                deg!(self.zenith_max).into(),
                deg!(self.azimuth_min).into(),
                deg!(self.azimuth_max).into(),
            );
        }
        if ui.button("Unit Disk").clicked() {
            self.samples = measure::uniform_sampling_on_unit_disk(self.sample_count as usize)
        }
        let response = ui.add(
            egui::Image::new(egui::load::SizedTexture {
                id: self.color_attachment_id,
                size: [256.0, 256.0].into(),
            })
            .sense(egui::Sense::drag()),
        );
        if response.dragged_by(PointerButton::Primary) {
            let delta = response.drag_delta();
            self.proj_view_model *= Mat4::from_rotation_y(delta.x / 256.0 * std::f32::consts::PI)
                * Mat4::from_rotation_x(delta.y / 256.0 * std::f32::consts::PI);
            self.gpu.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[self.proj_view_model.to_cols_array()]),
            );
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

struct Vertex([f32; 3]);

impl SamplingInspector {
    pub fn new(
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        output_format: wgpu::TextureFormat,
        event_loop: EventLoopProxy,
    ) -> SamplingInspector {
        let camera = Camera::new(Vec3::new(2.0, 2.0, 1.5), Vec3::ZERO);
        let projection = Projection::new(0.1, 100.0, 45.0, 256, 256);
        let proj_view_model = projection.matrix(ProjectionKind::Perspective) * camera.view_matrix();
        let sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampling-debugger-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        let color_attachment = Texture::new(
            &gpu.device,
            &wgpu::TextureDescriptor {
                label: Some("sampling-debugger-color-attachment"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: output_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Some(sampler.clone()),
        );
        let color_attachment_id = gui.write().unwrap().register_native_texture(
            &gpu.device,
            &color_attachment.view,
            wgpu::FilterMode::Linear,
        );
        let depth_attachment = Texture::create_depth_texture(
            &gpu.device,
            256,
            256,
            None,
            Some(sampler),
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
            label: Some("sampling-debugger-uniform-bind-group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sampling-debugger-pipeline-layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sampling-debugger-shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sampling.wgsl"))),
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sampling-debugger-pipeline"),
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
                        format: output_format,
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
            label: Some("sampling-debugger-vertex-buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<Vertex>() as u64 * 1024,
            mapped_at_creation: false,
        });

        Self {
            gpu,
            color_attachment,
            depth_attachment,
            uniform_buffer,
            pipeline,
            uniform_bind_group,
            vertex_buffer,
            color_attachment_id,
            event_loop,
            azimuth_min: 0.0,
            azimuth_max: 360.0,
            zenith_min: 0.0,
            zenith_max: 180.0,
            sample_count: 1024,
            samples: Box::new([]),
            proj_view_model,
        }
    }

    pub fn record_render_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.samples.len() as u32;
        if count as u64 * std::mem::size_of::<Vertex>() as u64 > self.vertex_buffer.size() {
            self.vertex_buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sampling-debugger-vertex-buffer"),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                size: std::mem::size_of::<Vertex>() as u64 * count as u64,
                mapped_at_creation: false,
            });
        }
        self.gpu
            .queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.samples));
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sampling-debugger-render-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_attachment.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
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
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..count, 0..1);
    }

    pub fn render(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sampling-debugger-command-encoder"),
            });
        self.record_render_pass(&mut encoder);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}

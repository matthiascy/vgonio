use std::{borrow::Cow, sync::Arc};

use wgpu::util::DeviceExt;
use winit::event_loop::EventLoopProxy;

use crate::app::{
    gfx::{
        camera::{Camera, Projection},
        GpuContext, Texture,
    },
    gui::{state::GuiRenderer, VgonioEvent},
};

use super::Tool;

// TODO: use paint callback in the future

/// A debugger for sampling debugging.
/// It renders the samples into a texture (offline mode) and then displays it on
/// the screen.
pub struct SamplingDebugger {
    pub color_attachment_id: egui::TextureId,
    pub color_attachment: Texture,
    pub depth_attachment: Texture,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub event_loop: EventLoopProxy<VgonioEvent>,
    sample_count: u32,
    azimuth_min: f32, // in degrees
    azimuth_max: f32, // in degrees
    zenith_min: f32,  // in degrees
    zenith_max: f32,  // in degrees
}

impl Tool for SamplingDebugger {
    fn name(&self) -> &'static str { "Sampling Debugger" }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
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
            self.event_loop
                .send_event(VgonioEvent::UpdateSamplingDebugger {
                    count: self.sample_count,
                    azimuth: (self.azimuth_min, self.azimuth_max),
                    zenith: (self.zenith_min, self.zenith_max),
                })
                .unwrap();
        }
        ui.add(egui::Image::new(self.color_attachment_id, [256.0, 256.0]));
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

struct Vertex([f32; 30]);

impl SamplingDebugger {
    pub fn new(
        gpu: &GpuContext,
        gui: &mut GuiRenderer,
        output_format: wgpu::TextureFormat,
        event_loop: EventLoopProxy<VgonioEvent>,
    ) -> SamplingDebugger {
        let camera = Camera::new(
            glam::Vec3::new(2.0, 1.5, 2.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let projection = Projection::new(0.1, 100.0, 45.0, 256, 256);
        let proj_mat = projection.matrix(crate::app::gfx::camera::ProjectionKind::Perspective);
        let view_mat = camera.view_matrix();
        let matrix = proj_mat * view_mat;
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
            },
            Some(sampler.clone()),
        );
        let color_attachment_id = gui.register_native_texture(
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
        // let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("sampling-debugger-uniform-buffer"),
        //     size: std::mem::size_of::<[f32; 16]>() as u64,
        //     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });
        let uniform_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sampling-debugger-uniform-buffer"),
                contents: bytemuck::cast_slice(&matrix.to_cols_array()),
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
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "sampling_debugger.wgsl"
                ))),
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
        }
    }

    pub fn record_render_pass(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        samples: &[glam::Vec3],
    ) {
        let count = samples.len() as u32;
        if count as u64 > self.vertex_buffer.size() {
            self.vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sampling-debugger-vertex-buffer"),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                size: std::mem::size_of::<Vertex>() as u64 * count as u64,
                mapped_at_creation: false,
            });
        }
        gpu.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(samples));
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sampling-debugger-render-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_attachment.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_attachment.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..count, 0..1);
    }
}

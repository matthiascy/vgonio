pub(crate) mod camera;
mod context;
mod input;
mod renderer;

// TODO: create default config folder the first time the app is launched (gui
// and cli)

pub use context::GuiContext;
pub use input::InputState;
pub use renderer::GuiRenderer;

use crate::app::{
    gfx::{GpuContext, RenderPass, Texture, VertexLayout, WindowSurface},
    gui::VgonioEvent,
};

use crate::acq::Ray;

use crate::app::gfx::remap_depth;
use std::{default::Default, num::NonZeroU32, path::Path, sync::Arc};
use winit::{event::WindowEvent, event_loop::EventLoopWindowTarget, window::Window};

use super::EventResponse;

pub const AZIMUTH_BIN_SIZE_DEG: usize = 5;
pub const ZENITH_BIN_SIZE_DEG: usize = 2;
pub const AZIMUTH_BIN_SIZE_RAD: f32 = (AZIMUTH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
pub const ZENITH_BIN_SIZE_RAD: f32 = (ZENITH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
pub const NUM_AZIMUTH_BINS: usize = ((2.0 * std::f32::consts::PI) / AZIMUTH_BIN_SIZE_RAD) as _;
pub const NUM_ZENITH_BINS: usize = ((0.5 * std::f32::consts::PI) / ZENITH_BIN_SIZE_RAD) as _;

/// Uniform buffer used when rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct UiUniforms {
    /// Size of the window in logical pixels (points).
    logical_size: [f32; 2],
    /// Padding to align the struct to 16 bytes for the reason of the alignment
    /// requirement of the uniform buffer in WebGL.
    _padding: [f32; 2],
}

impl Default for UiUniforms {
    fn default() -> Self {
        Self {
            logical_size: [0.0, 0.0],
            _padding: [0.0, 0.0],
        }
    }
}

impl From<&ScreenDescriptor> for UiUniforms {
    fn from(desc: &ScreenDescriptor) -> Self {
        Self {
            logical_size: [desc.width as f32, desc.height as f32],
            _padding: [0.0, 0.0],
        }
    }
}

/// Information about the screen on which the UI is presented.
#[derive(Debug)]
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub width: u32,
    /// Height of the window in physical pixel.
    pub height: u32,
    /// HiDPI scale factor; pixels per point.
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    /// Returns the screen width in pixels.
    fn physical_size(&self) -> [u32; 2] { [self.width, self.height] }

    /// Returns the screen width in points.
    fn logical_size(&self) -> [f32; 2] {
        [
            self.width as f32 / self.scale_factor,
            self.height as f32 / self.scale_factor,
        ]
    }
}

// TODO: fix blending.

/// Stores the content of depth buffer.
/// In general the size of the depth map is equal to the size of the window.
/// Width will be recalculated when the window's bytes per row is not a multiple
/// of 256.
pub struct DepthMap {
    pub(crate) depth_attachment: Texture,
    pub(crate) depth_attachment_storage: wgpu::Buffer, // used to store depth attachment
    /// Manually padded width to make sure the bytes per row is a multiple of
    /// 256.
    pub(crate) width: u32,
}

impl DepthMap {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            None,
            None,
            None,
            Some("depth-texture"),
        );
        // Manually align the width to 256 bytes.
        let width = (width as f32 * 4.0 / 256.0).ceil() as u32 * 64;
        let depth_attachment_storage_size =
            (std::mem::size_of::<f32>() * (width * height) as usize) as wgpu::BufferAddress;
        let depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            depth_attachment,
            depth_attachment_storage,
            width,
        }
    }

    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        self.depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            None,
            None,
            None,
            Some("depth-texture"),
        );
        self.width = (width as f32 * 4.0 / 256.0).ceil() as u32 * 64;
        let depth_map_storage_size =
            (std::mem::size_of::<f32>() * (self.width * height) as usize) as wgpu::BufferAddress;
        self.depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_map_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    }

    pub fn copy_to_buffer(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                // texture: &self.depth_attachment.raw,
                texture: &self.depth_attachment.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                // buffer: &self.depth_attachment_storage,
                buffer: &self.depth_attachment_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(std::mem::size_of::<f32>() as u32 * self.width),
                    rows_per_image: NonZeroU32::new(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Save current depth buffer content to a PNG file.
    pub fn save_to_image(&mut self, ctx: &GpuContext, path: &Path, surface: &WindowSurface) {
        let mut image = image::GrayImage::new(self.width, surface.height());
        {
            let buffer_slice = self.depth_attachment_storage.slice(..);

            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });

            let buffer_view_f32 = buffer_slice.get_mapped_range();
            let data_u8 = unsafe {
                let (_, data, _) = buffer_view_f32.align_to::<f32>();
                data.iter()
                    .map(|d| (remap_depth(*d, 0.1, 100.0) * 255.0) as u8)
                    .collect::<Vec<u8>>()
            };

            image.copy_from_slice(&data_u8);
            image.save(path).unwrap();
        }
        self.depth_attachment_storage.unmap();
    }
}

pub struct DebugState {
    /// Vertex buffer storing all vertices.
    pub vert_buffer: wgpu::Buffer,

    pub vert_count: u32,

    // pub embree_rays: Vec<Ray>,
    pub rays: Vec<Ray>,

    /// Render pass for rays drawing.
    pub render_pass_rd: RenderPass,

    pub prim_pipeline: wgpu::RenderPipeline,
    pub prim_bind_group: wgpu::BindGroup,
    pub prim_uniform_buffer: wgpu::Buffer,
    pub prim_vert_buffer: wgpu::Buffer,
    pub ray_t: f32,
}

impl DebugState {
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        use wgpu::util::DeviceExt;
        let vert_layout = VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None);
        let vert_buffer_layout = vert_layout.buffer_layout(wgpu::VertexStepMode::Vertex);
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("debug-drawing-rays-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../gui/assets/shaders/wgsl/rays.wgsl").into(),
                ),
            });
        let prim_shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("debug-drawing-prim-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../gui/assets/shaders/wgsl/prim.wgsl").into(),
                ),
            });
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("debug-drawing-rays-bind-group-layout"),
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
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug-drawing-rays-uniform-buffer"),
                contents: bytemuck::cast_slice(&[0.0f32; 16 * 3 + 4]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let prim_uniform_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug-drawing-prim-uniform-buffer"),
                    contents: bytemuck::cast_slice(&[0.0f32; 16 * 3 + 4]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        Self {
            vert_buffer: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("debug-rays-vert-buffer"),
                size: std::mem::size_of::<f32>() as u64 * 1024, // initial capacity of 1024 rays
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            vert_count: 0,
            rays: vec![],
            render_pass_rd: RenderPass {
                pipeline: ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("debug-rays-pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: Some("debug-rays-pipeline-layout"),
                                bind_group_layouts: &[&bind_group_layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        vertex: wgpu::VertexState {
                            module: &shader_module,
                            entry_point: "vs_main",
                            buffers: &[vert_buffer_layout],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::LineStrip,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: Some(wgpu::Face::Back),
                            unclipped_depth: false,
                            polygon_mode: wgpu::PolygonMode::Line,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &shader_module,
                            entry_point: "fs_main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: target_format,
                                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview: None,
                    }),
                bind_groups: vec![ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("debug-rays-bind-group"),
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                })],
                uniform_buffer: Some(uniform_buffer),
            },
            prim_pipeline: ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("debug-prim-pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("debug-prim-pipeline-layout"),
                            bind_group_layouts: &[&bind_group_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    vertex: wgpu::VertexState {
                        module: &prim_shader_module,
                        entry_point: "vs_main",
                        buffers: &[VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None)
                            .buffer_layout(wgpu::VertexStepMode::Vertex)],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Line,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &prim_shader_module,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                }),
            prim_bind_group: ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("debug-prim-bind-group"),
                layout: &ctx
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("debug-drawing-prim-bind-group-layout"),
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
                    }),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prim_uniform_buffer.as_entire_binding(),
                }],
            }),
            prim_uniform_buffer,
            prim_vert_buffer: ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug-drawing-prim-vert-buffer"),
                    contents: bytemuck::cast_slice(&[0.0f32; std::mem::size_of::<f32>() * 3 * 3]),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
            ray_t: 10.0,
        }
    }

    pub fn update_uniform_buffer(&mut self, ctx: &GpuContext, uniform: &[f32; 16 * 3 + 4]) {
        ctx.queue.write_buffer(
            self.render_pass_rd.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(uniform),
        );
    }
}

/// Context for rendering the UI.
pub struct GuiState {
    //// The wgpu device.
    device: Arc<wgpu::Device>,
    //// The wgpu queue.
    queue: Arc<wgpu::Queue>,
    /// Context for GUI painting.
    context: GuiContext,
    /// Rendering state for the GUI.
    pub renderer: GuiRenderer, // TODO: make private
}

pub struct GuiRenderOutput {
    pub user_cmds: Vec<wgpu::CommandBuffer>,
    pub ui_cmd: wgpu::CommandBuffer,
}

impl GuiState {
    /// Creates related resources used for UI rendering.
    ///
    /// If the format passed is not a *Srgb format, the shader will
    /// automatically convert to sRGB colors in the shader.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        event_loop: &EventLoopWindowTarget<VgonioEvent>,
        msaa_samples: u32,
    ) -> Self {
        let context = GuiContext::new(event_loop);
        let renderer = GuiRenderer::new(&device, surface_format, None, msaa_samples);
        Self {
            device,
            queue,
            context,
            renderer,
        }
    }

    /// Prepares the rendering. This should be called before any rendering.
    pub fn update(&mut self, window: &Window) { self.context.prepare(window); }

    /// Returns the encapsulated GUI context.
    pub fn context(&self) -> &GuiContext { &self.context }

    /// Run the UI and record the rendering commands.
    pub fn render(
        &mut self,
        window: &Window,
        screen_desc: ScreenDescriptor,
        target: &wgpu::TextureView,
        ui: impl FnOnce(&egui::Context),
    ) -> GuiRenderOutput {
        let mut ui_cmd_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gui-command-encoder"),
                });
        let output = self.context.run(ui);
        self.context
            .handle_platform_output(window, output.platform_output);

        let primitives = self.context.inner.tessellate(output.shapes);

        let user_cmds = {
            for (id, image_delta) in &output.textures_delta.set {
                self.renderer
                    .update_texture(&self.device, &self.queue, *id, image_delta);
            }
            self.renderer.update_buffers(
                &self.device,
                &self.queue,
                &mut ui_cmd_encoder,
                &primitives,
                &screen_desc,
            )
        };

        {
            let mut render_pass = ui_cmd_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            self.renderer
                .render(&mut render_pass, &primitives, &screen_desc);
        }

        {
            for id in &output.textures_delta.free {
                self.renderer.remove_texture(*id);
            }
        }

        GuiRenderOutput {
            user_cmds,
            ui_cmd: ui_cmd_encoder.finish(),
        }
    }

    /// Update the context whenever there is a window event.
    pub fn on_event(&mut self, event: &WindowEvent) -> EventResponse {
        self.context.on_event(event).into()
    }
}

pub(crate) mod camera;
mod context;
pub(crate) mod debug;
mod renderer;

// TODO: create default config folder the first time the app is launched (gui
// and cli)

pub use context::RawGuiContext;
use egui_winit::EventResponse;
pub use renderer::GuiRenderer;

use gfxkit::{
    context::{GpuContext, ScreenDescriptor, WindowSurface},
    render_pass::remap_depth,
    texture::Texture,
};
use std::{
    ops::Deref,
    path::Path,
    sync::{Arc, RwLock},
};
use winit::{event::WindowEvent, window::Window};

pub const AZIMUTH_BIN_SIZE_DEG: usize = 5;
pub const ZENITH_BIN_SIZE_DEG: usize = 2;
pub const AZIMUTH_BIN_SIZE_RAD: f32 = (AZIMUTH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
pub const ZENITH_BIN_SIZE_RAD: f32 = (ZENITH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
pub const NUM_AZIMUTH_BINS: usize = ((2.0 * std::f32::consts::PI) / AZIMUTH_BIN_SIZE_RAD) as _;
pub const NUM_ZENITH_BINS: usize = ((0.5 * std::f32::consts::PI) / ZENITH_BIN_SIZE_RAD) as _;

// TODO: fix blending.

/// Stores the content of depth buffer.
/// In general the size of the depth map is equal to the size of the window.
/// Width will be recalculated when the window's bytes per row is not a multiple
/// of 256.
pub struct DepthMap {
    pub(crate) depth_attachment: Texture,
    /// Buffer storing depth attachment data.
    pub(crate) depth_attachment_storage: wgpu::Buffer,
    /// Manually padded width to make sure the bytes per row is a multiple of
    /// 256.
    pub(crate) width: u32,
}

impl Deref for DepthMap {
    type Target = Texture;

    fn deref(&self) -> &Self::Target { &self.depth_attachment }
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
                    bytes_per_row: Some(std::mem::size_of::<f32>() as u32 * self.width),
                    rows_per_image: Some(height),
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

/// Context for rendering the UI.
pub struct GuiContext {
    //// The wgpu device.
    device: Arc<wgpu::Device>,
    //// The wgpu queue.
    queue: Arc<wgpu::Queue>,
    /// Context for GUI painting.
    context: RawGuiContext,
    /// Rendering state for the GUI.
    pub renderer: Arc<RwLock<GuiRenderer>>, // TODO: make private
}

pub struct GuiRenderOutput {
    pub user_cmds: Vec<wgpu::CommandBuffer>,
    pub ui_cmd: wgpu::CommandBuffer,
}

impl GuiContext {
    /// Creates related resources used for UI rendering.
    ///
    /// If the format passed is not a *Srgb format, the shader will
    /// automatically convert to sRGB colors in the shader.
    pub fn new(
        window: &Window,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let context = RawGuiContext::new(window);
        let renderer = Arc::new(RwLock::new(GuiRenderer::new(
            &device,
            surface_format,
            None,
            msaa_samples,
        )));
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
    pub fn ctx(&self) -> &RawGuiContext { &self.context }

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

        let primitives = self
            .context
            .inner
            .tessellate(output.shapes, output.pixels_per_point);

        {
            let mut renderer = self.renderer.write().unwrap();
            let user_cmds = {
                for (id, image_delta) in &output.textures_delta.set {
                    renderer.update_texture(&self.device, &self.queue, *id, image_delta);
                }
                renderer.update_buffers(
                    &self.device,
                    &self.queue,
                    &mut ui_cmd_encoder,
                    &primitives,
                    &screen_desc,
                )
            };

            {
                let mut render_pass =
                    ui_cmd_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gui_render_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                renderer.render(&mut render_pass, &primitives, &screen_desc);
            }

            {
                for id in &output.textures_delta.free {
                    renderer.remove_texture(*id);
                }
            }

            GuiRenderOutput {
                user_cmds,
                ui_cmd: ui_cmd_encoder.finish(),
            }
        }
    }

    /// Update the context whenever there is a window event.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.context.on_window_event(window, event)
    }
}

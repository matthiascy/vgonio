use crate::app::gui::{
    event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
    state::GuiContext,
};
use egui::{Sense, TextureFilter, TextureWrapMode};
use gfxkit::{context::GpuContext, render_pass::remap_depth};

const IMG_WIDTH: usize = 480;
const IMG_HEIGHT: usize = 270;

pub(crate) struct DepthMapPane {
    /// The event loop proxy used to send events to the main event loop.
    event_loop: EventLoopProxy,
    depth_map_image: image::RgbaImage,
    depth_map_handle: Option<egui::TextureHandle>,
}

impl DepthMapPane {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            event_loop,
            depth_map_handle: None,
            depth_map_image: image::RgbaImage::new(IMG_WIDTH as _, IMG_HEIGHT as _),
        }
    }

    pub fn update_depth_map(
        &mut self,
        gpu_ctx: &GpuContext,
        gui_state: &GuiContext,
        buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let mut image = image::RgbaImage::new(width, height);
        {
            let buffer_slice = buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            gpu_ctx.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });

            let buffer_view_f32 = buffer_slice.get_mapped_range();
            let data_u8 = unsafe {
                let (_, data, _) = buffer_view_f32.align_to::<f32>();
                data.iter()
                    .flat_map(|d| {
                        let c = (remap_depth(*d, 0.1, 100.0) * 255.0) as u8;
                        [c, c, c, 255]
                    })
                    .collect::<Vec<u8>>()
            };

            image.copy_from_slice(&data_u8);
        }
        buffer.unmap();
        self.depth_map_image = image::imageops::resize(
            &image,
            IMG_WIDTH as _,
            IMG_HEIGHT as _,
            image::imageops::FilterType::CatmullRom,
        );
        self.depth_map_handle = Some(gui_state.ctx().load_texture(
            "depth_map",
            egui::ColorImage::from_rgba_unmultiplied(
                [IMG_WIDTH, IMG_HEIGHT],
                self.depth_map_image.as_flat_samples().as_slice(),
            ),
            egui::TextureOptions {
                minification: TextureFilter::Nearest,
                magnification: TextureFilter::Nearest,
                wrap_mode: TextureWrapMode::MirroredRepeat,
            },
        ));
    }
}

impl egui::Widget for &mut DepthMapPane {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        if ui.button("Update").clicked() {
            self.event_loop
                .send_event(VgonioEvent::Debugging(DebuggingEvent::UpdateDepthMap));
        }

        if let Some(handle) = &self.depth_map_handle {
            ui.image(egui::load::SizedTexture {
                id: handle.id(),
                size: handle.size_vec2(),
            })
        } else {
            let (_, response) = ui.allocate_exact_size(
                egui::vec2(IMG_WIDTH as f32, IMG_HEIGHT as f32),
                Sense::click(),
            );
            response
        }
    }
}

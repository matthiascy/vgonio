use crate::app::gui::{GuiContext, VgonioEvent};
use crate::app::state::remap_depth;
use crate::gfx::GpuContext;
use egui::Sense;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

const IMG_WIDTH: usize = 480;
const IMG_HEIGHT: usize = 270;

pub(crate) struct ShadowMapPane {
    /// The event loop proxy used to send events to the main event loop.
    evlp: Arc<EventLoopProxy<VgonioEvent>>,
    depth_map_image: image::RgbaImage,
    depth_map_handle: Option<egui::TextureHandle>,
}

impl ShadowMapPane {
    pub fn new(evlp: Arc<EventLoopProxy<VgonioEvent>>) -> Self {
        Self {
            evlp,
            depth_map_handle: None,
            depth_map_image: image::RgbaImage::new(IMG_WIDTH as _, IMG_HEIGHT as _),
        }
    }

    pub fn update_depth_map(
        &mut self,
        gpu_ctx: &GpuContext,
        gui_ctx: &GuiContext,
        buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let mut image = image::RgbaImage::new(width, height);
        {
            let buffer_slice = buffer.slice(..);

            let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
            gpu_ctx.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async { mapping.await.unwrap() });

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
        self.depth_map_handle = Some(gui_ctx.egui_context().load_texture(
            "depth_map",
            egui::ColorImage::from_rgba_unmultiplied(
                [IMG_WIDTH, IMG_HEIGHT],
                self.depth_map_image.as_flat_samples().as_slice(),
            ),
        ));
    }
}

impl egui::Widget for &mut ShadowMapPane {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        if ui.button("Update").clicked() {
            self.evlp.send_event(VgonioEvent::UpdateDepthMap).unwrap();
        }

        if let Some(handle) = &self.depth_map_handle {
            ui.image(handle, handle.size_vec2())
        } else {
            let (_, response) = ui.allocate_exact_size(
                egui::vec2(IMG_WIDTH as f32, IMG_HEIGHT as f32),
                Sense::click(),
            );
            response
        }
    }
}

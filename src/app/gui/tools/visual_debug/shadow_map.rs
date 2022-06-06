use crate::app::gui::VgonioEvent;
use crate::app::state::remap_depth;
use crate::gfx::GpuContext;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

const IMG_WIDTH: usize = 480;
const IMG_HEIGHT: usize = 270;

pub(crate) struct ShadowMapPane {
    /// The event loop proxy used to send events to the main event loop.
    evlp: Arc<EventLoopProxy<VgonioEvent>>,
    depth_map_image: image::RgbaImage,
    depth_map_handle: Option<egui::TextureHandle>,
    depth_map_updated: bool,
}

impl ShadowMapPane {
    pub fn new(evlp: Arc<EventLoopProxy<VgonioEvent>>) -> Self {
        Self {
            evlp,
            depth_map_handle: None,
            depth_map_image: image::RgbaImage::new(IMG_WIDTH as _, IMG_HEIGHT as _),
            depth_map_updated: true,
        }
    }

    pub fn update_depth_map(&mut self, ctx: &GpuContext, buffer: &wgpu::Buffer) {
        let mut image = image::RgbaImage::new(ctx.surface_config.width, ctx.surface_config.height);
        {
            let buffer_slice = buffer.slice(..);

            let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
            ctx.device.poll(wgpu::Maintain::Wait);
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
        self.depth_map_updated = true;
    }
}

impl egui::Widget for &mut ShadowMapPane {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        if ui.button("Update").clicked() {
            self.evlp.send_event(VgonioEvent::UpdateDepthMap).unwrap();
        }

        if self.depth_map_updated {
            self.depth_map_handle = Some(ui.ctx().load_texture(
                "depth_map",
                egui::ColorImage::from_rgba_unmultiplied(
                    [IMG_WIDTH, IMG_HEIGHT],
                    self.depth_map_image.as_flat_samples().as_slice(),
                ),
            ));
        };

        let texture = self.depth_map_handle.as_ref().unwrap();

        ui.image(texture, texture.size_vec2())
    }
}

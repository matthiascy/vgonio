use crate::app::gui::UserEvent;
use egui::{Response, Ui};
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

const IMG_WIDTH: usize = 360;
const IMG_HEIGHT: usize = 203;

// TODO: adaptive image size

pub struct VisualDebugger {
    pub(crate) opened_panel: PanelKind,
    pub(crate) shadow_map_panel: ShadowMapPanel,
}

impl VisualDebugger {
    pub fn new(evlp: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            opened_panel: Default::default(),
            shadow_map_panel: ShadowMapPanel::new(evlp),
        }
    }

    pub const fn name(&self) -> &'static str {
        "Visual Debugger"
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.opened_panel, PanelKind::ShadowMap, "Shadow Map");
        });
        ui.separator();

        match self.opened_panel {
            PanelKind::ShadowMap => {
                ui.add(&mut self.shadow_map_panel);
            }
        }
    }
}

#[derive(Eq, PartialEq)]
pub enum PanelKind {
    ShadowMap,
}

impl Default for PanelKind {
    fn default() -> Self {
        Self::ShadowMap
    }
}

pub struct ShadowMapPanel {
    /// The event loop proxy used to send events to the main event loop.
    evlp: Arc<EventLoopProxy<UserEvent>>,
    shadow_map: Option<egui::TextureHandle>,
}

impl ShadowMapPanel {
    pub fn new(evlp: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            evlp,
            shadow_map: None,
        }
    }
}

impl egui::Widget for &mut ShadowMapPanel {
    fn ui(self, ui: &mut Ui) -> Response {
        if ui.button("Save depth map").clicked() {
            self.evlp.send_event(UserEvent::SaveDepthMap).unwrap();
        }

        let texture: &egui::TextureHandle = self.shadow_map.get_or_insert_with(|| {
            let image =
                image::load_from_memory(include_bytes!("../../assets/damascus001.jpg")).unwrap();
            let image_buffer = image.to_rgba8();
            let resized = image::imageops::resize(
                &image_buffer,
                IMG_WIDTH as _,
                IMG_HEIGHT as _,
                image::imageops::FilterType::CatmullRom,
            );
            let pixels = resized.as_flat_samples();
            ui.ctx().load_texture(
                "damascus",
                egui::ColorImage::from_rgba_unmultiplied(
                    [IMG_WIDTH, IMG_HEIGHT],
                    pixels.as_slice(),
                ),
            )
        });

        // Show the image
        ui.add(egui::Image::new(texture, texture.size_vec2()))
        // ui.image(texture, texture.size_vec2()); // shorter
        //ui.image(self.depth_map_id, [IMG_WIDTH as f32, IMG_HEIGHT as f32])
    }
}

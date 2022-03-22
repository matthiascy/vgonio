use egui::{Response, Ui};

const IMG_WIDTH: usize = 360;
const IMG_HEIGHT: usize = 203;

// TODO: adaptive image size

pub struct VisualDebugger {
    pub(crate) opened_panel: DebuggerPanel,
    pub(crate) shadow_map_panel: ShadowMapPanel,
}

#[derive(Eq, PartialEq)]
pub enum DebuggerPanel {
    ShadowMap,
}

impl Default for DebuggerPanel {
    fn default() -> Self {
        Self::ShadowMap
    }
}

pub struct ShadowMapPanel {
    shadow_map: Option<egui::TextureHandle>,
}

impl egui::Widget for &mut ShadowMapPanel {
    fn ui(self, ui: &mut Ui) -> Response {
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
    }
}

impl VisualDebugger {
    pub fn new() -> Self {
        Self {
            opened_panel: Default::default(),
            shadow_map_panel: ShadowMapPanel { shadow_map: None },
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
            ui.selectable_value(
                &mut self.opened_panel,
                DebuggerPanel::ShadowMap,
                "Shadow Map",
            );
        });
        ui.separator();

        match self.opened_panel {
            DebuggerPanel::ShadowMap => {
                ui.add(&mut self.shadow_map_panel);
            }
        }
    }
}

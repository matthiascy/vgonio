use crate::app::gui::UserEvent;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

mod ray_tracing;
mod shadow_map;

use ray_tracing::RayTracingPane;
use shadow_map::ShadowMapPane;

#[non_exhaustive]
#[derive(Eq, PartialEq)]
enum PaneKind {
    ShadowMap,
    RayTracing,
}

impl Default for PaneKind {
    fn default() -> Self {
        Self::ShadowMap
    }
}

// TODO: adaptive image size
pub(crate) struct VisualDebugTool {
    opened: PaneKind,
    pub(crate) shadow_map_pane: ShadowMapPane,
    pub(crate) ray_tracing_pane: RayTracingPane,
}

impl VisualDebugTool {
    pub fn new(evlp: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            opened: Default::default(),
            shadow_map_pane: ShadowMapPane::new(evlp),
            ray_tracing_pane: RayTracingPane::default(),
        }
    }

    pub const fn name(&self) -> &'static str {
        "Visual Debugger"
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.opened, PaneKind::ShadowMap, "Shadow Map");
            ui.selectable_value(&mut self.opened, PaneKind::RayTracing, "Ray Tracing");
        });
        ui.separator();

        match self.opened {
            PaneKind::ShadowMap => {
                ui.add(&mut self.shadow_map_pane);
            }
            PaneKind::RayTracing => {
                ui.add(&mut self.ray_tracing_pane);
            }
        }
    }
}

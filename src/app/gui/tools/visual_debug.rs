use crate::app::gui::VgonioEvent;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

mod ray_tracing;
mod shadow_map;

use crate::app::gui::widgets::{toggle, toggle_ui};
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
    debug_drawing_enabled: bool,
    event_loop: Arc<EventLoopProxy<VgonioEvent>>,
    pub(crate) shadow_map_pane: ShadowMapPane,
    pub(crate) ray_tracing_pane: RayTracingPane,
}

impl<'surface> VisualDebugTool {
    pub fn new(evlp: Arc<EventLoopProxy<VgonioEvent>>) -> Self {
        Self {
            opened: Default::default(),
            debug_drawing_enabled: true,
            event_loop: evlp.clone(),
            shadow_map_pane: ShadowMapPane::new(evlp.clone()),
            ray_tracing_pane: RayTracingPane::new(evlp),
        }
    }

    pub const fn name(&self) -> &'static str {
        "Visual Debug"
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Debug Draw");
            if toggle_ui(ui, &mut self.debug_drawing_enabled).changed()
                && self.event_loop.send_event(VgonioEvent::ToggleDebugDrawing).is_err()
            {
                log::warn!("Failed to send VgonioEvent::ToggleDebugDrawing");
            }
        });

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

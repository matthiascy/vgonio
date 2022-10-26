use crate::app::gui::VgonioEvent;
use egui::Response;
use std::{any::Any, sync::Arc};
use winit::event_loop::EventLoopProxy;

mod ray_tracing;
mod shadow_map;

use crate::app::gui::{tools::Tool, widgets::toggle_ui};
use ray_tracing::RayTracingPane;
use shadow_map::ShadowMapPane;

#[non_exhaustive]
#[derive(Eq, PartialEq)]
enum PaneKind {
    ShadowMap,
    RayTracing,
}

impl Default for PaneKind {
    fn default() -> Self { Self::ShadowMap }
}

// TODO: adaptive image size
pub(crate) struct VisualDebugger {
    opened_pane: PaneKind,
    debug_drawing_enabled: bool,
    event_loop: EventLoopProxy<VgonioEvent>,
    pub(crate) shadow_map_pane: ShadowMapPane,
    pub(crate) ray_tracing_pane: RayTracingPane,
}

impl Tool for VisualDebugger {
    fn name(&self) -> &'static str { "Visual Debugger" }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Debug Draw");
            if toggle_ui(ui, &mut self.debug_drawing_enabled).changed()
                && self
                    .event_loop
                    .send_event(VgonioEvent::ToggleDebugDrawing)
                    .is_err()
            {
                log::warn!("Failed to send VgonioEvent::ToggleDebugDrawing");
            }
        });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.opened_pane, PaneKind::ShadowMap, "Shadow Map");
            ui.selectable_value(&mut self.opened_pane, PaneKind::RayTracing, "Ray Tracing");
        });
        ui.separator();

        match self.opened_pane {
            PaneKind::ShadowMap => {
                ui.add(&mut self.shadow_map_pane);
            }
            PaneKind::RayTracing => {
                ui.add(&mut self.ray_tracing_pane);
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

impl VisualDebugger {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>) -> Self {
        Self {
            opened_pane: Default::default(),
            debug_drawing_enabled: true,
            event_loop: event_loop.clone(),
            shadow_map_pane: ShadowMapPane::new(event_loop.clone()),
            ray_tracing_pane: RayTracingPane::new(event_loop),
        }
    }
}

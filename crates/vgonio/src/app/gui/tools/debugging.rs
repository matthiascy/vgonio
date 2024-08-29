use crate::app::gui::event::{DebuggingEvent, EventLoopProxy, VgonioEvent};
use std::{any::Any, default::Default};

mod microfacet;
mod shadow_map;

use crate::app::gui::tools::Tool;
use microfacet::MicrofacetDebugging;
use shadow_map::DepthMapPane;
use uxtk::widgets::ToggleSwitch;

#[non_exhaustive]
#[derive(Eq, PartialEq)]
enum PaneKind {
    ShadowMap,
    Microfacet,
}

impl Default for PaneKind {
    fn default() -> Self { Self::ShadowMap }
}

// TODO: adaptive image size
pub(crate) struct DebuggingInspector {
    opened_pane: PaneKind,
    pub debug_drawing_enabled: bool,
    event_loop: EventLoopProxy,
    pub(crate) depth_map_pane: DepthMapPane,
    pub(crate) microfacet_debugging: MicrofacetDebugging,
}

// TODO: offline rendering or egui paint function for shadow map

impl Tool for DebuggingInspector {
    fn name(&self) -> &'static str { "Debugging" }

    fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Debug Draw");
            if ui
                .add(ToggleSwitch::new(&mut self.debug_drawing_enabled))
                .changed()
            {
                self.event_loop.send_event(VgonioEvent::Debugging(
                    DebuggingEvent::ToggleDebugDrawing(self.debug_drawing_enabled),
                ));
            }
        });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.opened_pane, PaneKind::ShadowMap, "Shadow Map");
            ui.selectable_value(&mut self.opened_pane, PaneKind::Microfacet, "Microfacet");
        });
        ui.separator();

        match self.opened_pane {
            PaneKind::ShadowMap => {
                ui.add(&mut self.depth_map_pane);
            },
            PaneKind::Microfacet => {
                ui.add(&mut self.microfacet_debugging);
            },
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

impl DebuggingInspector {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            opened_pane: Default::default(),
            debug_drawing_enabled: false,
            event_loop: event_loop.clone(),
            depth_map_pane: DepthMapPane::new(event_loop.clone()),
            microfacet_debugging: MicrofacetDebugging::new(event_loop),
        }
    }
}

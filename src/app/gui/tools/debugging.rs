use crate::app::gui::{VgonioEvent, VgonioEventLoop};
use egui_toast::Toasts;
use std::{
    any::Any,
    default::Default,
    sync::{Arc, RwLock},
};
use winit::event_loop::EventLoopProxy;

mod brdf_measurement;
mod microfacet;
mod shadow_map;

use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{tools::Tool, widgets::ToggleSwitch},
    },
    msurf::MicroSurface,
};
use brdf_measurement::BrdfMeasurementPane;
use microfacet::MicrofacetMeasurementPane;
use shadow_map::ShadowMapPane;

#[non_exhaustive]
#[derive(Eq, PartialEq)]
enum PaneKind {
    ShadowMap,
    Brdf,
    Microfacet,
}

impl Default for PaneKind {
    fn default() -> Self { Self::ShadowMap }
}

// TODO: adaptive image size
pub(crate) struct DebuggingInspector {
    opened_pane: PaneKind,
    pub debug_drawing_enabled: bool,
    event_loop: VgonioEventLoop,
    pub(crate) shadow_map_pane: ShadowMapPane,
    pub(crate) brdf_pane: BrdfMeasurementPane,
    pub(crate) microfacet_pane: MicrofacetMeasurementPane,
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
            ui.add(ToggleSwitch::new(&mut self.debug_drawing_enabled));
        });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.opened_pane, PaneKind::ShadowMap, "Shadow Map");
            ui.selectable_value(&mut self.opened_pane, PaneKind::Brdf, "Ray Tracing");
            ui.selectable_value(&mut self.opened_pane, PaneKind::Microfacet, "Microfacet");
        });
        ui.separator();

        match self.opened_pane {
            PaneKind::ShadowMap => {
                ui.add(&mut self.shadow_map_pane);
            }
            PaneKind::Brdf => {
                ui.add(&mut self.brdf_pane);
            }
            PaneKind::Microfacet => {
                ui.add(&mut self.microfacet_pane);
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

impl DebuggingInspector {
    pub fn new(
        event_loop: VgonioEventLoop,
        toasts: Arc<RwLock<Toasts>>,
        cache: Arc<RwLock<Cache>>,
    ) -> Self {
        Self {
            opened_pane: Default::default(),
            debug_drawing_enabled: true,
            event_loop: event_loop.clone(),
            shadow_map_pane: ShadowMapPane::new(event_loop.clone()),
            brdf_pane: BrdfMeasurementPane::new(event_loop.clone(), toasts, cache),
            microfacet_pane: MicrofacetMeasurementPane::new(event_loop),
        }
    }

    pub fn update_surfaces(&mut self, surfaces: &[Handle<MicroSurface>], cache: &Cache) {
        for surface in surfaces {
            if !self
                .brdf_pane
                .loaded_surfaces
                .iter()
                .any(|s| s.surf == *surface)
            {
                let record = cache.get_micro_surface_record(*surface).unwrap();
                self.brdf_pane.loaded_surfaces.push(record.clone());
            }
        }
    }
}

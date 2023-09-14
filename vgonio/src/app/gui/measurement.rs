mod bsdf;
mod madf;
mod mmsf;

use crate::{
    app::{
        cache::{Handle, InnerCache},
        gui::{
            event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
            measurement::{
                bsdf::BsdfMeasurementTab, madf::AdfMeasurementTab, mmsf::MsfMeasurementTab,
            },
            widgets::{SurfaceSelector, ToggleSwitch},
        },
    },
    measure::params::MeasurementKind,
};
use egui::Widget;
use vgsurf::MicroSurface;

impl MeasurementKind {
    pub fn selectable_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Measurement kind: ");
            ui.selectable_value(self, MeasurementKind::Bsdf, "BSDF");
            ui.selectable_value(self, MeasurementKind::Adf, "ADF");
            ui.selectable_value(self, MeasurementKind::Msf, "MSF");
        });
    }
}

pub struct MeasurementDialog {
    kind: MeasurementKind,
    selector: SurfaceSelector,
    tab_bsdf: BsdfMeasurementTab,
    tab_adf: AdfMeasurementTab,
    tab_msf: MsfMeasurementTab,
    open: bool,
    event_loop: EventLoopProxy,
    #[cfg(debug_assertions)]
    debug: MeasurementDialogDebug,
}

impl MeasurementDialog {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        MeasurementDialog {
            kind: MeasurementKind::Bsdf,
            selector: SurfaceSelector::multiple(),
            tab_bsdf: BsdfMeasurementTab::new(event_loop.clone()),
            tab_adf: AdfMeasurementTab::new(event_loop.clone()),
            tab_msf: MsfMeasurementTab::new(event_loop.clone()),
            open: false,
            event_loop,
            #[cfg(debug_assertions)]
            debug: MeasurementDialogDebug {
                enable_debug_draw: false,
                surf_prim_id: 0,
                show_surf_prim: false,
                surface_viewers: vec![],
                focused_viewer: None,
            },
        }
    }

    pub fn update_surface_selector(&mut self, surfs: &[Handle<MicroSurface>], cache: &InnerCache) {
        self.selector.update(surfs, cache);
    }

    #[cfg(debug_assertions)]
    pub fn update_surface_viewers(&mut self, viewers: &[uuid::Uuid]) {
        for viewer in viewers {
            if !self.debug.surface_viewers.contains(viewer) {
                self.debug.surface_viewers.push(*viewer);
            }
        }
    }

    pub fn open(&mut self) { self.open = true; }

    pub fn show(&mut self, ctx: &egui::Context) {
        egui::Window::new("New Measurement")
            .open(&mut self.open)
            .fixed_size((400.0, 600.0))
            .show(ctx, |ui| {
                self.kind.selectable_ui(ui);
                #[cfg(debug_assertions)]
                {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Debug draw: ");
                        if ui
                            .add(ToggleSwitch::new(&mut self.debug.enable_debug_draw))
                            .changed()
                        {
                            self.event_loop
                                .send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleDebugDrawing(
                                        self.debug.enable_debug_draw,
                                    ),
                                ))
                                .unwrap();
                        }
                    });
                    if self.debug.enable_debug_draw {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Target viewer: ");
                            egui::ComboBox::new("brdf_measurement_debugging_selector", "")
                                .selected_text(match self.debug.focused_viewer {
                                    None => "Select a surface viewer".into(),
                                    Some(uuid) => {
                                        format!(
                                            "Viewer-{}",
                                            &uuid.to_string()[..6].to_ascii_uppercase()
                                        )
                                    }
                                })
                                .show_ui(ui, |ui| {
                                    for viewer in &self.debug.surface_viewers {
                                        ui.selectable_value(
                                            &mut self.debug.focused_viewer,
                                            Some(*viewer),
                                            format!(
                                                "Viewer-{}",
                                                &viewer.to_string()[..6].to_ascii_uppercase()
                                            ),
                                        );
                                    }
                                });
                        });
                    }
                }

                let first_selected = self.selector.first_selected();

                egui::CollapsingHeader::new("Specimen")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::Grid::new("specimen_grid").show(ui, |ui| {
                            ui.label("MicroSurfaces: ");
                            self.selector.ui("micro_surface_selector", ui);
                            ui.end_row();

                            #[cfg(debug_assertions)]
                            if self.debug.enable_debug_draw {
                                ui.label("Primitive ID: ");
                                ui.horizontal_wrapped(|ui| {
                                    ui.add(egui::DragValue::new(&mut self.debug.surf_prim_id));
                                    let prev_clicked = if ui.button("\u{25C0}").clicked() {
                                        self.debug.surf_prim_id =
                                            self.debug.surf_prim_id.max(1) - 1;
                                        true
                                    } else {
                                        false
                                    };

                                    let next_clicked = if ui.button("\u{25B6}").clicked() {
                                        self.debug.surf_prim_id =
                                            (self.debug.surf_prim_id + 1).min(usize::MAX);
                                        true
                                    } else {
                                        false
                                    };

                                    let toggle_changed = ui
                                        .add(ToggleSwitch::new(&mut self.debug.show_surf_prim))
                                        .changed();

                                    if prev_clicked || next_clicked || toggle_changed {
                                        self.event_loop
                                            .send_event(VgonioEvent::Debugging(
                                                DebuggingEvent::UpdateSurfacePrimitiveId {
                                                    surf: first_selected,
                                                    id: self.debug.surf_prim_id as u32,
                                                    status: self.debug.show_surf_prim,
                                                },
                                            ))
                                            .unwrap();
                                    }
                                });
                                ui.end_row();
                            }
                        });
                    });

                match self.kind {
                    MeasurementKind::Bsdf => self.tab_bsdf.ui(
                        ui,
                        #[cfg(debug_assertions)]
                        self.debug.enable_debug_draw,
                    ),
                    MeasurementKind::Adf => self.tab_adf.ui(ui),
                    MeasurementKind::Msf => self.tab_msf.ui(ui),
                }
                ui.separator();

                if ui
                    .button("Simulate")
                    .on_hover_text("Starts the measurement")
                    .clicked()
                {
                    // TODO: launch simulation on a separate thread and show progress bar
                    log::info!("Starting measurement");
                }
            });
    }
}

// if ui.button("Simulate").clicked() {
// // TODO: launch simulation on a separate thread and show progress bar
// self.event_loop
// .send_event(VgonioEvent::Measure(MeasureEvent::Madf {
// params: self.params,
// surfaces: self.selector.selected().collect(),
// }))
// .unwrap();
// }

// if ui.button("Simulate").clicked() {
// self.event_loop
// .send_event(VgonioEvent::Measure(MeasureEvent::Mmsf {
// params: self.params,
// surfaces: self.selector.selected().collect(),
// }))
// .unwrap();
// }

#[cfg(debug_assertions)]
pub struct MeasurementDialogDebug {
    enable_debug_draw: bool,
    surf_prim_id: usize,
    show_surf_prim: bool,
    surface_viewers: Vec<uuid::Uuid>,
    focused_viewer: Option<uuid::Uuid>,
}

mod bsdf;
mod madf;
mod mmsf;

use crate::{
    app::{
        args::OutputFormat,
        cache::{Cache, Handle, InnerCache},
        gui::{
            event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
            measurement::{
                bsdf::BsdfMeasurementTab, madf::AdfMeasurementTab, mmsf::MsfMeasurementTab,
            },
            misc,
            notify::NotifyKind,
            widgets::{SurfaceSelector, ToggleSwitch},
        },
    },
    measure::{
        bsdf::receiver::{DataRetrievalMode, ReceiverParams, ReceiverScheme},
        params::{MeasurementKind, MeasurementParams},
    },
    SphericalDomain,
};
use egui::Widget;
use vgcore::io::{CompressionScheme, FileEncoding};
use vgsurf::MicroSurface;

impl ReceiverParams {
    /// UI for detector parameters.
    pub fn ui<R>(
        &mut self,
        ui: &mut egui::Ui,
        add_contents: impl FnOnce(&mut ReceiverParams, &mut egui::Ui) -> R,
    ) {
        egui::CollapsingHeader::new("Detector")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("detector_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Precision:");
                        misc::drag_angle(&mut self.precision, "").ui(ui);
                        ui.end_row();

                        ui.label("Domain:");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(&mut self.domain, SphericalDomain::Upper, "Upper");
                            ui.selectable_value(&mut self.domain, SphericalDomain::Lower, "Lower");
                            ui.selectable_value(&mut self.domain, SphericalDomain::Whole, "Whole");
                        });
                        ui.end_row();

                        ui.label("Scheme:");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(
                                &mut self.scheme,
                                ReceiverScheme::Beckers,
                                "Beckers",
                            );
                            ui.selectable_value(
                                &mut self.scheme,
                                ReceiverScheme::Tregenza,
                                "Tregenza",
                            );
                        });
                        ui.end_row();

                        ui.label("Data Retrieval:");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(
                                &mut self.retrieval_mode,
                                DataRetrievalMode::BsdfOnly,
                                "BSDF Only",
                            );
                            ui.selectable_value(
                                &mut self.retrieval_mode,
                                DataRetrievalMode::FullData,
                                "Full Data",
                            );
                        });

                        add_contents(self, ui);
                    });
            });
    }
}

impl MeasurementKind {
    /// UI for measurement kind.
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
    /// Whether the dialog is open.
    is_open: bool,
    /// Output format of the measurement.
    format: OutputFormat,
    /// File encoding of the measurement.
    encoding: FileEncoding,
    /// Compression scheme of the file.
    compression: CompressionScheme,
    /// Whether to save the measurement.
    write_to_file: bool,
    /// Event loop proxy.
    event_loop: EventLoopProxy,
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    debug: MeasurementDialogDebug,
    #[cfg(feature = "visu-dbg")]
    cache: Cache,
}

impl MeasurementDialog {
    pub fn new(
        event_loop: EventLoopProxy,
        #[cfg(any(feature = "visu-dbg", debug_assertions))] cache: Cache,
    ) -> Self {
        MeasurementDialog {
            kind: MeasurementKind::Bsdf,
            selector: SurfaceSelector::multiple(),
            tab_bsdf: BsdfMeasurementTab::new(event_loop.clone()),
            tab_adf: AdfMeasurementTab::new(event_loop.clone()),
            tab_msf: MsfMeasurementTab::new(event_loop.clone()),
            is_open: false,
            format: OutputFormat::Vgmo,
            encoding: FileEncoding::Binary,
            compression: CompressionScheme::None,
            write_to_file: false,
            event_loop,
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            debug: MeasurementDialogDebug {
                enable_debug_draw: false,
                surf_prim_id: 0,
                show_surf_prim: false,
                surface_viewers: vec![],
                focused_viewer: None,
            },
            #[cfg(feature = "visu-dbg")]
            cache,
        }
    }

    pub fn update_surface_selector(&mut self, surfs: &[Handle<MicroSurface>], cache: &InnerCache) {
        self.selector.update(surfs, cache);
    }

    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    pub fn update_surface_viewers(&mut self, viewers: &[uuid::Uuid]) {
        for viewer in viewers {
            if !self.debug.surface_viewers.contains(viewer) {
                self.debug.surface_viewers.push(*viewer);
            }
        }
    }

    pub fn open(&mut self) { self.is_open = true; }

    pub fn show(&mut self, ctx: &egui::Context) {
        egui::Window::new("New Measurement")
            .open(&mut self.is_open)
            .fixed_size((400.0, 600.0))
            .show(ctx, |ui| {
                self.kind.selectable_ui(ui);
                #[cfg(feature = "visu-dbg")]
                {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Debug draw: ");
                        if ui
                            .add(ToggleSwitch::new(&mut self.debug.enable_debug_draw))
                            .changed()
                        {
                            self.event_loop.send_event(VgonioEvent::Debugging(
                                DebuggingEvent::ToggleDebugDrawing(self.debug.enable_debug_draw),
                            ));
                        }
                    });
                    if self.debug.enable_debug_draw {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Target viewer: ");
                            let prev = self.debug.focused_viewer;
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
                            if prev != self.debug.focused_viewer {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::FocusSurfaceViewer(self.debug.focused_viewer),
                                ));
                            }
                        });
                    }
                }
                egui::CollapsingHeader::new("Specimen")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::Grid::new("specimen_grid").show(ui, |ui| {
                            let old_surf = self.selector.first_selected();
                            ui.label("MicroSurfaces: ");
                            self.selector.ui("micro_surface_selector", ui);
                            ui.end_row();
                            let new_surf = self.selector.first_selected();
                            if old_surf != new_surf {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::UpdateFocusedSurface(new_surf),
                                ));
                            }

                            #[cfg(any(feature = "visu-dbg", debug_assertions))]
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
                                        self.event_loop.send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::UpdateSurfacePrimitiveId {
                                                id: self.debug.surf_prim_id as u32,
                                                status: self.debug.show_surf_prim,
                                            },
                                        ));
                                    }
                                });
                                ui.end_row();
                            }
                        });
                    });

                match self.kind {
                    MeasurementKind::Bsdf => {
                        #[cfg(feature = "visu-dbg")]
                        let orbit_radius = match self.selector.first_selected() {
                            None => 0.0,
                            Some(surf) => self.cache.read(|cache| {
                                crate::measure::estimate_orbit_radius(
                                    cache.get_micro_surface_mesh_by_surface_id(surf).unwrap(),
                                )
                            }),
                        };
                        self.tab_bsdf.ui(
                            ui,
                            #[cfg(feature = "visu-dbg")]
                            self.debug.enable_debug_draw,
                            #[cfg(feature = "visu-dbg")]
                            orbit_radius,
                        )
                    }
                    MeasurementKind::Adf => self.tab_adf.ui(ui),
                    MeasurementKind::Msf => self.tab_msf.ui(ui),
                }
                ui.separator();

                ui.horizontal_wrapped(|ui| {
                    ui.label("Write to file: ");
                    ui.add(ToggleSwitch::new(&mut self.write_to_file));
                });

                if self.write_to_file {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Output format: ");
                        ui.selectable_value(&mut self.format, OutputFormat::Vgmo, "vgmo");
                        if self.kind != MeasurementKind::Msf {
                            ui.selectable_value(&mut self.format, OutputFormat::Exr, "exr");
                        }
                    });

                    if self.format == OutputFormat::Vgmo {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("File encoding: ");
                            ui.selectable_value(&mut self.encoding, FileEncoding::Binary, "binary");
                            if self.kind == MeasurementKind::Adf {
                                ui.selectable_value(
                                    &mut self.encoding,
                                    FileEncoding::Ascii,
                                    "ascii",
                                );
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Compression scheme: ");
                            ui.selectable_value(
                                &mut self.compression,
                                CompressionScheme::None,
                                "none",
                            );
                            ui.selectable_value(
                                &mut self.compression,
                                CompressionScheme::Zlib,
                                "zlib",
                            );
                            ui.selectable_value(
                                &mut self.compression,
                                CompressionScheme::Gzip,
                                "gzip",
                            );
                        });
                    }
                }

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button("Simulate")
                        .on_hover_text("Starts the measurement")
                        .clicked()
                    {
                        if !self.selector.any_selected() {
                            self.event_loop.send_event(VgonioEvent::Notify {
                                kind: NotifyKind::Error,
                                text: "No surfaces selected!".to_string(),
                                time: 3.0,
                            });
                        } else {
                            let params = match self.kind {
                                MeasurementKind::Bsdf => {
                                    MeasurementParams::Bsdf(self.tab_bsdf.params)
                                }
                                MeasurementKind::Adf => MeasurementParams::Adf(self.tab_adf.params),
                                MeasurementKind::Msf => MeasurementParams::Msf(self.tab_msf.params),
                            };
                            self.event_loop.send_event(VgonioEvent::Measure {
                                params,
                                surfaces: self.selector.selected().collect::<Vec<_>>(),
                                format: self.format,
                                encoding: self.encoding,
                                compression: self.compression,
                                write_to_file: self.write_to_file,
                            });
                        }
                    }
                });
            });
    }
}

#[cfg(any(feature = "visu-dbg", debug_assertions))]
pub struct MeasurementDialogDebug {
    enable_debug_draw: bool,
    surf_prim_id: usize,
    show_surf_prim: bool,
    surface_viewers: Vec<uuid::Uuid>,
    focused_viewer: Option<uuid::Uuid>,
}

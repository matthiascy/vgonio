mod bsdf;
mod msf;
mod ndf;
mod sdf;

use crate::{
    app::{
        args::OutputFormat,
        cache::{Cache, RawCache},
        gui::{
            event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
            measurement::{
                bsdf::BsdfMeasurementTab, msf::GafMeasurementTab, ndf::NdfMeasurementTab,
                sdf::SdfMeasurementTab,
            },
            misc,
            notify::NotifyKind,
        },
    },
    io::{OutputFileFormatOption, OutputOptions},
    measure::{bsdf::receiver::ReceiverParams, params::MeasurementParams},
};
use egui::Widget;
use uxtk::widgets::{SurfaceSelector, ToggleSwitch};
use vgonio_core::{
    io::{CompressionScheme, FileEncoding},
    res::Handle,
    utils::partition::{PartitionScheme, SphericalDomain},
    MeasurementKind,
};

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
                        ui.horizontal_wrapped(|ui| {
                            misc::drag_angle(&mut self.precision.theta, "θ").ui(ui);
                            if self.scheme == PartitionScheme::EqualAngle {
                                misc::drag_angle(&mut self.precision.phi, "φ").ui(ui);
                            }
                        });
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
                                PartitionScheme::Beckers,
                                "Beckers",
                            );
                            ui.selectable_value(
                                &mut self.scheme,
                                PartitionScheme::EqualAngle,
                                "EqualAngle",
                            );
                        });
                        ui.end_row();

                        add_contents(self, ui);
                    });
            });
    }
}

pub fn measurement_kind_selectable_ui(kind: &mut MeasurementKind, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Measurement kind: ");
        ui.selectable_value(kind, MeasurementKind::Bsdf, "BSDF");
        ui.selectable_value(kind, MeasurementKind::Ndf, "NDF");
        ui.selectable_value(kind, MeasurementKind::Gaf, "MSF");
        ui.selectable_value(kind, MeasurementKind::Sdf, "SDF");
    });
}

pub struct MeasurementDialog {
    kind: MeasurementKind,
    selector: SurfaceSelector,
    tab_bsdf: BsdfMeasurementTab,
    tab_adf: NdfMeasurementTab,
    tab_msf: GafMeasurementTab,
    tab_sdf: SdfMeasurementTab,
    /// Whether the dialog is open.
    is_open: bool,
    /// Output format of the measurement.
    format: OutputFormat,
    /// Output embedding image resolution.
    img_res: u32,
    /// File encoding of the measurement.
    encoding: FileEncoding,
    /// Compression scheme of the file.
    compression: CompressionScheme,
    /// Whether to save the measurement.
    write_to_file: bool,
    /// Event loop proxy.
    event_loop: EventLoopProxy,
    #[cfg(any(feature = "vdbg", debug_assertions))]
    debug: MeasurementDialogDebug,
    #[cfg(any(feature = "vdbg", debug_assertions))]
    cache: Cache,
}

impl MeasurementDialog {
    pub fn new(
        event_loop: EventLoopProxy,
        #[cfg(any(feature = "vdbg", debug_assertions))] cache: Cache,
    ) -> Self {
        MeasurementDialog {
            kind: MeasurementKind::Bsdf,
            selector: SurfaceSelector::multiple(),
            tab_bsdf: BsdfMeasurementTab::new(event_loop.clone()),
            tab_adf: NdfMeasurementTab::new(event_loop.clone()),
            tab_msf: GafMeasurementTab::new(event_loop.clone()),
            tab_sdf: SdfMeasurementTab::new(),
            is_open: false,
            format: OutputFormat::Vgmo,
            img_res: 512,
            encoding: FileEncoding::Binary,
            compression: CompressionScheme::None,
            write_to_file: false,
            event_loop,
            #[cfg(any(feature = "vdbg", debug_assertions))]
            debug: MeasurementDialogDebug {
                enable_debug_draw: false,
                surf_prim_id: 0,
                show_surf_prim: false,
                surface_viewers: vec![],
                focused_viewer: None,
            },
            #[cfg(any(feature = "vdbg", debug_assertions))]
            cache,
        }
    }

    pub fn update_surface_selector(&mut self, surfs: &[Handle], cache: &RawCache) {
        let surfs = cache.get_micro_surface_records(surfs.iter());
        let surfs = surfs.iter().map(|r| (r.surf, r.name()));
        self.selector.update(surfs);
    }

    #[cfg(any(feature = "vdbg", debug_assertions))]
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
                measurement_kind_selectable_ui(&mut self.kind, ui);
                #[cfg(feature = "vdbg")]
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
                                    },
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

                            #[cfg(any(feature = "vdbg", debug_assertions))]
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
                        #[cfg(feature = "vdbg")]
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
                            #[cfg(feature = "vdbg")]
                            self.debug.enable_debug_draw,
                            #[cfg(feature = "vdbg")]
                            orbit_radius,
                        )
                    },
                    MeasurementKind::Ndf => self.tab_adf.ui(ui),
                    MeasurementKind::Gaf => self.tab_msf.ui(ui),
                    MeasurementKind::Sdf => {
                        self.tab_sdf.ui(ui);
                    },
                    _ => {},
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
                        if self.kind != MeasurementKind::Gaf {
                            ui.selectable_value(&mut self.format, OutputFormat::Exr, "exr");
                        }
                        ui.selectable_value(&mut self.format, OutputFormat::VgmoExr, "vgmo+exr");
                    });

                    if self.format.is_vgmo() {
                        egui::CollapsingHeader::new("Vgmo Options")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("File encoding: ");
                                    ui.selectable_value(
                                        &mut self.encoding,
                                        FileEncoding::Binary,
                                        "binary",
                                    );
                                    if self.kind == MeasurementKind::Ndf {
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
                            });
                    }

                    if self.format.is_exr() {
                        egui::CollapsingHeader::new("Exr Options")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Image resolution: ");
                                    ui.add(
                                        egui::DragValue::new(&mut self.img_res).range(256..=2048),
                                    );
                                });
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
                        }
                        let params = match self.kind {
                            MeasurementKind::Bsdf => {
                                MeasurementParams::Bsdf(self.tab_bsdf.params.clone())
                            },
                            MeasurementKind::Ndf => MeasurementParams::Ndf(self.tab_adf.params),
                            MeasurementKind::Gaf => MeasurementParams::Gaf(self.tab_msf.params),
                            MeasurementKind::Sdf => MeasurementParams::Sdf(self.tab_sdf.params),
                            _ => {
                                unreachable!("Invalid measurement kind")
                            },
                        };
                        let options = self.write_to_file.then(|| match self.format {
                            OutputFormat::Vgmo => OutputOptions {
                                dir: None,
                                formats: Box::new([OutputFileFormatOption::Vgmo {
                                    encoding: self.encoding,
                                    compression: self.compression,
                                }]),
                            },
                            OutputFormat::Exr => OutputOptions {
                                dir: None,
                                formats: Box::new([OutputFileFormatOption::Exr {
                                    resolution: self.img_res,
                                }]),
                            },
                            OutputFormat::VgmoExr => OutputOptions {
                                dir: None,
                                formats: Box::new([
                                    OutputFileFormatOption::Vgmo {
                                        encoding: self.encoding,
                                        compression: self.compression,
                                    },
                                    OutputFileFormatOption::Exr {
                                        resolution: self.img_res,
                                    },
                                ]),
                            },
                        });
                        self.event_loop.send_event(VgonioEvent::Measure {
                            params,
                            surfaces: self.selector.selected().collect::<Vec<_>>(),
                            output_opts: options,
                        });
                    }
                });
            });
    }
}

#[cfg(any(feature = "vdbg", debug_assertions))]
pub struct MeasurementDialogDebug {
    enable_debug_draw: bool,
    surf_prim_id: usize,
    show_surf_prim: bool,
    surface_viewers: Vec<uuid::Uuid>,
    focused_viewer: Option<uuid::Uuid>,
}

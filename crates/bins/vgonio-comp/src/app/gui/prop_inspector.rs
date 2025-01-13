use crate::app::gui::docking::{Dockable, WidgetKind};
#[cfg(feature = "fitting")]
use base::Symmetry;
use base::{partition::beckers, MeasurementKind};
#[cfg(feature = "fitting")]
use bxdf::{brdf::BxdfFamily, distro::MicrofacetDistroKind, fitting::FittingProblemKind};
use std::sync::{Arc, RwLock};
use surf::subdivision::SubdivisionKind;
use uuid::Uuid;

use crate::{
    app::{
        cache::Cache,
        gui::{
            data::PropertyData,
            event::{EventLoopProxy, VgonioEvent},
            outliner::OutlinerItem,
        },
    },
    measure::{
        mfd::{MeasuredGafData, MeasuredNdfData},
        params::NdfMeasurementMode,
        MeasurementSource,
    },
};

/// The property inspector.
///
/// The property inspector is a dockable widget that shows the properties of
/// selected objects. It is also used to edit the properties of objects.
pub struct PropertyInspector {
    /// The unique identifier of the property inspector.
    uuid: Uuid,
    /// Property inspector data.
    data: Arc<RwLock<PropertyData>>,
    /// Cache of the application.
    cache: Cache,
    /// Event channel.
    event_loop: EventLoopProxy,
}

impl PropertyInspector {
    pub fn new(event_loop: EventLoopProxy, cache: Cache, data: Arc<RwLock<PropertyData>>) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            data,
            cache,
            event_loop,
        }
    }

    fn grid_layout(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
        let available = ui.available_size();
        egui::Grid::new("surface_collapsable_header_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .min_col_width(available.x * 0.415)
            .show(ui, add_contents);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let properties = self.data.read().unwrap();
        ui.group(|ui| {
            ui.set_width(available.x - 12.0);
            ui.set_height(available.y - 12.0);
            match properties.selected {
                Some(item) => {
                    match item {
                        OutlinerItem::MicroSurface(surf) => {
                            let state = properties.surfaces.get(&surf).unwrap();
                            Self::grid_layout(ui, |ui| {
                                ui.label("Micro Surface");
                                ui.end_row();

                                ui.add(egui::Label::new("Resolution:"));
                                ui.add(egui::Label::new(format!(
                                    "{} x {}",
                                    state.size.0, state.size.1
                                )));
                                ui.end_row();

                                ui.add(egui::Label::new("Spacing:"));
                                ui.add(egui::Label::new(format!(
                                    "{} {} x {} {}",
                                    state.spacing.0, state.unit, state.spacing.1, state.unit,
                                )));
                                ui.end_row();

                                ui.add(egui::Label::new("Offset:"));
                                ui.add(egui::Label::new(format!(
                                    "{} {}",
                                    state.height_offset, state.unit
                                )));
                                ui.end_row();

                                ui.add(egui::Label::new("Lowest:"));
                                ui.add(egui::Label::new(format!(
                                    "{:.4} {}",
                                    state.min, state.unit
                                )));
                                ui.end_row();

                                ui.add(egui::Label::new("Highest:"));
                                ui.add(egui::Label::new(format!(
                                    "{:.4} {}",
                                    state.max, state.unit
                                )));
                                ui.end_row();

                                ui.add(egui::Label::new("Macro area:"));
                                ui.add(egui::Label::new(format!("{:.4}", state.macro_area)));
                                ui.end_row();

                                ui.add(egui::Label::new("Micro area:"));
                                ui.add(egui::Label::new(format!("{:.4}", state.micro_area)));
                                ui.end_row();

                                ui.add(egui::Label::new("RMS height:"));
                                ui.add(egui::Label::new(format!("{:.4}", state.rms_height)));
                                ui.end_row();

                                ui.add(egui::Label::new("RMS slope - X:"));
                                ui.add(egui::Label::new(format!("{:.4}", state.rms_slope_x)));
                                ui.end_row();

                                ui.add(egui::Label::new("RMS slope - Y:"));
                                ui.add(egui::Label::new(format!("{:.4}", state.rms_slope_y)));
                                ui.end_row();

                                // TODO: Add scale slider.
                                // ui.add(egui::Label::new("Scale:")).
                                // on_hover_text(
                                //     "Scales the surface visually. Doest
                                // not
                                // affect the actual surface.",
                                // );
                                // ui.add(
                                //     egui::Slider::new(&mut state.scale,
                                // 0.005..=1.5)
                                //         .trailing_fill(true),
                                // );
                                // ui.end_row();

                                let curved_subdivided = state
                                    .subdivided_micro_area
                                    .iter()
                                    .filter(|(s, _)| s.kind() == SubdivisionKind::Curved)
                                    .collect::<Box<_>>();

                                if !curved_subdivided.is_empty() {
                                    ui.add(egui::Label::new("Curved Subdivision:"));
                                    ui.end_row();

                                    for curved in curved_subdivided {
                                        ui.add(egui::Label::new(format!(
                                            "  L#{}:",
                                            curved.0.level()
                                        )));
                                        ui.add(egui::Label::new(format!("{}", curved.1)));
                                        ui.end_row();
                                    }
                                }

                                let wiggly_subdivided = state
                                    .subdivided_micro_area
                                    .iter()
                                    .filter(|(s, _)| s.kind() == SubdivisionKind::Wiggly)
                                    .collect::<Box<_>>();

                                if !wiggly_subdivided.is_empty() {
                                    ui.add(egui::Label::new("Wiggly Subdivision:"));
                                    ui.end_row();

                                    for wiggly in wiggly_subdivided {
                                        ui.add(egui::Label::new(format!(
                                            "  L#{}, offset {}:",
                                            wiggly.0.level(),
                                            wiggly.0.offset().unwrap_or(0)
                                        )));
                                        ui.add(egui::Label::new(format!("{}", wiggly.1)));
                                        ui.end_row();
                                    }
                                }
                            });
                        },
                        OutlinerItem::MeasurementData(meas) => {
                            let state = properties.measured.get(&meas).unwrap();
                            Self::grid_layout(ui, |ui| {
                                ui.label("Measurement Data");
                                ui.end_row();

                                ui.add(egui::Label::new("Timestamp:"));
                                ui.add(egui::Label::new(base::utils::iso_timestamp_display(
                                    &state.timestamp,
                                )));
                                ui.end_row();

                                ui.add(egui::Label::new("Kind:"));
                                ui.add(egui::Label::new(format!("{}", state.kind)));
                                ui.end_row();

                                ui.add(egui::Label::new("Source:"));
                                match &state.source {
                                    MeasurementSource::Loaded(path) => {
                                        ui.add(egui::Label::new("Loaded")).on_hover_text(format!(
                                            "Surface path: {}",
                                            path.display()
                                        ));
                                    },
                                    MeasurementSource::Measured(hdl) => {
                                        ui.add(egui::Label::new("Measured"))
                                            .on_hover_text(format!("Surface ID: {}", hdl));
                                    },
                                }
                                ui.end_row();

                                self.cache.read(|cache| {
                                    let measured = &cache.get_measurement(meas).unwrap().measured;
                                    match measured.kind() {
                                        MeasurementKind::Ndf => {
                                            let ndf =
                                                measured.downcast_ref::<MeasuredNdfData>().unwrap();
                                            match ndf.params.mode {
                                                NdfMeasurementMode::ByPoints {
                                                    azimuth,
                                                    zenith,
                                                } => {
                                                    ui.label("θ:");
                                                    ui.label(format!(
                                                        "{} ~ {}, every {}",
                                                        zenith.start.prettified(),
                                                        zenith.stop.prettified(),
                                                        zenith.step_size.prettified()
                                                    ));
                                                    ui.end_row();

                                                    #[cfg(debug_assertions)]
                                                    {
                                                        ui.label("θ bins count:");
                                                        ui.label(format!(
                                                            "{}",
                                                            zenith.step_count_wrapped()
                                                        ));
                                                        ui.end_row()
                                                    }

                                                    ui.label("φ:");
                                                    ui.label(format!(
                                                        "{} ~ {}, every {}",
                                                        azimuth.start.prettified(),
                                                        azimuth.stop.prettified(),
                                                        azimuth.step_size.prettified(),
                                                    ));
                                                    ui.end_row();

                                                    #[cfg(debug_assertions)]
                                                    {
                                                        ui.label("φ bins count:");
                                                        ui.label(format!(
                                                            "{}",
                                                            azimuth.step_count_wrapped()
                                                        ));
                                                        ui.end_row()
                                                    }
                                                },
                                                NdfMeasurementMode::ByPartition { precision } => {
                                                    ui.label("Precision:");
                                                    ui.label(format!("{:.4}", precision));
                                                    ui.end_row();

                                                    ui.label("Patch count:");
                                                    ui.label(format!(
                                                        "{}",
                                                        beckers::compute_hemisphere_patch_count(
                                                            precision
                                                        )
                                                    ));
                                                },
                                            }
                                        },
                                        MeasurementKind::Gaf => {
                                            let msf =
                                                measured.downcast_ref::<MeasuredGafData>().unwrap();
                                            ui.label("θ:");
                                            ui.label(format!(
                                                "{} ~ {}, every {}",
                                                msf.params.zenith.start.prettified(),
                                                msf.params.zenith.stop.prettified(),
                                                msf.params.zenith.step_size.prettified()
                                            ));
                                            ui.end_row();

                                            #[cfg(debug_assertions)]
                                            {
                                                ui.label("θ bins count:");
                                                ui.label(format!(
                                                    "{}",
                                                    msf.params.zenith.step_count_wrapped()
                                                ));
                                                ui.end_row()
                                            }

                                            ui.label("φ:");
                                            ui.label(format!(
                                                "{} ~ {}, every {}",
                                                msf.params.azimuth.start.prettified(),
                                                msf.params.azimuth.stop.prettified(),
                                                msf.params.azimuth.step_size.prettified(),
                                            ));
                                            ui.end_row();

                                            #[cfg(debug_assertions)]
                                            {
                                                ui.label("φ bins count:");
                                                ui.label(format!(
                                                    "{}",
                                                    msf.params.azimuth.step_count_wrapped()
                                                ));
                                                ui.end_row()
                                            }
                                        },
                                        _ => {},
                                    }
                                })
                            });
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("Graph(new window)").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Graphing {
                                        kind: state.kind,
                                        data: meas,
                                        independent: true,
                                    });
                                }
                                if ui.button("Graph(new tab)").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Graphing {
                                        kind: state.kind,
                                        data: meas,
                                        independent: false,
                                    });
                                }
                                if ui.button("Export").clicked() {
                                    self.event_loop
                                        .send_event(VgonioEvent::ExportMeasurement(meas));
                                }
                                #[cfg(feature = "fitting")]
                                // Temporary fitting button. TODO: Remove.
                                if ui.button("Fit Beckmann(iso)").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Fitting {
                                        kind: FittingProblemKind::Bxdf {
                                            family: BxdfFamily::Microfacet,
                                            distro: Some(MicrofacetDistroKind::Beckmann),
                                            symmetry: Symmetry::Isotropic,
                                        },
                                        data: meas,
                                        scale: 1.0,
                                    });
                                }
                                #[cfg(feature = "fitting")]
                                // Temporary fitting button. TODO: Remove.
                                if ui.button("Fit Beckmann(aniso)").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Fitting {
                                        kind: FittingProblemKind::Bxdf {
                                            family: BxdfFamily::Microfacet,
                                            distro: Some(MicrofacetDistroKind::Beckmann),
                                            symmetry: Symmetry::Anisotropic,
                                        },
                                        data: meas,
                                        scale: 1.0,
                                    });
                                }
                                #[cfg(feature = "fitting")]
                                // Temporary fitting button. TODO: Remove.
                                if ui.button("Fit GGX(iso)").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Fitting {
                                        kind: FittingProblemKind::Bxdf {
                                            family: BxdfFamily::Microfacet,
                                            distro: Some(MicrofacetDistroKind::TrowbridgeReitz),
                                            symmetry: Symmetry::Isotropic,
                                        },
                                        data: meas,
                                        scale: 1.0,
                                    });
                                }
                                #[cfg(feature = "fitting")]
                                // Temporary fitting button. TODO: Remove.
                                if ui.button("Fit GGX(aniso)").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Fitting {
                                        kind: FittingProblemKind::Bxdf {
                                            family: BxdfFamily::Microfacet,
                                            distro: Some(MicrofacetDistroKind::TrowbridgeReitz),
                                            symmetry: Symmetry::Isotropic,
                                        },
                                        data: meas,
                                        scale: 1.0,
                                    });
                                }
                            });
                        },
                    }
                },
                None => {
                    ui.label("No object selected");
                },
            }
        });
    }
}

impl Dockable for PropertyInspector {
    fn kind(&self) -> WidgetKind { WidgetKind::Properties }

    fn title(&self) -> egui::WidgetText { "Properties".into() }

    fn uuid(&self) -> Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) { self.ui(ui); }
}

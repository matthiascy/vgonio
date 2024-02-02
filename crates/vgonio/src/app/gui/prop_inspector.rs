use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::app::gui::docking::{Dockable, WidgetKind};

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
        data::{MeasuredData, MeasurementDataSource},
        params::AdfMeasurementMode,
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
                            });
                        }
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
                                    MeasurementDataSource::Loaded(path) => {
                                        ui.add(egui::Label::new("Loaded")).on_hover_text(format!(
                                            "Surface path: {}",
                                            path.display()
                                        ));
                                    }
                                    MeasurementDataSource::Measured(hdl) => {
                                        ui.add(egui::Label::new("Measured"))
                                            .on_hover_text(format!("Surface ID: {}", hdl));
                                    }
                                }
                                ui.end_row();

                                self.cache.read(|cache| {
                                    match &cache.get_measurement_data(meas).unwrap().measured {
                                        MeasuredData::Adf(adf) => match adf.params.mode {
                                            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
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
                                            }
                                            AdfMeasurementMode::ByPartition { .. } => {
                                                // TODO: Add partition info.
                                                log::info!("Partition mode not implemented");
                                            }
                                        },
                                        MeasuredData::Msf(mmsf) => {
                                            ui.label("θ:");
                                            ui.label(format!(
                                                "{} ~ {}, every {}",
                                                mmsf.params.zenith.start.prettified(),
                                                mmsf.params.zenith.stop.prettified(),
                                                mmsf.params.zenith.step_size.prettified()
                                            ));
                                            ui.end_row();

                                            #[cfg(debug_assertions)]
                                            {
                                                ui.label("θ bins count:");
                                                ui.label(format!(
                                                    "{}",
                                                    mmsf.params.zenith.step_count_wrapped()
                                                ));
                                                ui.end_row()
                                            }

                                            ui.label("φ:");
                                            ui.label(format!(
                                                "{} ~ {}, every {}",
                                                mmsf.params.azimuth.start.prettified(),
                                                mmsf.params.azimuth.stop.prettified(),
                                                mmsf.params.azimuth.step_size.prettified(),
                                            ));
                                            ui.end_row();

                                            #[cfg(debug_assertions)]
                                            {
                                                ui.label("φ bins count:");
                                                ui.label(format!(
                                                    "{}",
                                                    mmsf.params.azimuth.step_count_wrapped()
                                                ));
                                                ui.end_row()
                                            }
                                        }
                                        MeasuredData::Bsdf(_) => {}
                                        MeasuredData::Sdf(_) => {}
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
                            });
                        }
                    }
                }
                None => {
                    ui.label("No object selected");
                }
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
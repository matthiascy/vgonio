use std::sync::{Arc, RwLock};
use uuid::Uuid;

use crate::app::gui::docking::{Dockable, WidgetKind};

use crate::{
    app::{
        cache::Cache,
        gui::{data::PropertyData, outliner::OutlinerItem},
    },
    measure::measurement::{MeasuredData, MeasurementDataSource},
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
    cache: Arc<RwLock<Cache>>,
}

impl PropertyInspector {
    pub fn new(cache: Arc<RwLock<Cache>>, data: Arc<RwLock<PropertyData>>) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            data,
            cache,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let data = self.data.read().unwrap();
        ui.group(|ui| {
            ui.set_width(available.x - 12.0);
            ui.set_height(available.y - 12.0);
            egui::Grid::new("surface_collapsable_header_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .min_col_width(available.x * 0.415)
                .show(ui, |ui| {
                    match data.selected {
                        Some(item) => {
                            match item {
                                OutlinerItem::MicroSurface(surf) => {
                                    let state = data.surfaces.get(&surf).unwrap();
                                    ui.label("Micro Surface");
                                    ui.end_row();

                                    ui.add(egui::Label::new("Resolution:"));
                                    ui.add(egui::Label::new(format!(
                                        "{} x {}",
                                        state.size.0, state.size.1
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
                                }
                                OutlinerItem::MeasurementData(meas) => {
                                    let state = data.measured.get(&meas).unwrap();
                                    ui.label("Measurement Data");
                                    ui.end_row();

                                    ui.add(egui::Label::new("Kind:"));
                                    ui.add(egui::Label::new(format!("{}", state.kind)));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Source:"));
                                    match &state.source {
                                        MeasurementDataSource::Loaded(path) => {
                                            ui.add(egui::Label::new("Loaded")).on_hover_text(
                                                format!("Surface path: {}", path.display()),
                                            );
                                        }
                                        MeasurementDataSource::Measured(hdl) => {
                                            ui.add(egui::Label::new("Measured"))
                                                .on_hover_text(format!("Surface ID: {}", hdl));
                                        }
                                    }
                                    ui.end_row();

                                    let cache = self.cache.read().unwrap();
                                    let data = cache.get_measurement_data(meas).unwrap();
                                    match &data.measured {
                                        MeasuredData::Madf(madf) => {
                                            ui.label("θ:");
                                            ui.label(format!(
                                                "{} ~ {}, every {}",
                                                madf.params.zenith.start.prettified(),
                                                madf.params.zenith.stop.prettified(),
                                                madf.params.zenith.step_size.prettified()
                                            ));
                                            ui.end_row();

                                            #[cfg(debug_assertions)]
                                            {
                                                ui.label("θ bins count:");
                                                ui.label(format!(
                                                    "{}",
                                                    madf.params.zenith.step_count_wrapped()
                                                ));
                                                ui.end_row()
                                            }

                                            ui.label("φ:");
                                            ui.label(format!(
                                                "{} ~ {}, every {}",
                                                madf.params.azimuth.start.prettified(),
                                                madf.params.azimuth.stop.prettified(),
                                                madf.params.azimuth.step_size.prettified(),
                                            ));
                                            ui.end_row();

                                            #[cfg(debug_assertions)]
                                            {
                                                ui.label("φ bins count:");
                                                ui.label(format!(
                                                    "{}",
                                                    madf.params.azimuth.step_count_wrapped()
                                                ));
                                                ui.end_row()
                                            }
                                        }
                                        MeasuredData::Mmsf(mmsf) => {
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
                                    }
                                }
                            }
                        }
                        None => {
                            ui.label("No object selected");
                            ui.end_row();
                        }
                    }
                });
        });
    }
}

impl Dockable for PropertyInspector {
    fn kind(&self) -> WidgetKind { WidgetKind::Properties }

    fn title(&self) -> egui::WidgetText { "Properties".into() }

    fn uuid(&self) -> Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) { self.ui(ui); }
}

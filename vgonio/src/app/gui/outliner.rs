use crate::{
    app::{
        cache::Handle,
        gui::{
            data::MeasurementDataProp,
            event::{EventLoopProxy, OutlinerEvent, VgonioEvent},
        },
    },
    measure::measurement::MeasurementData,
};
use egui::WidgetText;
use std::sync::{Arc, RwLock};
use vgsurf::MicroSurface;

use crate::app::gui::{
    data::{MicroSurfaceProp, PropertyData},
    docking::{Dockable, WidgetKind},
};

/// Outliner is a widget that displays the scene graph of the current scene.
///
/// It will reads the micro-surfaces from the cache and display them in a tree
/// structure. The user can toggle the visibility of the micro surfaces.
pub struct Outliner {
    /// The unique id of the outliner.
    uuid: uuid::Uuid,
    // gpu_ctx: Arc<GpuContext>,
    event_loop: EventLoopProxy,
    /// Collapsable headers for the micro surfaces.
    surface_headers: Vec<CollapsableHeader<Handle<MicroSurface>>>,
    /// Collapsable headers for the measured data.
    measured_headers: Vec<CollapsableHeader<Handle<MeasurementData>>>,
    /// The currently selected item.
    selected: Option<OutlinerItem>,
    /// The property data of each item.
    data: Arc<RwLock<PropertyData>>,
}

impl Outliner {
    /// Creates a new outliner.
    pub fn new(data: Arc<RwLock<PropertyData>>, event_loop: EventLoopProxy) -> Self {
        log::info!("Creating outliner");
        Self {
            uuid: uuid::Uuid::new_v4(),
            event_loop,
            surface_headers: vec![],
            measured_headers: vec![],
            selected: None,
            data,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OutlinerItem {
    MicroSurface(Handle<MicroSurface>),
    MeasurementData(Handle<MeasurementData>),
}

pub struct CollapsableHeader<T> {
    item: T,
}

impl CollapsableHeader<Handle<MicroSurface>> {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut MicroSurfaceProp,
        current: &mut Option<OutlinerItem>,
        event_loop: &EventLoopProxy,
    ) {
        let id = ui.make_persistent_id(&state.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    ui.horizontal(|ui| {
                        let mut selected = match current {
                            None => false,
                            Some(item) => match item {
                                OutlinerItem::MicroSurface(surf) => {
                                    if surf == &self.item {
                                        true
                                    } else {
                                        false
                                    }
                                }
                                OutlinerItem::MeasurementData(_) => false,
                            },
                        };
                        if ui.selectable_label(selected, &state.name).clicked() {
                            if selected == false {
                                *current = Some(OutlinerItem::MicroSurface(self.item));
                                event_loop
                                    .send_event(VgonioEvent::Outliner(OutlinerEvent::SelectItem(
                                        OutlinerItem::MicroSurface(self.item),
                                    )))
                                    .unwrap();
                            }
                        }
                    })
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut state.visible, "");
                })
            })
            .body(|ui| {
                // Scale
                egui::Grid::new("surface_collapsable_header_grid")
                    .num_columns(3)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new("Size:"));
                        ui.add(egui::Label::new(format!(
                            "{} x {}",
                            state.size.0, state.size.1
                        )));
                        ui.end_row();

                        ui.add(egui::Label::new("Min:"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.min, state.unit)));
                        ui.end_row();

                        ui.add(egui::Label::new("Max:"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.max, state.unit)));
                        ui.end_row();

                        ui.add(egui::Label::new("Scale:")).on_hover_text(
                            "Scales the surface visually. Doest not affect the actual surface.",
                        );
                        ui.add(
                            egui::Slider::new(&mut state.scale, 0.005..=1.5).trailing_fill(true),
                        );
                        ui.end_row();
                    });
            });
    }
}

impl CollapsableHeader<Handle<MeasurementData>> {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        prop: &mut MeasurementDataProp,
        current: &mut Option<OutlinerItem>,
        event_loop: &EventLoopProxy,
    ) {
        let id = ui.make_persistent_id(&prop.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    ui.horizontal(|ui| {
                        let mut selected = match current {
                            None => false,
                            Some(item) => match item {
                                OutlinerItem::MicroSurface(_) => false,
                                OutlinerItem::MeasurementData(data) => {
                                    if data == &self.item {
                                        true
                                    } else {
                                        false
                                    }
                                }
                            },
                        };
                        if ui.selectable_label(selected, &prop.name).clicked() {
                            if selected == false {
                                *current = Some(OutlinerItem::MeasurementData(self.item));
                                event_loop
                                    .send_event(VgonioEvent::Outliner(OutlinerEvent::SelectItem(
                                        OutlinerItem::MeasurementData(self.item),
                                    )))
                                    .unwrap();
                            }
                        }
                    })
                });
            })
            .body(|ui| {
                let measurement_kind = prop.kind;
                egui::Grid::new("measurement_data_body")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Type:");
                        ui.label(format!("{}", measurement_kind));
                        ui.end_row();
                        // ui.label("Source:");
                        // match prop.source {
                        //     MeasurementDataSource::Loaded(_) => {
                        //         ui.label("loaded");
                        //     }
                        //     MeasurementDataSource::Measured(_) => {
                        //         ui.label("measured");
                        //     }
                        // }
                        // ui.end_row();

                        // if measurement_kind ==
                        // MeasurementKind::MicrofacetAreaDistribution     ||
                        // measurement_kind ==
                        // MeasurementKind::MicrofacetMaskingShadowing
                        // {
                        //     let zenith =
                        // .measured.madf_or_mmsf_zenith().unwrap();
                        //     let azimuth =
                        // measured.measured.madf_or_mmsf_azimuth().unwrap();
                        //     ui.label("θ:");
                        //     ui.label(format!(
                        //         "{:.2}° ~ {:.2}°, every {:.2}°",
                        //         zenith.start.to_degrees(),
                        //         zenith.stop.to_degrees(),
                        //         zenith.step_size.to_degrees(),
                        //     ));
                        //     ui.end_row();
                        //     #[cfg(debug_assertions)]
                        //     {
                        //         ui.label("θ bins:");
                        //         ui.label(format!("{}",
                        // zenith.step_count_wrapped()));
                        //         ui.end_row()
                        //     }
                        //     ui.label("φ:");
                        //     ui.label(format!(
                        //         "{:.2}° ~ {:.2}°, every {:.2}°",
                        //         azimuth.start.to_degrees(),
                        //         azimuth.stop.to_degrees(),
                        //         azimuth.step_size.to_degrees(),
                        //     ));
                        //     ui.end_row();
                        //     #[cfg(debug_assertions)]
                        //     {
                        //         ui.label("φ bins:");
                        //         ui.label(format!("{}",
                        // azimuth.step_count_wrapped()));
                        //         ui.end_row()
                        //     }
                        // }
                    });
                ui.add_space(5.0);
                if ui.button("Plot").clicked() {
                    // TODO: Send message to bsdf viewer to plot this data.
                    // self.show_plot = true;
                    // if !plots.iter_mut().any(|p| p.0.ptr_eq(&data)) {
                    //     match &measured.measured {
                    //         MeasuredData::Madf(_) => {
                    //             plots.push((
                    //                 data.clone(),
                    //                 Box::new(PlottingInspector::new(
                    //                     measured.name.clone(),
                    //                     measured.clone(),
                    //                     MadfPlottingControls::default(),
                    //                     gpu,
                    //                     event_loop,
                    //                 )),
                    //             ));
                    //         }
                    //         MeasuredData::Mmsf(_) => {
                    //             plots.push((
                    //                 data.clone(),
                    //                 Box::new(PlottingInspector::new(
                    //                     measured.name.clone(),
                    //                     measured.clone(),
                    //                     MmsfPlottingControls::default(),
                    //                     gpu,
                    //                     event_loop,
                    //                 )),
                    //             ));
                    //         }
                    //         MeasuredData::Bsdf(_) => {
                    //             plots.push((
                    //                 data.clone(),
                    //                 Box::new(PlottingInspector::new(
                    //                     measured.name.clone(),
                    //                     measured.clone(),
                    //                     BsdfPlottingControls::new(
                    //
                    // bsdf_viewer.write().unwrap().create_new_view(),
                    //                     ),
                    //                     gpu,
                    //                     event_loop,
                    //                 )),
                    //             ));
                    //         }
                    //     }
                    // }
                }
            });
    }
}

// GUI related functions
impl Outliner {
    /// Creates the ui for the outliner.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // Update the list of surfaces and measurements.
        {
            let data = self.data.read().unwrap();
            for (hdl, _) in &data.surfaces {
                if !self.surface_headers.iter().any(|h| h.item == *hdl) {
                    self.surface_headers
                        .push(CollapsableHeader { item: hdl.clone() });
                }
            }

            for (hdl, _) in &data.measured {
                if !self.measured_headers.iter().any(|h| h.item == *hdl) {
                    self.measured_headers
                        .push(CollapsableHeader { item: hdl.clone() });
                }
            }
        }

        let mut prop_data = self.data.write().unwrap();
        egui::CollapsingHeader::new("MicroSurfaces")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for hdr in self.surface_headers.iter_mut() {
                        hdr.ui(
                            ui,
                            &mut prop_data.surfaces.get_mut(&hdr.item).unwrap(),
                            &mut self.selected,
                            &self.event_loop,
                        );
                    }
                });
            });
        egui::CollapsingHeader::new("Measurements")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for hdr in self.measured_headers.iter_mut() {
                        hdr.ui(
                            ui,
                            &mut prop_data.measured.get_mut(&hdr.item).unwrap(),
                            &mut self.selected,
                            &self.event_loop,
                        );
                    }
                })
            });

        // TODO: draw plots in main loop.
        // for (data, plot) in self.plotting_inspectors.iter_mut() {
        //     let open = &mut self
        //         .measurements
        //         .iter_mut()
        //         .find(|(d, _)| d.ptr_eq(data))
        //         .unwrap()
        //         .1
        //         .show_plot;
        //     plot.show(ui.ctx(), open);
        // }
    }
}

impl Dockable for Outliner {
    fn kind(&self) -> WidgetKind { WidgetKind::Outliner }

    fn title(&self) -> WidgetText { "Outliner".into() }

    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) { self.ui(ui) }
}

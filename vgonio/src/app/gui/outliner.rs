use crate::{
    app::{
        cache::Handle,
        gui::{
            data::PropertyData,
            docking::{Dockable, WidgetKind},
            event::{EventLoopProxy, OutlinerEvent, VgonioEvent},
        },
    },
    measure::measurement::MeasurementData,
};
use egui::WidgetText;
use std::sync::{Arc, RwLock};
use vgsurf::MicroSurface;

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
            data,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Item {
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
        prop: &Arc<RwLock<PropertyData>>,
        event_loop: &EventLoopProxy,
    ) {
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(self.item.id()),
            false,
        )
        .show_header(ui, |ui| {
            ui.vertical_centered_justified(|ui| {
                ui.horizontal(|ui| {
                    let mut data = prop.write().unwrap();
                    let selected = match &data.selected {
                        None => false,
                        Some(item) => match item {
                            Item::MicroSurface(surf) => surf == &self.item,
                            Item::MeasurementData(_) => false,
                        },
                    };
                    let state = data.surfaces.get(&self.item).unwrap();
                    if ui.selectable_label(selected, &state.name).clicked() && !selected {
                        data.selected = Some(Item::MicroSurface(self.item));
                        event_loop
                            .send_event(VgonioEvent::Outliner(OutlinerEvent::SelectItem(
                                Item::MicroSurface(self.item),
                            )))
                            .unwrap();
                    }
                })
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut prop = prop.write().unwrap();
                let state = prop.surfaces.get_mut(&self.item).unwrap();
                ui.checkbox(&mut state.visible, "");
            })
        })
        .body(|ui| {
            let mut prop = prop.write().unwrap();
            let scale = &mut prop.surfaces.get_mut(&self.item).unwrap().scale;
            // Scale
            egui::Grid::new("surface_collapsable_header_grid")
                .num_columns(3)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.add(egui::Label::new("Scale:")).on_hover_text(
                        "Scales the surface visually. Doest not affect the actual surface.",
                    );
                    ui.add(egui::Slider::new(scale, 0.005..=1.5).trailing_fill(true));
                    ui.end_row();
                });
        });
    }
}

impl CollapsableHeader<Handle<MeasurementData>> {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        prop: &Arc<RwLock<PropertyData>>,
        event_loop: &EventLoopProxy,
    ) {
        let _ = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(self.item.id()),
            false,
        )
        .show_header(ui, |ui| {
            ui.vertical_centered_justified(|ui| {
                ui.horizontal(|ui| {
                    let mut data = prop.write().unwrap();
                    let selected = match &data.selected {
                        None => false,
                        Some(item) => match item {
                            Item::MicroSurface(_) => false,
                            Item::MeasurementData(data) => {
                                if data == &self.item {
                                    true
                                } else {
                                    false
                                }
                            }
                        },
                    };
                    let state = data.measured.get(&self.item).unwrap();
                    if ui.selectable_label(selected, &state.name).clicked() && !selected {
                        data.selected = Some(Item::MeasurementData(self.item));
                        event_loop
                            .send_event(VgonioEvent::Outliner(OutlinerEvent::SelectItem(
                                Item::MeasurementData(self.item),
                            )))
                            .unwrap();
                    }
                })
            });
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

        egui::CollapsingHeader::new("MicroSurfaces")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for hdr in self.surface_headers.iter_mut() {
                        hdr.ui(ui, &self.data, &self.event_loop);
                    }
                });
            });

        egui::CollapsingHeader::new("Measurements")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for hdr in self.measured_headers.iter_mut() {
                        hdr.ui(ui, &self.data, &self.event_loop);
                    }
                })
            });
    }
}

impl Dockable for Outliner {
    fn kind(&self) -> WidgetKind { WidgetKind::Outliner }

    fn title(&self) -> WidgetText { "Outliner".into() }

    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) { self.ui(ui) }
}

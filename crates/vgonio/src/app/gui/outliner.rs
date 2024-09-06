use crate::{
    app::gui::{
        data::PropertyData,
        docking::{Dockable, WidgetKind},
        event::{EventLoopProxy, OutlinerEvent, VgonioEvent},
    },
    measure::Measurement,
};
use base::handle::Handle;
use egui::WidgetText;
use std::sync::{Arc, RwLock};
use surf::{
    subdivision::{Subdivision, SubdivisionKind},
    MicroSurface,
};

/// Outliner is a widget that displays the scene graph of the current scene.
///
/// It will read the micro-surfaces from the cache and display them in a tree
/// structure. The user can toggle the visibility of the micro surfaces.
pub struct Outliner {
    /// The unique id of the outliner.
    uuid: uuid::Uuid,
    // gpu_ctx: Arc<GpuContext>,
    event_loop: EventLoopProxy,
    /// Collapsable headers for the micro surfaces.
    surface_headers: Vec<CollapsableHeader<Handle<MicroSurface>>>,
    /// Collapsable headers for the measured data.
    measured_headers: Vec<CollapsableHeader<Handle<Measurement>>>,
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
pub enum OutlinerItem {
    MicroSurface(Handle<MicroSurface>),
    MeasurementData(Handle<Measurement>),
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
                            OutlinerItem::MicroSurface(surf) => surf == &self.item,
                            OutlinerItem::MeasurementData(_) => false,
                        },
                    };
                    let state = data.surfaces.get(&self.item).unwrap();
                    if ui.selectable_label(selected, &state.name).clicked() && !selected {
                        data.selected = Some(OutlinerItem::MicroSurface(self.item));
                        event_loop.send_event(VgonioEvent::Outliner(OutlinerEvent::SelectItem(
                            OutlinerItem::MicroSurface(self.item),
                        )));
                    }
                })
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut prop = prop.write().unwrap();
                let state = prop.surfaces.get_mut(&self.item).unwrap();
                ui.checkbox(&mut state.visible, "");
                if ui.button("X").clicked() {
                    event_loop.send_event(VgonioEvent::Outliner(OutlinerEvent::RemoveItem(
                        OutlinerItem::MicroSurface(self.item),
                    )));
                }
            })
        })
        .body(|ui| {
            let mut prop = prop.write().unwrap();
            let surf_props = prop.surfaces.get_mut(&self.item).unwrap();
            let scale = &mut surf_props.scale;
            let subdivision_level = &mut surf_props.subdivision_level;
            let subdivision_kind = &mut surf_props.subdivision_kind;
            let subdivision_offset = &mut surf_props.subdivision_offset;
            egui::Grid::new("surface_collapsable_header_grid")
                .num_columns(3)
                .spacing([20.0, 4.0])
                .show(ui, |ui| {
                    ui.add(egui::Label::new("Scale:")).on_hover_text(
                        "Scales the surface visually. Doest not affect the actual surface.",
                    );
                    ui.add(egui::Slider::new(scale, 0.005..=1.5).trailing_fill(true));
                    ui.end_row();

                    ui.add(egui::Label::new("Subdivision:"))
                        .on_hover_text("Subdivides the surface.");
                    ui.end_row();

                    ui.horizontal(|ui| {
                        if ui.button("<").clicked() {
                            *subdivision_level = subdivision_level.saturating_sub(1);
                        }
                        if ui.button(">").clicked() {
                            *subdivision_level = subdivision_level.saturating_add(1);
                        }
                        ui.add(egui::DragValue::new(subdivision_level).speed(1))
                            .on_hover_text("Subdivision level for the surface.");
                    });
                    ui.horizontal(|ui| {
                        ui.selectable_value(subdivision_kind, SubdivisionKind::Curved, "Curved");
                        ui.selectable_value(subdivision_kind, SubdivisionKind::Wiggly, "Wiggly");
                    });
                    ui.end_row();

                    if subdivision_kind == &SubdivisionKind::Wiggly {
                        ui.add(egui::Label::new("Offset:")).on_hover_text(
                            "The offset to add randomly to the z coordinate of the new points in \
                             percentage.",
                        );
                        ui.add(egui::Slider::new(subdivision_offset, 0..=500).trailing_fill(true));
                        ui.end_row();
                    }

                    if ui.button("Subdivide").clicked() {
                        let subdivision = if *subdivision_kind == SubdivisionKind::Curved {
                            Subdivision::Curved(*subdivision_level)
                        } else {
                            Subdivision::Wiggly {
                                level: *subdivision_level,
                                offset: *subdivision_offset,
                            }
                        };
                        event_loop.send_event(VgonioEvent::SubdivideSurface {
                            surf: self.item,
                            subdivision,
                        });
                    }
                    ui.end_row();
                });
        });
    }
}

impl CollapsableHeader<Handle<Measurement>> {
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
                            OutlinerItem::MicroSurface(_) => false,
                            OutlinerItem::MeasurementData(data) => data == &self.item,
                        },
                    };
                    let state = data.measured.get(&self.item).unwrap();
                    if ui.selectable_label(selected, &state.name).clicked() && !selected {
                        data.selected = Some(OutlinerItem::MeasurementData(self.item));
                        event_loop.send_event(VgonioEvent::Outliner(OutlinerEvent::SelectItem(
                            OutlinerItem::MeasurementData(self.item),
                        )));
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
            for hdl in data.surfaces.keys() {
                if !self.surface_headers.iter().any(|h| h.item == *hdl) {
                    self.surface_headers.push(CollapsableHeader { item: *hdl });
                }
            }

            for hdl in data.measured.keys() {
                if !self.measured_headers.iter().any(|h| h.item == *hdl) {
                    self.measured_headers.push(CollapsableHeader { item: *hdl });
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

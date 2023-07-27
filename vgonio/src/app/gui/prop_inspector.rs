use std::sync::{Arc, RwLock};

use crate::app::gui::docking::{Dockable, WidgetKind};

use crate::app::gui::data::PropertyData;

/// The property inspector.
///
/// The property inspector is a dockable widget that shows the properties of
/// selected objects. It is also used to edit the properties of objects.
pub struct PropertyInspector {
    /// The unique identifier of the property inspector.
    uuid: uuid::Uuid,
    /// Property inspector data.
    data: Arc<RwLock<PropertyData>>,
}

impl PropertyInspector {
    pub fn new(data: Arc<RwLock<PropertyData>>) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4(),
            data,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let data = self.data.read().unwrap();
        match data.selected {
            Some(item) => match item {
                super::outliner::OutlinerItem::MicroSurface(surf) => {
                    let state = data.surfaces.get(&surf).unwrap();
                    egui::Grid::new("surface_collapsable_header_grid")
                        .num_columns(3)
                        .spacing([40.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("Micro Surface");
                            ui.end_row();

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

                            // ui.add(egui::Label::new("Scale:")).on_hover_text(
                            //     "Scales the surface visually. Doest not
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
                super::outliner::OutlinerItem::MeasurementData(meas) => {}
            },
            None => {
                ui.label("Nothing selected");
            }
        }
    }
}

impl Dockable for PropertyInspector {
    fn kind(&self) -> WidgetKind { WidgetKind::Properties }

    fn title(&self) -> egui::WidgetText { "Properties".into() }

    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) { self.ui(ui); }
}

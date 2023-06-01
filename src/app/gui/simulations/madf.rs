use crate::{app::cache::Handle, measure::measurement::MadfMeasurementParams, msurf::MicroSurface};
use egui::{epaint::ahash::HashSet, WidgetType::CollapsingHeader};

pub struct MadfSimulation {
    pub params: MadfMeasurementParams,
    pub loaded: HashSet<Handle<MicroSurface>>,
    pub selected: HashSet<Handle<MicroSurface>>,
}

impl MadfSimulation {
    pub fn new() -> Self {
        Self {
            params: MadfMeasurementParams::default(),
            loaded: HashSet::default(),
            selected: HashSet::default(),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("madf_sim_grid")
            .striped(true)
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Zenith angle θ:");
                self.params.zenith.ui(ui);
                ui.end_row();
                ui.label("Azimuthal angle φ:");
                self.params.azimuth.ui(ui);
                ui.end_row();
                ui.label("Micro-surfaces:");
                egui::CollapsingHeader::new("micro-surfaces")
                    .default_open(self.selected.len() < 3)
                    .show(ui, |ui| {
                        ui.columns(1, |uis| {
                            let ui = &mut uis[0];
                            for hdl in &self.loaded {
                                ui.label(hdl.to_string());
                            }
                        });
                    });
                ui.columns(1, |uis| {
                    let ui = &mut uis[0];
                    if self.selected.is_empty() {
                        ui.label("No micro-surface selected");
                    } else {
                        for hdl in &self.selected {
                            ui.label(hdl.to_string());
                        }
                    }
                });
                ui.end_row();
                ui.label("");
                ui.horizontal_centered(|ui| {
                    egui::ComboBox::from_label("Add").show_ui(ui, |ui| {
                        // for hdl in &self.loaded {
                        //     ui.selectable_value(hdl, hdl.to_string(),
                        // hdl.to_string());
                        // }
                    });
                    if ui.button("Clear").clicked() {
                        println!("Clear micro-surface");
                    }
                    if ui.button("Remove").clicked() {
                        println!("Remove micro-surface");
                    }
                });
            });
        if ui.button("Run").clicked() {
            println!("Run Madf simulation");
        }
    }

    pub fn update_loaded_surfaces(&mut self, loaded: &Vec<Handle<MicroSurface>>) {
        self.loaded.extend(loaded.iter().cloned());
    }
}

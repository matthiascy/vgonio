use crate::measure::measurement::MmsfMeasurementParams;

pub struct MmsfSimulation {
    pub params: MmsfMeasurementParams,
}

impl MmsfSimulation {
    pub fn new() -> Self {
        Self {
            params: MmsfMeasurementParams::default(),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("mmsf_sim_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("Madf Simulation");
                ui.label("TODO");
            });
    }
}

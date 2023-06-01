use crate::measure::measurement::BsdfMeasurementParams;

pub struct BsdfSimulation {
    pub params: BsdfMeasurementParams,
}

impl BsdfSimulation {
    pub fn new() -> Self {
        BsdfSimulation {
            params: BsdfMeasurementParams::default(),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("bsdf_sim_grid")
            .striped(true)
            .show(ui, |ui| {
                ui.label("Madf Simulation");
                ui.label("TODO");
            });
    }
}

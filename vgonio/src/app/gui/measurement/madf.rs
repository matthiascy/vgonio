use crate::{app::gui::event::EventLoopProxy, measure::params::AdfMeasurementParams};

#[derive(Debug)]
pub struct AdfMeasurementTab {
    pub params: AdfMeasurementParams,
    event_loop: EventLoopProxy,
}

impl AdfMeasurementTab {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: AdfMeasurementParams::default(),
            event_loop,
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
            });
    }
}

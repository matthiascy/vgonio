use crate::{app::gui::event::EventLoopProxy, measure::params::MsfMeasurementParams};

pub struct MsfMeasurementTab {
    pub params: MsfMeasurementParams,
    event_loop: EventLoopProxy,
}

impl MsfMeasurementTab {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: MsfMeasurementParams::default(),
            event_loop,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("mmsf_sim_grid")
            .striped(true)
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Zenith angle θ:");
                self.params.zenith.ui(ui);
                ui.end_row();
                ui.label("Azimuthal angle φ:");
                self.params.azimuth.ui(ui);
                ui.end_row();
                ui.label("GPU Texture resolution:");
                ui.add(
                    egui::DragValue::new(&mut self.params.resolution)
                        .speed(1.0)
                        .clamp_range(256.0..=2048.0),
                );
                ui.end_row();
            });
    }
}

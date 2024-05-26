use crate::{
    app::gui::{event::EventLoopProxy, misc::range_step_size_inclusive_angle_ui},
    measure::params::MsfMeasurementParams,
};

pub struct MsfMeasurementTab {
    pub params: MsfMeasurementParams,
    _event_loop: EventLoopProxy,
}

impl MsfMeasurementTab {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: MsfMeasurementParams::default(),
            _event_loop: event_loop,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("mmsf_sim_grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Zenith angle θ:");
                range_step_size_inclusive_angle_ui(&mut self.params.zenith, ui);
                ui.end_row();
                ui.label("Azimuthal angle φ:");
                range_step_size_inclusive_angle_ui(&mut self.params.azimuth, ui);
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

use crate::{
    app::gui::{widgets::SurfaceSelector, MeasureEvent, VgonioEvent, VgonioEventLoop},
    measure::measurement::MmsfMeasurementParams,
};

pub struct MmsfSimulation {
    pub params: MmsfMeasurementParams,
    pub(crate) selector: SurfaceSelector,
    event_loop: VgonioEventLoop,
}

impl MmsfSimulation {
    pub fn new(event_loop: VgonioEventLoop) -> Self {
        Self {
            params: MmsfMeasurementParams::default(),
            event_loop,
            selector: SurfaceSelector::multiple(),
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
                ui.label("Micro-surfaces:");
                self.selector.ui("micro_surface_selector", ui);
                ui.end_row();
            });
        if ui.button("Simulate").clicked() {
            self.event_loop
                .send_event(VgonioEvent::Measure(MeasureEvent::Mmsf {
                    params: self.params,
                    surfaces: self.selector.selected().collect(),
                }))
                .unwrap();
        }
    }
}

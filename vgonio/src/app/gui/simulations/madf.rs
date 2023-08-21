use crate::{
    app::gui::{
        event::{EventLoopProxy, MeasureEvent, VgonioEvent},
        widgets::SurfaceSelector,
    },
    measure::measurement::MndfMeasurementParams,
};

#[derive(Debug)]
pub struct MadfSimulation {
    pub params: MndfMeasurementParams,
    pub(crate) selector: SurfaceSelector,
    event_loop: EventLoopProxy,
}

impl MadfSimulation {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: MndfMeasurementParams::default(),
            event_loop,
            selector: SurfaceSelector::multiple(),
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
                self.selector.ui("micro_surface_selector", ui);
                ui.end_row();
            });
        if ui.button("Simulate").clicked() {
            // TODO: launch simulation on a separate thread and show progress bar
            self.event_loop
                .send_event(VgonioEvent::Measure(MeasureEvent::Madf {
                    params: self.params,
                    surfaces: self.selector.selected().collect(),
                }))
                .unwrap();
        }
    }
}

use crate::{
    app::gui::{simulations::SurfaceSelector, VgonioEvent, VgonioEventLoop},
    measure::measurement::MadfMeasurementParams,
};
use winit::event_loop::EventLoopProxy;

#[derive(Debug)]
pub struct MadfSimulation {
    pub params: MadfMeasurementParams,
    pub(crate) selector: SurfaceSelector,
    event_loop: VgonioEventLoop,
}

impl MadfSimulation {
    pub fn new(event_loop: VgonioEventLoop) -> Self {
        Self {
            params: MadfMeasurementParams::default(),
            event_loop,
            selector: SurfaceSelector::default(),
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
                .send_event(VgonioEvent::MeasureAreaDistribution {
                    params: self.params,
                    surfaces: self.selector.selected.clone().into_iter().collect(),
                })
                .unwrap();
        }
    }
}
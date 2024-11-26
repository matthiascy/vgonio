use crate::{
    app::gui::{event::EventLoopProxy, misc, misc::range_step_size_inclusive_angle_ui},
    measure::params::{NdfMeasurementMode, NdfMeasurementParams},
};
use egui::Widget;

#[derive(Debug)]
pub struct NdfMeasurementTab {
    pub params: NdfMeasurementParams,
    _event_loop: EventLoopProxy,
}

impl NdfMeasurementTab {
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: NdfMeasurementParams::default(),
            _event_loop: event_loop,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("ndf_sim_grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.checkbox(&mut self.params.crop_to_disk, "Crop to disk")
                    .on_hover_text(
                        "Crop the surface measurement area to disk. This is useful for simulating \
                         a circular sample.",
                    );
                ui.end_row();

                ui.checkbox(&mut self.params.use_facet_area, "Use facet area")
                    .on_hover_text(
                        "Use the facet area instead of number of facets for the measurement.",
                    );
                ui.end_row();

                ui.selectable_value(
                    &mut self.params.mode,
                    NdfMeasurementMode::default_by_points(),
                    "By Points",
                );
                ui.selectable_value(
                    &mut self.params.mode,
                    NdfMeasurementMode::default_by_partition(),
                    "By Partition",
                );
                ui.end_row();

                match &mut self.params.mode {
                    NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                        ui.label("Zenith angle θ:");
                        range_step_size_inclusive_angle_ui(zenith, ui);
                        ui.end_row();
                        ui.label("Azimuthal angle φ:");
                        range_step_size_inclusive_angle_ui(azimuth, ui);
                        ui.end_row();
                    },
                    NdfMeasurementMode::ByPartition { precision } => {
                        ui.label("Partition precision:");
                        ui.horizontal_wrapped(|ui| {
                            misc::drag_angle(precision, "θ").ui(ui);
                        });
                    },
                }
            });
    }
}

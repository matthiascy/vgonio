use crate::{
    app::gui::{event::EventLoopProxy, misc, misc::range_step_size_inclusive_angle_ui},
    measure::params::{AdfMeasurementMode, AdfMeasurementParams},
};
use base::partition::PartitionScheme;
use egui::Widget;

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
                    AdfMeasurementMode::default_by_points(),
                    "By Points",
                );
                ui.selectable_value(
                    &mut self.params.mode,
                    AdfMeasurementMode::default_by_partition(),
                    "By Partition",
                );
                ui.end_row();

                match &mut self.params.mode {
                    AdfMeasurementMode::ByPoints { azimuth, zenith } => {
                        ui.label("Zenith angle θ:");
                        range_step_size_inclusive_angle_ui(zenith, ui);
                        ui.end_row();
                        ui.label("Azimuthal angle φ:");
                        range_step_size_inclusive_angle_ui(azimuth, ui);
                        ui.end_row();
                    }
                    AdfMeasurementMode::ByPartition { scheme, precision } => {
                        ui.label("Partition scheme:");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(scheme, PartitionScheme::Beckers, "Beckers");
                            ui.selectable_value(scheme, PartitionScheme::EqualAngle, "EqualAngle");
                        });
                        ui.end_row();

                        ui.label("Partition precision:");
                        ui.horizontal_wrapped(|ui| {
                            misc::drag_angle(&mut precision.theta, "θ").ui(ui);
                            if scheme == &PartitionScheme::EqualAngle {
                                misc::drag_angle(&mut precision.phi, "φ").ui(ui);
                            }
                        });
                    }
                }
            });
    }
}

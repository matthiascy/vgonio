use crate::{
    app::gui::{event::EventLoopProxy, misc},
    measure::params::{AdfMeasurementMode, AdfMeasurementParams},
    partition::PartitionScheme,
};
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
                ui.checkbox(&mut self.params.crop_to_disk, "Crop to disk");
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
                        zenith.ui(ui);
                        ui.end_row();
                        ui.label("Azimuthal angle φ:");
                        azimuth.ui(ui);
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

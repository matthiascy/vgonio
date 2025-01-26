#[cfg(feature = "vdbg")]
use crate::app::gui::{DebuggingEvent, VgonioEvent};
use crate::{
    app::gui::{
        event::EventLoopProxy,
        misc::{range_step_size_inclusive_angle_ui, range_step_size_inclusive_length_ui},
    },
    measure::{
        bsdf::{emitter::EmitterParams, rtc::RtcMethod, BsdfKind},
        params::{BsdfMeasurementParams, SimulationKind},
    },
};
use vgcore::utils::{medium::Medium, partition::SphericalDomain};
#[cfg(feature = "vdbg")]
use vgcore::{math::Sph2, units::Rads};
use std::hash::Hash;
use uxtk::widgets::ToggleSwitch;

impl BsdfKind {
    /// Creates the UI for selecting the BSDF kind.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_salt(id_source)
            .selected_text(format!("{}", self))
            .show_ui(ui, |ui| {
                ui.selectable_value(self, BsdfKind::Brdf, "BRDF");
                ui.selectable_value(self, BsdfKind::Btdf, "BTDF");
                ui.selectable_value(self, BsdfKind::Bssdf, "BSSDF");
                ui.selectable_value(self, BsdfKind::Bssrdf, "BSSRDF");
                ui.selectable_value(self, BsdfKind::Bssrdf, "BSSRDF");
            });
    }
}

pub fn medium_selectable_ui(medium: &mut Medium, id_source: impl Hash, ui: &mut egui::Ui) {
    egui::ComboBox::from_id_salt(id_source)
        .selected_text(format!("{:?}", medium))
        .show_ui(ui, |ui| {
            ui.selectable_value(medium, Medium::Air, "Air");
            ui.selectable_value(medium, Medium::Copper, "Copper");
            ui.selectable_value(medium, Medium::Aluminium, "Aluminium");
            ui.selectable_value(medium, Medium::Vacuum, "Vacuum");
        });
}

impl EmitterParams {
    /// Creates the UI for parameterizing the emitter.
    pub fn ui<R>(
        &mut self,
        ui: &mut egui::Ui,
        add_contents: impl FnOnce(&mut EmitterParams, &mut egui::Ui) -> R,
    ) {
        egui::CollapsingHeader::new("Emitter")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("emitter_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Number of rays: ");
                        ui.add(
                            egui::DragValue::new(&mut self.num_rays)
                                .speed(1.0)
                                .range(1..=100_000_000),
                        );
                        ui.end_row();

                        ui.label("Max. bounces: ");
                        ui.add(
                            egui::DragValue::new(&mut self.max_bounces)
                                .speed(1.0)
                                .range(1..=100),
                        );
                        ui.end_row();

                        ui.label("Azimuthal range φ: ");
                        range_step_size_inclusive_angle_ui(&mut self.azimuth, ui);
                        ui.end_row();

                        ui.label("Zenith range θ: ");
                        range_step_size_inclusive_angle_ui(&mut self.zenith, ui);
                        ui.end_row();

                        ui.label("Wavelength range: ");
                        range_step_size_inclusive_length_ui(&mut self.spectrum, ui);
                        ui.end_row();

                        add_contents(self, ui);
                    });
            });
    }
}

pub fn spherical_domain_ui(domain: &mut SphericalDomain, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        if ui
            .selectable_label(*domain == SphericalDomain::Upper, "Upper")
            .on_hover_text("The upper hemisphere.")
            .clicked()
        {
            *domain = SphericalDomain::Upper;
        }

        if ui
            .selectable_label(*domain == SphericalDomain::Lower, "Lower")
            .on_hover_text("The lower hemisphere.")
            .clicked()
        {
            *domain = SphericalDomain::Lower;
        }

        if ui
            .selectable_label(*domain == SphericalDomain::Whole, "Whole")
            .on_hover_text("The whole sphere.")
            .clicked()
        {
            *domain = SphericalDomain::Whole;
        }
    });
}

pub struct BsdfMeasurementTab {
    pub params: BsdfMeasurementParams,
    event_loop: EventLoopProxy,
    #[cfg(feature = "vdbg")]
    debug: BsdfMeasurementDebug,
}

impl BsdfMeasurementTab {
    /// Creates a new BSDF simulation UI.
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: BsdfMeasurementParams::default(),
            event_loop,
            #[cfg(feature = "vdbg")]
            debug: BsdfMeasurementDebug {
                detector_dome_drawing: false,
                emitter_samples_drawing: false,
                emitter_ray_param_t: 0.0,
                emitter_rays_drawing: false,
                measurement_points_drawing: false,
                measurement_point_index: 0,
                ray_trajectory_index: 0,
                ray_trajectories_drawing_reflected: false,
                ray_trajectories_drawing_missed: false,
                ray_hit_points_drawing: false,
            },
        }
    }

    #[cfg(feature = "vdbg")]
    pub fn measurement_point(&self) -> Sph2 {
        let zenith_count = self.params.emitter.zenith.step_count_wrapped();
        let azimuth_idx = self.debug.measurement_point_index / zenith_count as i32;
        let zenith_idx = self.debug.measurement_point_index % zenith_count as i32;
        let zenith = self.params.emitter.zenith.step(zenith_idx as usize);
        let azimuth = self.params.emitter.azimuth.step(azimuth_idx as usize);
        Sph2::new(zenith, azimuth)
    }

    /// UI for BSDF simulation parameters.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        #[cfg(feature = "vdbg")] debug_draw: bool,
        #[cfg(feature = "vdbg")] orbit_radius: f32,
    ) {
        egui::CollapsingHeader::new("Parameters")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("new_bsdf_measurement_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("BSDF type:");
                        self.params.kind.selectable_ui("bsdf_kind_choice", ui);
                        ui.end_row();

                        ui.label("Incident medium:");
                        medium_selectable_ui(
                            &mut self.params.incident_medium,
                            "incident_medium_choice",
                            ui,
                        );
                        ui.end_row();

                        ui.label("Surface medium:");
                        medium_selectable_ui(
                            &mut self.params.transmitted_medium,
                            "transmitted_medium_choice",
                            ui,
                        );
                        ui.end_row();

                        ui.label("Fresnel reflection:");
                        ui.checkbox(&mut self.params.fresnel, "Fresnel reflection");
                        ui.end_row();

                        ui.label("Simulation kind: ");
                        ui.horizontal_wrapped(|ui| {
                            #[cfg(feature = "embree")]
                            ui.selectable_value(
                                &mut self.params.sim_kind,
                                SimulationKind::GeomOptics(RtcMethod::Embree),
                                "Geometric optics",
                            );
                            ui.selectable_value(
                                &mut self.params.sim_kind,
                                SimulationKind::WaveOptics,
                                "Wave optics",
                            );
                        });
                        ui.end_row();

                        if self.params.sim_kind != SimulationKind::WaveOptics {
                            ui.label("Ray tracing method: ");
                            ui.horizontal_wrapped(|ui| {
                                #[cfg(feature = "embree")]
                                ui.selectable_value(
                                    &mut self.params.sim_kind,
                                    SimulationKind::GeomOptics(RtcMethod::Embree),
                                    "Embree",
                                );
                                #[cfg(feature = "optix")]
                                ui.selectable_value(
                                    &mut self.params.sim_kind,
                                    SimulationKind::GeomOptics(RtcMethod::Optix),
                                    "Optix",
                                );
                                ui.selectable_value(
                                    &mut self.params.sim_kind,
                                    SimulationKind::GeomOptics(RtcMethod::Grid),
                                    "Grid",
                                );
                            });
                            ui.end_row();
                        }
                    });
                // TODO: Emitter sector
                #[cfg(feature = "vdbg")]
                self.params.emitter.ui(ui, |params, ui| {
                    if debug_draw {
                        ui.label("Samples");
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .button("Generate")
                                .on_hover_text("Generate samples")
                                .clicked()
                            {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::UpdateEmitterSamples(
                                        params.generate_unit_samples(
                                            Rads::ZERO,
                                            Rads::TAU,
                                            params.num_rays as usize,
                                        ),
                                    ),
                                ));
                            }
                            if ui
                                .add(ToggleSwitch::new(&mut self.debug.emitter_samples_drawing))
                                .changed()
                            {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleEmitterSamplesDrawing(
                                        self.debug.emitter_samples_drawing,
                                    ),
                                ));
                            }
                        });
                        ui.end_row();

                        ui.label("Sample rays:");
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .add(ToggleSwitch::new(&mut self.debug.emitter_rays_drawing))
                                .changed()
                            {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleEmitterRaysDrawing(
                                        self.debug.emitter_rays_drawing,
                                    ),
                                ));

                                if self.debug.emitter_rays_drawing {
                                    self.event_loop.send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::EmitRays,
                                    ));
                                }
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.debug.emitter_ray_param_t,
                                        1.0..=orbit_radius * 2.0,
                                    )
                                    .text("t"),
                                )
                                .changed()
                            {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::UpdateRayParams {
                                        t: self.debug.emitter_ray_param_t,
                                    },
                                ));
                            }
                        });
                        ui.end_row();

                        ui.label("Measurement Points:");
                        ui.horizontal_wrapped(|ui| {
                            let zenith_count = params.zenith.step_count_wrapped();
                            let count = zenith_count * params.azimuth.step_count_wrapped();

                            let old_index = self.debug.measurement_point_index;

                            if ui.button("\u{25C0}").clicked() {
                                self.debug.measurement_point_index =
                                    (self.debug.measurement_point_index - 1).max(0) % count as i32;
                            }

                            if ui.button("\u{25B6}").clicked() {
                                self.debug.measurement_point_index =
                                    (self.debug.measurement_point_index + 1).max(0) % count as i32;
                            }

                            if old_index != self.debug.measurement_point_index {
                                let azimuth_idx =
                                    self.debug.measurement_point_index / zenith_count as i32;
                                let zenith_idx =
                                    self.debug.measurement_point_index % zenith_count as i32;
                                let zenith = params.zenith.step(zenith_idx as usize);
                                let azimuth = params.azimuth.step(azimuth_idx as usize);

                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::UpdateEmitterPosition {
                                        position: Sph2::new(zenith, azimuth),
                                    },
                                ));
                            }

                            if ui
                                .add(ToggleSwitch::new(
                                    &mut self.debug.measurement_points_drawing,
                                ))
                                .changed()
                            {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleMeasurementPointsDrawing(
                                        self.debug.measurement_points_drawing,
                                    ),
                                ));
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::UpdateMeasurementPoints(
                                        params.generate_measurement_points(),
                                    ),
                                ));
                            }
                        });
                        ui.end_row();
                    }
                });
                // TODO: Add multiple receivers
                #[cfg(feature = "vdbg")]
                for receiver in &mut self.params.receivers {
                    receiver.ui(ui, |params, ui| {
                        if debug_draw {
                            ui.label("Dome:");
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("update").clicked() {
                                    self.event_loop.send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::UpdateDetectorPatches(
                                            params.partitioning(),
                                        ),
                                    ));
                                }
                                if ui
                                    .add(ToggleSwitch::new(&mut self.debug.detector_dome_drawing))
                                    .changed()
                                {
                                    self.event_loop.send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::ToggleDetectorDomeDrawing(
                                            self.debug.detector_dome_drawing,
                                        ),
                                    ));
                                }
                            });
                            ui.end_row();

                            ui.label("Trajectories:");
                            ui.horizontal_wrapped(|ui| {
                                let mut ray_trajectory_index_changed = false;
                                if ui.button("\u{25C0}").clicked() {
                                    self.debug.ray_trajectory_index =
                                        (self.debug.ray_trajectory_index - 1).max(0);
                                    ray_trajectory_index_changed = true;
                                }
                                if ui.button("\u{25B6}").clicked() {
                                    self.debug.ray_trajectory_index =
                                        (self.debug.ray_trajectory_index + 1).max(0);
                                    ray_trajectory_index_changed = true;
                                }
                                ui.label("reflected");
                                let changed0 = ui
                                    .add(ToggleSwitch::new(
                                        &mut self.debug.ray_trajectories_drawing_reflected,
                                    ))
                                    .changed();
                                ui.label("missed");
                                let changed1 = ui
                                    .add(ToggleSwitch::new(
                                        &mut self.debug.ray_trajectories_drawing_missed,
                                    ))
                                    .changed();
                                if changed0 || changed1 || ray_trajectory_index_changed {
                                    self.event_loop.send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::UpdateRayTrajectoriesDrawing {
                                            index: self.debug.ray_trajectory_index as usize,
                                            missed: self.debug.ray_trajectories_drawing_missed,
                                            reflected: self
                                                .debug
                                                .ray_trajectories_drawing_reflected,
                                        },
                                    ));
                                }
                            });
                            ui.end_row();

                            ui.label("Collected rays:");
                            if ui
                                .add(ToggleSwitch::new(&mut self.debug.ray_hit_points_drawing))
                                .changed()
                            {
                                self.event_loop.send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleCollectedRaysDrawing(
                                        self.debug.ray_hit_points_drawing,
                                    ),
                                ));
                            }
                        }
                    });
                }
            });
    }
}

#[cfg(feature = "vdbg")]
pub struct BsdfMeasurementDebug {
    pub detector_dome_drawing: bool,
    pub emitter_samples_drawing: bool,
    pub emitter_ray_param_t: f32,
    pub emitter_rays_drawing: bool,
    pub measurement_points_drawing: bool,
    pub measurement_point_index: i32,
    pub ray_trajectory_index: i32,
    pub ray_trajectories_drawing_reflected: bool,
    pub ray_trajectories_drawing_missed: bool,
    pub ray_hit_points_drawing: bool,
}

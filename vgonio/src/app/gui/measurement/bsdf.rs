use crate::{
    app::gui::{
        event::{EventLoopProxy, MeasureEvent, VgonioEvent},
        widgets::{SurfaceSelector, ToggleSwitch},
    },
    measure::{
        bsdf::{
            detector::{DetectorParams, DetectorScheme},
            emitter::EmitterParams,
            rtc::RtcMethod,
            BsdfKind,
        },
        params::{BsdfMeasurementParams, SimulationKind},
    },
    Medium, SphericalDomain,
};
use std::hash::Hash;

impl BsdfKind {
    /// Creates the UI for selecting the BSDF kind.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_source(id_source)
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

impl Medium {
    /// Creates the UI for selecting the medium.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_source(id_source)
            .selected_text(format!("{:?}", self))
            .show_ui(ui, |ui| {
                ui.selectable_value(self, Medium::Air, "Air");
                ui.selectable_value(self, Medium::Copper, "Copper");
                ui.selectable_value(self, Medium::Aluminium, "Aluminium");
                ui.selectable_value(self, Medium::Vacuum, "Vacuum");
            });
    }
}

impl EmitterParams {
    /// Creates the UI for parameterizing the emitter.
    pub fn ui<R>(&mut self, ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui) -> R) {
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
                                .clamp_range(1..=100_000_000),
                        );
                        ui.end_row();

                        ui.label("Max. bounces: ");
                        ui.add(
                            egui::DragValue::new(&mut self.max_bounces)
                                .speed(1.0)
                                .clamp_range(1..=100),
                        );
                        ui.end_row();

                        ui.label("Azimuthal range φ: ");
                        self.azimuth.ui(ui);
                        ui.end_row();

                        ui.label("Zenith range θ: ");
                        self.zenith.ui(ui);
                        ui.end_row();

                        ui.label("Wavelength range: ");
                        self.spectrum.ui(ui);
                        ui.end_row();

                        add_contents(ui);
                    });
            });
    }
}

impl SphericalDomain {
    /// Creates the UI for parameterizing the spherical domain.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(*self == SphericalDomain::Upper, "Upper")
                .on_hover_text("The upper hemisphere.")
                .clicked()
            {
                *self = SphericalDomain::Upper;
            }

            if ui
                .selectable_label(*self == SphericalDomain::Lower, "Lower")
                .on_hover_text("The lower hemisphere.")
                .clicked()
            {
                *self = SphericalDomain::Lower;
            }

            if ui
                .selectable_label(*self == SphericalDomain::Whole, "Whole")
                .on_hover_text("The whole sphere.")
                .clicked()
            {
                *self = SphericalDomain::Whole;
            }
        });
    }
}

pub struct BsdfMeasurementTab {
    pub params: BsdfMeasurementParams,
    event_loop: EventLoopProxy,
    #[cfg(debug_assertions)]
    debug: BsdfMeasurementDebug,
}

impl BsdfMeasurementTab {
    /// Creates a new BSDF simulation UI.
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: BsdfMeasurementParams::default(),
            event_loop,
            debug: BsdfMeasurementDebug {
                detector_dome_drawing: false,
                emitter_samples_drawing: false,
                emitter_ray_param_t: 0.0,
                emitter_rays_drawing: false,
                measurement_points_drawing: false,
                ray_trajectories_drawing_reflected: false,
                ray_trajectories_drawing_missed: false,
                ray_hit_points_drawing: false,
            },
        }
    }

    /// UI for BSDF simulation parameters.
    pub fn ui(&mut self, ui: &mut egui::Ui, #[cfg(debug_assertions)] debug_draw: bool) {
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
                        self.params
                            .incident_medium
                            .selectable_ui("incident_medium_choice", ui);
                        ui.end_row();

                        ui.label("Surface medium:");
                        self.params
                            .transmitted_medium
                            .selectable_ui("transmitted_medium_choice", ui);
                        ui.end_row();

                        ui.label("Simulation kind: ");
                        ui.horizontal_wrapped(|ui| {
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
                                ui.selectable_value(
                                    &mut self.params.sim_kind,
                                    SimulationKind::GeomOptics(RtcMethod::Embree),
                                    "Embree",
                                );
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
                self.params.emitter.ui(ui, |ui| {
                    #[cfg(debug_assertions)]
                    {
                        if debug_draw {
                            ui.label("Samples");
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .button("Generate")
                                    .on_hover_text("Generate samples")
                                    .clicked()
                                {
                                    // TODO: Generate samples.
                                }
                                if ui
                                    .add(ToggleSwitch::new(&mut self.debug.emitter_samples_drawing))
                                    .changed()
                                {
                                    // TODO: Draw samples.
                                }
                            });
                            ui.end_row();

                            ui.label("Sample rays:");
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .add(ToggleSwitch::new(&mut self.debug.emitter_rays_drawing))
                                    .changed()
                                {
                                    // TODO: Draw sample rays.
                                }

                                // TODO: calc max t
                                if ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.debug.emitter_ray_param_t,
                                            1.0..=10.0,
                                        )
                                        .text("t"),
                                    )
                                    .changed()
                                {
                                    // TODO: Draw sample rays.
                                }
                            });
                            ui.end_row();

                            ui.label("Measurement Points:");
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("\u{25C0}").clicked() {
                                    // TODO: Decrement measurement point index.
                                }
                                if ui.button("\u{25B6}").clicked() {
                                    // TODO: Increment measurement point index.
                                }
                                if ui
                                    .add(ToggleSwitch::new(
                                        &mut self.debug.measurement_points_drawing,
                                    ))
                                    .changed()
                                {
                                    // TODO: Draw measurement points.
                                }
                            });
                            ui.end_row();
                        }
                    }
                });
                self.params.detector.ui(ui, |ui| {
                    #[cfg(debug_assertions)]
                    {
                        if debug_draw {
                            ui.label("Dome:");
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("update").clicked() {
                                    // TODO: Update dome.
                                }
                                if ui
                                    .add(ToggleSwitch::new(&mut self.debug.detector_dome_drawing))
                                    .changed()
                                {
                                    // TODO: Draw dome.
                                }
                            });
                            ui.end_row();

                            ui.label("Trajectories:");
                            ui.horizontal_wrapped(|ui| {
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
                                if changed0 || changed1 {
                                    // TODO: Draw trajectories.
                                }
                            });
                            ui.end_row();

                            ui.label("Collected data:");
                            if ui
                                .add(ToggleSwitch::new(&mut self.debug.ray_hit_points_drawing))
                                .changed()
                            {
                                // TODO: Draw collected data.
                            }
                        }
                    }
                });
            });
    }
}

pub struct BsdfMeasurementDebug {
    pub detector_dome_drawing: bool,
    pub emitter_samples_drawing: bool,
    pub emitter_ray_param_t: f32,
    pub emitter_rays_drawing: bool,
    pub measurement_points_drawing: bool,
    pub ray_trajectories_drawing_reflected: bool,
    pub ray_trajectories_drawing_missed: bool,
    pub ray_hit_points_drawing: bool,
}

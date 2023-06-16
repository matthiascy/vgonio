use crate::{
    app::{
        cache::{Cache, Handle, MicroSurfaceRecord},
        gui::{
            misc::{input3_spherical, input3_xyz},
            widgets::ToggleSwitch,
            DebuggingEvent, VgonioEvent, VgonioEventLoop,
        },
    },
    math,
    measure::{
        emitter::RegionShape,
        measurement::{BsdfMeasurementParams, Radius},
        rtc::Ray,
        Emitter, RtcMethod,
    },
    msurf::MicroSurface,
    units::{mm, rad, UMillimetre},
    Handedness,
};
use egui_toast::{Toast, ToastKind, ToastOptions, Toasts};
use glam::{IVec2, Vec3};
use std::sync::{Arc, RwLock};
use winit::event_loop::EventLoopProxy;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RayMode {
    Cartesian,
    Spherical,
}

pub(crate) struct BrdfMeasurementDebugging {
    pub show_dome: bool,
    cache: Arc<RwLock<Cache>>,
    emitter_position_index: i32,
    params: BsdfMeasurementParams,
    pub loaded_surfaces: Vec<MicroSurfaceRecord>,
    pub selected_surface: Option<Handle<MicroSurface>>,
    ray_t: f32,
    orbit_radius: f32,
    event_loop: VgonioEventLoop,

    ray_origin_cartesian: Vec3,
    ray_origin_spherical: Vec3,
    ray_target: Vec3,
    ray_mode: RayMode,
    method: RtcMethod,
    prim_id: u32,
    cell_pos: IVec2,
}

impl BrdfMeasurementDebugging {
    pub fn new(event_loop: VgonioEventLoop, cache: Arc<RwLock<Cache>>) -> Self {
        Self {
            show_dome: false,
            cache,
            ray_origin_cartesian: Vec3::new(0.0, 5.0, 0.0),
            ray_origin_spherical: Vec3::new(5.0, 0.0, 0.0),
            ray_target: Default::default(),
            ray_mode: RayMode::Cartesian,
            method: RtcMethod::Grid,
            prim_id: 0,
            cell_pos: Default::default(),
            ray_t: 1.0,
            emitter_position_index: 0,
            params: BsdfMeasurementParams::default(),
            loaded_surfaces: vec![],
            selected_surface: None,
            event_loop,
            orbit_radius: 1.0,
        }
    }

    fn estimate_radii(&self, record: Option<&MicroSurfaceRecord>) -> Option<(f32, Option<f32>)> {
        match record {
            None => {
                self.event_loop
                    .send_event(VgonioEvent::Notify {
                        kind: ToastKind::Warning,
                        text: "No surface selected to evaluate the radius".to_string(),
                        time: 3.0,
                    })
                    .unwrap();
                None
            }
            Some(record) => {
                let cache = self.cache.read().unwrap();
                let mesh = cache.get_micro_surface_mesh(record.mesh).unwrap();
                let (radius, disk_radius) = match self.params.emitter.shape {
                    RegionShape::SphericalCap { .. } | RegionShape::SphericalRect { .. } => {
                        (self.params.emitter.radius.estimate(mesh), None)
                    }
                    RegionShape::Disk { radius } => (
                        self.params.emitter.radius.estimate(mesh),
                        Some(radius.estimate_disk_radius(mesh)),
                    ),
                };
                log::trace!(
                    "[BsdfMeasurementDebugging] evaluated radius: {}, {:?}",
                    radius,
                    disk_radius
                );
                Some((radius, disk_radius))
            }
        }
    }

    fn update_emitter_position_index(&mut self, delta: i32) {
        let record = self
            .loaded_surfaces
            .iter()
            .find(|s| s.surf == self.selected_surface.unwrap_or(Handle::default()));
        if let Some((orbit_radius, shape_radius)) = self.estimate_radii(record) {
            self.orbit_radius = orbit_radius;
            let zenith_step_count = self.params.emitter.zenith.step_count_wrapped();
            let azimuth_step_count = self.params.emitter.azimuth.step_count_wrapped();
            self.emitter_position_index = ((self.emitter_position_index + delta).max(0) as usize
                % (zenith_step_count * azimuth_step_count))
                as i32;
            let azimuth_idx = self.emitter_position_index / zenith_step_count as i32;
            let zenith_idx = self.emitter_position_index % zenith_step_count as i32;
            let zenith = self.params.emitter.zenith.step(zenith_idx as usize);
            let azimuth = self.params.emitter.azimuth.step(azimuth_idx as usize);
            log::trace!(
                "[BrdfMeasurementDebugging] updating emitter position index: {}, zenith: {}, \
                 index={}, azimuth: {}, index={}",
                self.emitter_position_index,
                zenith.to_degrees(),
                zenith_idx,
                azimuth.to_degrees(),
                azimuth_idx
            );
            self.event_loop
                .send_event(VgonioEvent::Debugging(
                    DebuggingEvent::UpdateEmitterPosition {
                        zenith,
                        azimuth,
                        orbit_radius,
                        shape_radius,
                    },
                ))
                .unwrap();
        }
    }
}

impl egui::Widget for &mut BrdfMeasurementDebugging {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.label("MicroSurface");
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_source("MicroSurface")
                    .selected_text(format!(
                        "{}",
                        match self.selected_surface {
                            None => "None",
                            Some(hdl) => {
                                let record =
                                    self.loaded_surfaces.iter().find(|s| s.surf == hdl).unwrap();
                                record.name()
                            }
                        }
                    ))
                    .show_ui(ui, |ui| {
                        for record in &self.loaded_surfaces {
                            ui.selectable_value(
                                &mut self.selected_surface,
                                Some(record.surf),
                                format!("{}", record.name()),
                            );
                        }
                    });
            });
        });

        egui::CollapsingHeader::new("Emitter")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("emitter_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Number of rays: ");
                        ui.add(
                            egui::DragValue::new(&mut self.params.emitter.num_rays)
                                .speed(1.0)
                                .clamp_range(1..=100_000_000),
                        );
                        ui.end_row();

                        ui.label("Max. bounces: ");
                        ui.add(
                            egui::DragValue::new(&mut self.params.emitter.max_bounces)
                                .speed(1.0)
                                .clamp_range(1..=100),
                        );
                        ui.end_row();

                        ui.label("Distance:")
                            .on_hover_text("Distance from emitter to the surface.");
                        ui.horizontal_wrapped(|ui| {
                            self.params.emitter.radius.ui(ui);
                            #[cfg(debug_assertions)]
                            if ui.button("debug_eval").clicked() {
                                let record = self.loaded_surfaces.iter().find(|s| {
                                    s.surf == self.selected_surface.unwrap_or(Handle::default())
                                });
                                match record {
                                    None => {
                                        self.event_loop
                                            .send_event(VgonioEvent::Notify {
                                                kind: ToastKind::Warning,
                                                text: "No surface selected to evaluate the dome \
                                                       radius"
                                                    .to_string(),
                                                time: 3.0,
                                            })
                                            .unwrap();
                                    }
                                    Some(record) => {
                                        let cache = self.cache.read().unwrap();
                                        let mesh =
                                            cache.get_micro_surface_mesh(record.mesh).unwrap();
                                        let radius = self.params.emitter.radius.estimate(mesh);
                                        self.event_loop
                                            .send_event(VgonioEvent::Notify {
                                                kind: ToastKind::Warning,
                                                text: format!(
                                                    "Evaluated emitter radius = {radius}"
                                                ),
                                                time: 5.0,
                                            })
                                            .unwrap();
                                    }
                                }
                            }
                            if ui.add(ToggleSwitch::new(&mut self.show_dome)).changed() {
                                self.event_loop
                                    .send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::ToggleDebugDrawingDome(self.show_dome),
                                    ))
                                    .unwrap();
                            }
                        });
                        ui.end_row();

                        ui.label("Azimuthal range φ: ");
                        self.params.emitter.azimuth.ui(ui);
                        ui.end_row();

                        ui.label("Zenith range θ: ");
                        self.params.emitter.zenith.ui(ui);
                        ui.end_row();

                        ui.label("Region shape: ");
                        self.params.emitter.shape.ui(ui);
                        ui.end_row();

                        ui.label("Wavelength range: ");
                        self.params.emitter.spectrum.ui(ui);
                        ui.end_row();

                        ui.label("Samples: ");
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Generate").clicked() {
                                let record = self.loaded_surfaces.iter().find(|s| {
                                    s.surf == self.selected_surface.unwrap_or(Handle::default())
                                });
                                if let Some((orbit_radius, shape_radius)) =
                                    self.estimate_radii(record)
                                {
                                    self.orbit_radius = orbit_radius;
                                    self.event_loop
                                        .send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::UpdateEmitterSamples {
                                                samples: self
                                                    .params
                                                    .emitter
                                                    .generate_unit_samples(),
                                                orbit_radius,
                                                shape_radius,
                                            },
                                        ))
                                        .unwrap();
                                }
                            }
                            if ui.button("\u{25C0}").clicked() {
                                self.update_emitter_position_index(-1);
                            }
                            if ui.button("\u{25B6}").clicked() {
                                self.update_emitter_position_index(1);
                            }
                        });
                        ui.end_row();

                        ui.label("Measurement Points: ");
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Display").clicked() {
                                let points = self
                                    .params
                                    .emitter
                                    .measurement_points()
                                    .into_iter()
                                    .map(|s| {
                                        math::spherical_to_cartesian(
                                            1.0,
                                            s.zenith,
                                            s.azimuth,
                                            Handedness::RightHandedYUp,
                                        )
                                    })
                                    .collect();
                                let record = self.loaded_surfaces.iter().find(|s| {
                                    s.surf == self.selected_surface.unwrap_or(Handle::default())
                                });
                                if let Some((orbit_radius, _)) = self.estimate_radii(record) {
                                    self.orbit_radius = orbit_radius;
                                    self.event_loop
                                        .send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::UpdateEmitterPoints {
                                                points,
                                                orbit_radius,
                                            },
                                        ))
                                        .unwrap();
                                }
                            }

                            if ui.button("Emit Rays").clicked() {
                                let record = self.loaded_surfaces.iter().find(|s| {
                                    s.surf == self.selected_surface.unwrap_or(Handle::default())
                                });
                                if let Some((orbit_radius, shape_radius)) =
                                    self.estimate_radii(record)
                                {
                                    self.orbit_radius = orbit_radius;
                                    self.event_loop
                                        .send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::EmitRays {
                                                orbit_radius,
                                                shape_radius,
                                            },
                                        ))
                                        .unwrap();
                                }
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.ray_t,
                                        1.0..=self.orbit_radius * 2.0,
                                    )
                                    .text("t"),
                                )
                                .changed()
                            {
                                let record = self.loaded_surfaces.iter().find(|s| {
                                    s.surf == self.selected_surface.unwrap_or(Handle::default())
                                });
                                if let Some((orbit_radius, shape_radius)) =
                                    self.estimate_radii(record)
                                {
                                    self.event_loop
                                        .send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::UpdateRayParams {
                                                t: self.ray_t,
                                                orbit_radius,
                                                shape_radius,
                                            },
                                        ))
                                        .unwrap();
                                }
                            }
                        });
                        ui.end_row();
                    });
            });

        ui.horizontal_wrapped(|ui| {
            ui.label("Method");
            #[cfg(feature = "embree")]
            ui.selectable_value(&mut self.method, RtcMethod::Embree, "Embree");
            #[cfg(feature = "optix")]
            ui.selectable_value(&mut self.method, RtcMethod::Optix, "OptiX");
            ui.selectable_value(&mut self.method, RtcMethod::Grid, "Grid");
        });

        if self.method == RtcMethod::Grid {
            ui.horizontal_wrapped(|ui| {
                ui.label("cell");
                ui.add(egui::DragValue::new(&mut self.cell_pos.x).prefix("x: "));
                ui.add(egui::DragValue::new(&mut self.cell_pos.y).prefix("y:"));

                if ui.button("show").clicked()
                    && self
                        .event_loop
                        .send_event(VgonioEvent::UpdateCellPos(self.cell_pos))
                        .is_err()
                {
                    log::warn!("Failed to send event VgonioEvent::UpdateCellPos");
                }
            });
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("prim id");
            ui.add(egui::DragValue::new(&mut self.prim_id));

            if ui.button("show").clicked()
                && self
                    .event_loop
                    .send_event(VgonioEvent::UpdatePrimId(self.prim_id))
                    .is_err()
            {
                log::warn!("Failed to send event VgonioEvent::UpdatePrimId");
            }

            if ui.button("prev").clicked() {
                self.prim_id -= 1;
                if self
                    .event_loop
                    .send_event(VgonioEvent::UpdatePrimId(self.prim_id))
                    .is_err()
                {
                    log::warn!("Failed to send event VgonioEvent::UpdatePrimId");
                }
            }

            if ui.button("next").clicked() {
                self.prim_id += 1;
                if self
                    .event_loop
                    .send_event(VgonioEvent::UpdatePrimId(self.prim_id))
                    .is_err()
                {
                    log::warn!("Failed to send event VgonioEvent::UpdatePrimId");
                }
            }
        });

        egui::CollapsingHeader::new("ray")
            .default_open(true)
            .show(ui, |ui| {
                egui::ComboBox::from_id_source("ray_mod")
                    .selected_text(format!("{:?}", self.ray_mode))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.ray_mode, RayMode::Cartesian, "Cartesian");
                        ui.selectable_value(&mut self.ray_mode, RayMode::Spherical, "Spherical");
                    });

                egui::Grid::new("ray_grid").num_columns(2).show(ui, |ui| {
                    match self.ray_mode {
                        RayMode::Cartesian => {
                            ui.label("origin");
                            ui.add(input3_xyz(&mut self.ray_origin_cartesian));
                            ui.end_row();
                        }
                        RayMode::Spherical => {
                            ui.label("origin");
                            ui.add(input3_spherical(&mut self.ray_origin_spherical));
                            ui.end_row();
                        }
                    }
                    ui.label("target");
                    ui.add(input3_xyz(&mut self.ray_target));
                    ui.end_row();
                })
            });

        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Trace").clicked() {
                let ray = match self.ray_mode {
                    RayMode::Cartesian => Ray::new(
                        self.ray_origin_cartesian.into(),
                        (self.ray_target - self.ray_origin_cartesian)
                            .normalize()
                            .into(),
                    ),
                    RayMode::Spherical => {
                        let r = self.ray_origin_spherical.x;
                        let theta = self.ray_origin_spherical.y.to_radians();
                        let phi = self.ray_origin_spherical.z.to_radians();
                        let origin = Vec3::new(
                            r * theta.sin() * phi.cos(),
                            r * theta.cos(),
                            r * theta.sin() * phi.sin(),
                        );
                        Ray::new(origin.into(), (self.ray_target - origin).normalize().into())
                    }
                };
                let event = VgonioEvent::TraceRayDbg {
                    ray,
                    max_bounces: self.params.emitter.max_bounces,
                    method: self.method,
                };
                if self.event_loop.send_event(event).is_err() {
                    log::warn!("Failed to send VgonioEvent::TraceRayDbg");
                }
            }

            if ui.button("Step").clicked() {
                println!("Stepping ray");
            }

            if ui.button("Reset").clicked() {
                println!("Resetting ray");
            }
        })
        .response
    }
}

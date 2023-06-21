use crate::{
    app::{
        cache::{Cache, Handle, MicroSurfaceRecord},
        gui::{
            misc::{input3_spherical, input3_xyz},
            widgets::{SurfaceSelector, ToggleSwitch},
            DebuggingEvent, VgonioEvent, VgonioEventLoop,
        },
    },
    math,
    measure::{
        emitter::RegionShape, measurement::BsdfMeasurementParams, rtc::Ray, CollectorScheme,
        RtcMethod,
    },
    msurf::MicroSurface,
    Handedness,
};
use egui_toast::ToastKind;
use glam::{IVec2, Vec3};
use std::sync::{Arc, RwLock};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RayMode {
    Cartesian,
    Spherical,
}

pub(crate) struct BrdfMeasurementDebugging {
    cache: Arc<RwLock<Cache>>,
    /// The index of the emitter position in the list of emitter measurement
    /// points. See [Emitter::measurement_points].
    emitter_position_index: i32,
    emitter_points_drawing: bool,
    emitter_rays_drawing: bool,
    emitter_samples_drawing: bool,
    params: BsdfMeasurementParams,
    selector: SurfaceSelector,
    ray_params_t: f32,
    orbit_radius: f32,
    collector_dome_drawing: bool,
    event_loop: VgonioEventLoop,
    method: RtcMethod,

    ray_origin_cartesian: Vec3,
    ray_origin_spherical: Vec3,
    ray_target: Vec3,
    ray_mode: RayMode,
    prim_id: u32,
    cell_pos: IVec2,
}

impl BrdfMeasurementDebugging {
    pub fn new(event_loop: VgonioEventLoop, cache: Arc<RwLock<Cache>>) -> Self {
        Self {
            collector_dome_drawing: false,
            cache,
            ray_origin_cartesian: Vec3::new(0.0, 5.0, 0.0),
            ray_origin_spherical: Vec3::new(5.0, 0.0, 0.0),
            ray_target: Default::default(),
            ray_mode: RayMode::Cartesian,
            method: RtcMethod::Grid,
            prim_id: 0,
            cell_pos: Default::default(),
            ray_params_t: 1.0,
            emitter_position_index: 0,
            emitter_points_drawing: false,
            emitter_rays_drawing: false,
            emitter_samples_drawing: false,
            params: BsdfMeasurementParams::default(),
            event_loop,
            orbit_radius: 1.0,
            selector: SurfaceSelector::single(),
        }
    }

    pub fn update_surface_selector(&mut self, surfaces: &[Handle<MicroSurface>]) {
        self.selector.update(surfaces, &self.cache.read().unwrap());
    }

    pub fn selected_surface(&self) -> Option<Handle<MicroSurface>> {
        self.selector.single_selected()
    }

    fn estimate_emitter_radii(
        &self,
        record: Option<&MicroSurfaceRecord>,
    ) -> Option<(f32, Option<f32>)> {
        match record {
            None => {
                self.event_loop
                    .send_event(VgonioEvent::Notify {
                        kind: ToastKind::Warning,
                        text: "No surface selected to evaluate the radius".to_string(),
                        time: 1.0,
                    })
                    .unwrap();
                None
            }
            Some(record) => {
                let (radius, disk_radius) = match self.params.emitter.shape {
                    RegionShape::SphericalCap { .. } | RegionShape::SphericalRect { .. } => {
                        let cache = self.cache.read().unwrap();
                        let mesh = cache
                            .get_micro_surface_mesh_by_surface_id(record.surf)
                            .unwrap();
                        (self.params.emitter.radius.estimate(mesh), None)
                    }
                    RegionShape::Disk { radius } => {
                        let cache = self.cache.read().unwrap();
                        let mesh = cache
                            .get_micro_surface_mesh_by_surface_id(record.surf)
                            .unwrap();
                        (
                            self.params.emitter.radius.estimate(mesh),
                            Some(radius.estimate_disk_radius(mesh)),
                        )
                    }
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

    fn estimate_collector_radii(
        &self,
        record: Option<&MicroSurfaceRecord>,
    ) -> Option<(f32, Option<f32>)> {
        match record {
            None => {
                self.event_loop
                    .send_event(VgonioEvent::Notify {
                        kind: ToastKind::Warning,
                        text: "No surface selected to evaluate the radius".to_string(),
                        time: 1.0,
                    })
                    .unwrap();
                None
            }
            Some(record) => {
                let cache = self.cache.read().unwrap();
                let mesh = cache
                    .get_micro_surface_mesh_by_surface_id(record.surf)
                    .unwrap();
                let orbit_radius = self.params.collector.radius.estimate(mesh);

                let (radius, disk_radius) = match self.params.collector.scheme {
                    CollectorScheme::Partitioned { .. } => (orbit_radius, None),
                    CollectorScheme::SingleRegion { shape, .. } => match shape {
                        RegionShape::SphericalCap { .. } | RegionShape::SphericalRect { .. } => {
                            (orbit_radius, None)
                        }
                        RegionShape::Disk { radius } => {
                            (orbit_radius, Some(radius.estimate_disk_radius(mesh)))
                        }
                    },
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

    fn update_emitter_position_index(&mut self, delta: i32, record: Option<&MicroSurfaceRecord>) {
        if let Some((orbit_radius, shape_radius)) = self.estimate_emitter_radii(record) {
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
            self.selector
                .ui("brdf_measurement_debugging_surface_selector", ui);
        });

        // Get a copy of the selected surface record to avoid locking the cache for the
        // whole function.
        let record = self.selector.single_selected().and_then(|s| {
            self.cache
                .read()
                .unwrap()
                .get_micro_surface_record(s)
                .map(|r| r.clone())
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
                                match &record {
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
                                        let radius = self.params.emitter.radius.estimate(
                                            self.cache
                                                .read()
                                                .unwrap()
                                                .get_micro_surface_mesh_by_surface_id(record.surf)
                                                .unwrap(),
                                        );
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
                                if let Some((orbit_radius, shape_radius)) =
                                    self.estimate_emitter_radii(record.as_ref())
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
                                self.update_emitter_position_index(-1, record.as_ref());
                            }
                            if ui.button("\u{25B6}").clicked() {
                                self.update_emitter_position_index(1, record.as_ref());
                            }
                            if ui
                                .add(ToggleSwitch::new(&mut self.emitter_samples_drawing))
                                .changed()
                            {
                                self.event_loop
                                    .send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::ToggleEmitterSamplesDrawing(
                                            self.emitter_samples_drawing,
                                        ),
                                    ))
                                    .unwrap();
                            }
                        });
                        ui.end_row();

                        ui.label("Sample rays:");
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .add(ToggleSwitch::new(&mut self.emitter_rays_drawing))
                                .changed()
                            {
                                if let Some((orbit_radius, shape_radius)) =
                                    self.estimate_emitter_radii(record.as_ref())
                                {
                                    self.orbit_radius = orbit_radius;

                                    self.event_loop
                                        .send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::ToggleEmitterRaysDrawing(
                                                self.emitter_rays_drawing,
                                            ),
                                        ))
                                        .unwrap();

                                    if self.emitter_rays_drawing {
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
                            }

                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut self.ray_params_t,
                                        1.0..=self.orbit_radius * 2.0,
                                    )
                                    .text("t"),
                                )
                                .changed()
                            {
                                if let Some((orbit_radius, shape_radius)) =
                                    self.estimate_emitter_radii(record.as_ref())
                                {
                                    self.event_loop
                                        .send_event(VgonioEvent::Debugging(
                                            DebuggingEvent::UpdateRayParams {
                                                t: self.ray_params_t,
                                                orbit_radius,
                                                shape_radius,
                                            },
                                        ))
                                        .unwrap();
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Measurement Points: ");
                        if ui
                            .add(ToggleSwitch::new(&mut self.emitter_points_drawing))
                            .changed()
                        {
                            self.event_loop
                                .send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleEmitterPointsDrawing(
                                        self.emitter_points_drawing,
                                    ),
                                ))
                                .unwrap();

                            if self.emitter_points_drawing {
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
                                if let Some((orbit_radius, _)) =
                                    self.estimate_emitter_radii(record.as_ref())
                                {
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
                        }
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("Collector")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Visibility:");
                    let toggle_res = ui.add(ToggleSwitch::new(&mut self.collector_dome_drawing));
                    let button_res = ui.button("Update");

                    if toggle_res.changed() || button_res.clicked() {
                        if let Some((orbit_radius, shape_radius)) =
                            self.estimate_collector_radii(record.as_ref())
                        {
                            self.event_loop
                                .send_event(VgonioEvent::Debugging(
                                    DebuggingEvent::ToggleCollectorDrawing {
                                        status: self.collector_dome_drawing,
                                        scheme: self.params.collector.scheme,
                                        orbit_radius,
                                        shape_radius,
                                    },
                                ))
                                .unwrap();
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Distance:")
                        .on_hover_text("Distance from collector to the surface.");
                    self.params.collector.radius.ui(ui);
                });
                self.params.collector.scheme.ui(ui);
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

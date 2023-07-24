use crate::{
    app::{
        cache::{Cache, Handle, MicroSurfaceRecord},
        gui::{
            event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
            notify::NotifyKind,
            widgets::{SurfaceSelector, ToggleSwitch},
        },
    },
    measure::{
        emitter::RegionShape, measurement::BsdfMeasurementParams, CollectorScheme, RtcMethod,
    },
};
use std::sync::{Arc, RwLock};
use vgcore::{
    math,
    math::{Handedness, IVec2},
    units::Radians,
};
use vgsurf::MicroSurface;

pub(crate) struct BrdfMeasurementDebugging {
    cache: Arc<RwLock<Cache>>,
    /// The index of the emitter position in the list of emitter measurement
    /// points. See [Emitter::measurement_points].
    emitter_position_index: i32,
    emitter_points_drawing: bool,
    emitter_rays_drawing: bool,
    emitter_samples_drawing: bool,
    ray_trajectories_drawing_reflected: bool,
    ray_trajectories_drawing_missed: bool,
    ray_hit_points_drawing: bool,
    params: BsdfMeasurementParams,
    selector: SurfaceSelector,
    ray_params_t: f32,
    orbit_radius: f32,
    collector_dome_drawing: bool,
    event_loop: EventLoopProxy,
    method: RtcMethod,
    surface_primitive_id: u32,
    surface_primitive_drawing: bool,
    grid_cell_position: IVec2,
    grid_cell_drawing: bool,
}

impl BrdfMeasurementDebugging {
    pub fn new(event_loop: EventLoopProxy, cache: Arc<RwLock<Cache>>) -> Self {
        Self {
            collector_dome_drawing: false,
            cache,
            #[cfg(feature = "embree")]
            method: RtcMethod::Embree,
            #[cfg(not(feature = "embree"))]
            method: RtcMethod::Grid,
            surface_primitive_id: 0,
            surface_primitive_drawing: false,
            grid_cell_position: Default::default(),
            ray_params_t: 1.0,
            emitter_position_index: 0,
            emitter_points_drawing: false,
            emitter_rays_drawing: false,
            emitter_samples_drawing: false,
            ray_trajectories_drawing_reflected: false,
            ray_trajectories_drawing_missed: false,
            ray_hit_points_drawing: false,
            params: BsdfMeasurementParams::default(),
            event_loop,
            orbit_radius: 1.0,
            selector: SurfaceSelector::single(),
            grid_cell_drawing: false,
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
                        kind: NotifyKind::Warning,
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
                        kind: NotifyKind::Warning,
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

    fn calc_emitter_position(&self) -> (Radians, Radians) {
        let zenith_step_count = self.params.emitter.zenith.step_count_wrapped();
        let azimuth_idx = self.emitter_position_index / zenith_step_count as i32;
        let zenith_idx = self.emitter_position_index % zenith_step_count as i32;
        let zenith = self.params.emitter.zenith.step(zenith_idx as usize);
        let azimuth = self.params.emitter.azimuth.step(azimuth_idx as usize);
        log::trace!(
            "[BrdfMeasurementDebugging] calc emitter position index: {}, zenith: {}, index={}, \
             azimuth: {}, index={}",
            self.emitter_position_index,
            zenith.to_degrees(),
            zenith_idx,
            azimuth.to_degrees(),
            azimuth_idx
        );
        (zenith, azimuth)
    }

    fn update_emitter_position_index(&mut self, delta: i32, record: Option<&MicroSurfaceRecord>) {
        if let Some((orbit_radius, shape_radius)) = self.estimate_emitter_radii(record) {
            self.emitter_position_index = ((self.emitter_position_index + delta).max(0) as usize
                % (self.params.emitter.zenith.step_count_wrapped()
                    * self.params.emitter.azimuth.step_count_wrapped()))
                as i32;
            self.orbit_radius = orbit_radius;
            let (zenith, azimuth) = self.calc_emitter_position();
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
        let mut record = None;
        ui.horizontal_wrapped(|ui| {
            ui.label("Tracing Method: ");
            #[cfg(feature = "embree")]
            ui.selectable_value(&mut self.method, RtcMethod::Embree, "Embree");
            #[cfg(feature = "optix")]
            ui.selectable_value(&mut self.method, RtcMethod::Optix, "OptiX");
            ui.selectable_value(&mut self.method, RtcMethod::Grid, "Grid");
        });

        if self.method == RtcMethod::Grid {
            ui.horizontal_wrapped(|ui| {
                ui.label("Cell Position: ");
                ui.add(egui::DragValue::new(&mut self.grid_cell_position.x).prefix("x: "));
                ui.add(egui::DragValue::new(&mut self.grid_cell_position.y).prefix("y: "));

                if ui
                    .add(ToggleSwitch::new(&mut self.grid_cell_drawing))
                    .changed()
                {
                    self.event_loop
                        .send_event(VgonioEvent::Debugging(
                            DebuggingEvent::UpdateGridCellDrawing {
                                pos: self.grid_cell_position,
                                status: self.grid_cell_drawing,
                            },
                        ))
                        .unwrap();
                }
            });
        }

        egui::CollapsingHeader::new("Specimen")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("MicroSurface: ");
                    self.selector
                        .ui("brdf_measurement_debugging_surface_selector", ui);
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Primitive ID: ");
                    ui.add(egui::DragValue::new(&mut self.surface_primitive_id));

                    // Get a copy of the selected surface record to avoid locking the cache for the
                    // whole function.
                    record = self.selector.single_selected().and_then(|s| {
                        self.cache
                            .read()
                            .unwrap()
                            .get_micro_surface_record(s)
                            .cloned()
                    });

                    let renderable = record.as_ref().map(|r| r.renderable);

                    if ui
                        .add(ToggleSwitch::new(&mut self.surface_primitive_drawing))
                        .changed()
                    {
                        self.event_loop
                            .send_event(VgonioEvent::Debugging(
                                DebuggingEvent::UpdateSurfacePrimitiveId {
                                    mesh: renderable,
                                    id: self.surface_primitive_id,
                                    status: self.surface_primitive_drawing,
                                },
                            ))
                            .unwrap()
                    }

                    if ui.button("\u{25C0}").clicked() {
                        self.surface_primitive_id = (self.surface_primitive_id - 1).max(0);
                        self.event_loop
                            .send_event(VgonioEvent::Debugging(
                                DebuggingEvent::UpdateSurfacePrimitiveId {
                                    mesh: renderable,
                                    id: self.surface_primitive_id,
                                    status: self.surface_primitive_drawing,
                                },
                            ))
                            .unwrap()
                    }

                    if ui.button("\u{25B6}").clicked() {
                        self.surface_primitive_id = (self.surface_primitive_id + 1).min(u32::MAX);
                        self.event_loop
                            .send_event(VgonioEvent::Debugging(
                                DebuggingEvent::UpdateSurfacePrimitiveId {
                                    mesh: renderable,
                                    id: self.surface_primitive_id,
                                    status: self.surface_primitive_drawing,
                                },
                            ))
                            .unwrap()
                    }
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
                                match &record {
                                    None => {
                                        self.event_loop
                                            .send_event(VgonioEvent::Notify {
                                                kind: NotifyKind::Warning,
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
                                                kind: NotifyKind::Warning,
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
                                        patches: self.params.collector.generate_patches(),
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
        ui.separator();

        if ui.button("Simulate").clicked() {
            if let Some(record) = record {
                self.event_loop
                    .send_event(VgonioEvent::Debugging(DebuggingEvent::MeasureOnePoint {
                        method: self.method,
                        params: self.params,
                        mesh: record.mesh,
                    }))
                    .unwrap();
            }
        }
        ui.separator();

        ui.horizontal_wrapped(|ui| {
            ui.label("Ray trajectories: ");
            ui.label("reflected");
            let changed0 = ui
                .add(ToggleSwitch::new(
                    &mut self.ray_trajectories_drawing_reflected,
                ))
                .changed();

            ui.label("missed");
            let changed1 = ui
                .add(ToggleSwitch::new(&mut self.ray_trajectories_drawing_missed))
                .changed();

            if changed0 || changed1 {
                self.event_loop
                    .send_event(VgonioEvent::Debugging(
                        DebuggingEvent::ToggleRayTrajectoriesDrawing {
                            reflected: self.ray_trajectories_drawing_reflected,
                            missed: self.ray_trajectories_drawing_missed,
                        },
                    ))
                    .unwrap();
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Collected rays: ");
            if ui
                .add(ToggleSwitch::new(&mut self.ray_hit_points_drawing))
                .changed()
            {
                self.event_loop
                    .send_event(VgonioEvent::Debugging(
                        DebuggingEvent::ToggleCollectedRaysDrawing(self.ray_hit_points_drawing),
                    ))
                    .unwrap();
            }
        })
        .response
    }
}

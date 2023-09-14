use crate::{
    app::{
        cache::{Handle, InnerCache, MicroSurfaceRecord},
        gui::{
            event::{DebuggingEvent, EventLoopProxy, VgonioEvent},
            misc::drag_angle,
            notify::NotifyKind,
            widgets::{SurfaceSelector, ToggleSwitch},
        },
    },
    measure,
    measure::{
        bsdf::{
            detector::{Detector, DetectorParams, DetectorScheme},
            rtc::RtcMethod,
        },
        params::BsdfMeasurementParams,
    },
    SphericalDomain,
};
use egui::Widget;
use std::sync::{Arc, RwLock};
use uuid::Uuid;
use vgcore::math::{IVec2, Sph2};
use vgsurf::MicroSurface;

impl DetectorParams {
    pub fn ui<R>(&mut self, ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui) -> R) {
        egui::CollapsingHeader::new("Detector")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("detector_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Precision:");
                        drag_angle(&mut self.precision, "").ui(ui);
                        ui.end_row();

                        ui.label("Domain:");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(&mut self.domain, SphericalDomain::Upper, "Upper");
                            ui.selectable_value(&mut self.domain, SphericalDomain::Lower, "Lower");
                            ui.selectable_value(&mut self.domain, SphericalDomain::Whole, "Whole");
                        });
                        ui.end_row();

                        ui.label("Scheme");
                        ui.horizontal_wrapped(|ui| {
                            ui.selectable_value(
                                &mut self.scheme,
                                DetectorScheme::Beckers,
                                "Beckers",
                            );
                            ui.selectable_value(
                                &mut self.scheme,
                                DetectorScheme::Tregenza,
                                "Tregenza",
                            );
                        });
                        ui.end_row();

                        add_contents(ui);
                    });
            });
    }
}

pub(crate) struct BrdfMeasurementDebugging {
    cache: Arc<RwLock<InnerCache>>,
    /// The index of the emitter position in the list of emitter measurement
    /// points. See [EmitterParams::measurement_points].
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
    collector_dome_drawing: bool,
    event_loop: EventLoopProxy,
    method: RtcMethod,
    surface_primitive_id: u32,
    surface_primitive_drawing: bool,
    surface_normals_drawing: bool,
    grid_cell_position: IVec2,
    grid_cell_drawing: bool,
    surface_viewers: Vec<Uuid>,
    focused_viewer: Option<Uuid>,
}

impl BrdfMeasurementDebugging {
    pub fn new(event_loop: EventLoopProxy, cache: Arc<RwLock<InnerCache>>) -> Self {
        Self {
            collector_dome_drawing: false,
            cache,
            #[cfg(feature = "embree")]
            method: RtcMethod::Embree,
            #[cfg(not(feature = "embree"))]
            method: RtcMethod::Grid,
            surface_primitive_id: 0,
            surface_primitive_drawing: false,
            surface_normals_drawing: false,
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
            selector: SurfaceSelector::single(),
            grid_cell_drawing: false,
            surface_viewers: vec![],
            focused_viewer: None,
        }
    }

    pub fn update_surface_selector(&mut self, surfaces: &[Handle<MicroSurface>]) {
        self.selector.update(surfaces, &self.cache.read().unwrap());
    }

    pub fn update_surface_viewers(&mut self, viewers: &[Uuid]) {
        for viewer in viewers {
            if !self.surface_viewers.contains(viewer) {
                self.surface_viewers.push(*viewer);
            }
        }
    }

    pub fn selected_surface(&self) -> Option<Handle<MicroSurface>> {
        self.selector.single_selected()
    }

    pub fn selected_viewer(&self) -> Option<Uuid> { self.focused_viewer }

    fn calc_emitter_position(&self) -> Sph2 {
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
        Sph2::new(zenith, azimuth)
    }

    fn update_emitter_position_index(&mut self, delta: i32, record: Option<&MicroSurfaceRecord>) {
        if record.is_none() {
            return;
        }
        self.emitter_position_index = ((self.emitter_position_index + delta).max(0) as usize
            % (self.params.emitter.zenith.step_count_wrapped()
                * self.params.emitter.azimuth.step_count_wrapped()))
            as i32;
        self.event_loop
            .send_event(VgonioEvent::Debugging(
                DebuggingEvent::UpdateEmitterPosition {
                    position: self.calc_emitter_position(),
                },
            ))
            .unwrap();
    }
}

impl egui::Widget for &mut BrdfMeasurementDebugging {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        egui::ComboBox::new("brdf_measurement_debugging_selector", "Surface Viewer")
            .selected_text(match self.focused_viewer {
                None => "Select a surface viewer".into(),
                Some(uuid) => format!("Viewer-{}", &uuid.to_string()[..6].to_ascii_uppercase()),
            })
            .show_ui(ui, |ui| {
                for viewer in &self.surface_viewers {
                    ui.selectable_value(
                        &mut self.focused_viewer,
                        Some(*viewer),
                        format!("Viewer-{}", &viewer.to_string()[..6].to_ascii_uppercase()),
                    );
                }
            });
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
                        .ui("brdf_measurement_debugging_surface_selector", ui)
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

                    if self.selector.selection_changed() && record.is_some() {
                        self.event_loop
                            .send_event(VgonioEvent::Debugging(
                                DebuggingEvent::UpdateMicroSurface {
                                    surf: record.as_ref().unwrap().surf,
                                    mesh: record.as_ref().unwrap().mesh,
                                },
                            ))
                            .unwrap();
                    }

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
                ui.horizontal_wrapped(|ui| {
                    ui.label("Surface Normals: ");
                    if ui
                        .add(ToggleSwitch::new(&mut self.surface_normals_drawing))
                        .changed()
                    {
                        self.event_loop
                            .send_event(VgonioEvent::Debugging(
                                DebuggingEvent::ToggleSurfaceNormalDrawing,
                            ))
                            .unwrap();
                    }
                })
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

                        ui.label("Azimuthal range φ: ");
                        self.params.emitter.azimuth.ui(ui);
                        ui.end_row();

                        ui.label("Zenith range θ: ");
                        self.params.emitter.zenith.ui(ui);
                        ui.end_row();

                        ui.label("Wavelength range: ");
                        self.params.emitter.spectrum.ui(ui);
                        ui.end_row();

                        ui.label("Samples: ");
                        ui.horizontal_wrapped(|ui| {
                            if ui.button("Generate").clicked() {
                                self.event_loop
                                    .send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::UpdateEmitterSamples {
                                            samples: self.params.emitter.generate_unit_samples(),
                                        },
                                    ))
                                    .unwrap();
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
                                && record.is_some()
                            {
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
                                            DebuggingEvent::EmitRays,
                                        ))
                                        .unwrap();
                                }
                            }

                            let max_t = if record.is_some() {
                                let cache = self.cache.read().unwrap();
                                let mesh = cache
                                    .get_micro_surface_mesh(record.as_ref().unwrap().mesh)
                                    .unwrap();
                                measure::estimate_orbit_radius(mesh) * 2.0
                            } else {
                                f32::MAX / 4.0
                            };

                            if ui
                                .add(
                                    egui::Slider::new(&mut self.ray_params_t, 1.0..=max_t)
                                        .text("t"),
                                )
                                .changed()
                                && record.is_some()
                            {
                                self.event_loop
                                    .send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::UpdateRayParams {
                                            t: self.ray_params_t,
                                        },
                                    ))
                                    .unwrap();
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

                            if self.emitter_points_drawing && record.is_some() {
                                let points = self.params.emitter.generate_measurement_points();
                                self.event_loop
                                    .send_event(VgonioEvent::Debugging(
                                        DebuggingEvent::UpdateEmitterPoints { points },
                                    ))
                                    .unwrap();
                            }
                        }
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("Collector")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Dome:");
                    ui.add(ToggleSwitch::new(&mut self.collector_dome_drawing));
                });
                self.params.detector.ui(ui, |_| {});
                if ui.button("Update").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::Debugging(
                            DebuggingEvent::UpdateCollectorDrawing {
                                status: self.collector_dome_drawing,
                                patches: self.params.detector.generate_patches(),
                            },
                        ))
                        .unwrap();
                }
            });
        ui.separator();

        if ui.button("Simulate").clicked() {
            if let Some(record) = record.as_ref() {
                self.event_loop
                    .send_event(VgonioEvent::Debugging(DebuggingEvent::MeasureOnce {
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

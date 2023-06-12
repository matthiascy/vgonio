use crate::{
    app::{
        cache::{Cache, Handle, MicroSurfaceRecord},
        gui::{
            misc::{input3_spherical, input3_xyz},
            widgets::ToggleSwitch,
            VgonioEvent, VgonioEventLoop,
        },
    },
    measure::{measurement::Radius, rtc::Ray, RtcMethod},
    msurf::MicroSurface,
    units::{mm, UMillimetre},
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

pub(crate) struct BrdfMeasurementPane {
    pub dome_radius: f32,
    pub show_dome: bool,
    cache: Arc<RwLock<Cache>>,

    ray_origin_cartesian: Vec3,
    ray_origin_spherical: Vec3,
    ray_target: Vec3,
    ray_mode: RayMode,
    max_bounces: u32,
    method: RtcMethod,
    prim_id: u32,
    cell_pos: IVec2,
    t: f32,

    pub loaded_surfaces: Vec<MicroSurfaceRecord>,
    pub selected_surface: Option<Handle<MicroSurface>>,
    event_loop: VgonioEventLoop,
}

impl BrdfMeasurementPane {
    pub fn new(event_loop: VgonioEventLoop, cache: Arc<RwLock<Cache>>) -> Self {
        Self {
            dome_radius: 1.0,
            show_dome: false,
            cache,
            ray_origin_cartesian: Vec3::new(0.0, 5.0, 0.0),
            ray_origin_spherical: Vec3::new(5.0, 0.0, 0.0),
            ray_target: Default::default(),
            ray_mode: RayMode::Cartesian,
            max_bounces: 20,
            method: RtcMethod::Grid,
            prim_id: 0,
            cell_pos: Default::default(),
            t: 10.0,
            loaded_surfaces: vec![],
            selected_surface: None,
            event_loop,
        }
    }
}

impl egui::Widget for &mut BrdfMeasurementPane {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.label("MicroSurface");
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
        ui.horizontal_wrapped(|ui| {
            ui.label("Dome Radius:");
            ui.add(
                egui::DragValue::new(&mut self.dome_radius)
                    .speed(0.1)
                    .clamp_range(0.0..=10000.0),
            );
            if ui.button("eval").clicked() {
                let record = self
                    .loaded_surfaces
                    .iter()
                    .find(|s| s.surf == self.selected_surface.unwrap_or(Handle::default()));
                match record {
                    None => {
                        self.event_loop
                            .send_event(VgonioEvent::Notify {
                                kind: ToastKind::Warning,
                                text: "No surface selected to evaluate the dome radius".to_string(),
                                time: 3.0,
                            })
                            .unwrap();
                    }
                    Some(record) => {
                        let cache = self.cache.read().unwrap();
                        let mesh = cache.get_micro_surface_mesh(record.mesh).unwrap();
                        self.dome_radius = Radius::Auto(mm!(
                            self.dome_radius * mesh.unit.factor_convert_to::<UMillimetre>()
                        ))
                        .eval(mesh);
                    }
                }
            }
            ui.add(ToggleSwitch::new(&mut self.show_dome));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Emitter:");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Method");
            #[cfg(feature = "embree")]
            ui.selectable_value(&mut self.method, RtcMethod::Embree, "Embree");
            #[cfg(feature = "optix")]
            ui.selectable_value(&mut self.method, RtcMethod::Optix, "OptiX");
            ui.selectable_value(&mut self.method, RtcMethod::Grid, "Grid");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Max bounces");
            ui.add(
                egui::DragValue::new(&mut self.max_bounces)
                    .clamp_range(1..=32)
                    .speed(1),
            );
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("t");
            let res = ui.add(egui::DragValue::new(&mut self.t).clamp_range(0.1..=200.0));
            if res.changed()
                && self
                    .event_loop
                    .send_event(VgonioEvent::UpdateDebugT(self.t))
                    .is_err()
            {
                log::warn!("Failed to send event VgonioEvent::UpdateDebugT");
            }
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
                    max_bounces: self.max_bounces,
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

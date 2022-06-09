use crate::acq::ray::Ray;
use crate::acq::tracing::RayTracingMethod;
use crate::app::gui::widgets::{input3_spherical, input3_xyz};
use crate::app::gui::VgonioEvent;
use glam::{IVec2, Vec3};
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RayMode {
    OriginDirection,
    OriginTarget,
}

pub(crate) struct RayTracingPane {
    ray: Ray,
    ray_origin_spherical: Vec3,
    ray_target: Vec3,
    ray_mode: RayMode,
    max_bounces: u32,
    method: RayTracingMethod,
    prim_id: u32,
    cell_pos: IVec2,
    t: f32,
    event_loop: Arc<EventLoopProxy<VgonioEvent>>,
}

impl RayTracingPane {
    pub fn new(event_loop: Arc<EventLoopProxy<VgonioEvent>>) -> Self {
        Self {
            ray: Ray::new(Vec3::new(0.0, 100.0, 0.0), Vec3::new(0.0, -1.0, 0.0)),
            ray_origin_spherical: Default::default(),
            ray_target: Default::default(),
            ray_mode: RayMode::OriginDirection,
            max_bounces: 20,
            method: RayTracingMethod::Standard,
            prim_id: 0,
            cell_pos: Default::default(),
            t: 10.0,
            event_loop,
        }
    }
}

impl egui::Widget for &mut RayTracingPane {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal_wrapped(|ui| {
            ui.label("method");
            ui.selectable_value(&mut self.method, RayTracingMethod::Standard, "Standard");
            ui.selectable_value(&mut self.method, RayTracingMethod::Grid, "Grid");
            ui.selectable_value(&mut self.method, RayTracingMethod::Hybrid, "Hybrid");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("max_bounces");
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

        if self.method == RayTracingMethod::Grid {
            ui.horizontal_wrapped(|ui| {
                ui.label("cell");
                ui.add(egui::DragValue::new(&mut self.cell_pos.x).prefix("x: "));
                ui.add(egui::DragValue::new(&mut self.cell_pos.y).prefix("y:"));

                if ui.button("show").clicked() && self
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
                        ui.selectable_value(
                            &mut self.ray_mode,
                            RayMode::OriginDirection,
                            "Cartesian",
                        );
                        ui.selectable_value(&mut self.ray_mode, RayMode::OriginTarget, "Spherical");
                    });

                egui::Grid::new("ray_grid").num_columns(2).show(ui, |ui| {
                    match self.ray_mode {
                        RayMode::OriginDirection => {
                            ui.label("origin");
                            ui.add(input3_xyz(&mut self.ray.o));
                            ui.end_row();

                            ui.label("direction");
                            ui.add(input3_xyz(&mut self.ray.d));
                            ui.end_row();
                        }
                        RayMode::OriginTarget => {
                            ui.label("origin");
                            ui.add(input3_spherical(&mut self.ray_origin_spherical));
                            ui.end_row();

                            ui.label("target");
                            ui.add(input3_xyz(&mut self.ray_target));
                            ui.end_row();
                        }
                    }
                    ui.label("energy");
                    ui.add(egui::DragValue::new(&mut self.ray.e).clamp_range(0.0..=1.0));
                    ui.end_row();
                })
            });

        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Trace").clicked() {
                let ray = match self.ray_mode {
                    RayMode::OriginDirection => Ray {
                        o: self.ray.o,
                        d: self.ray.d.normalize(),
                        e: self.ray.e,
                    },
                    RayMode::OriginTarget => {
                        let r = self.ray_origin_spherical.x;
                        let theta = self.ray_origin_spherical.y.to_radians();
                        let phi = self.ray_origin_spherical.z.to_radians();
                        let origin = Vec3::new(
                            r * theta.sin() * phi.cos(),
                            r * theta.cos(),
                            r * theta.sin() * phi.sin(),
                        );
                        let d = (self.ray_target - origin).normalize();
                        Ray {
                            o: origin,
                            d,
                            e: self.ray.e,
                        }
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

use crate::{
    acq::{Ray, RtcMethod},
    app::gui::{
        widgets::{input3_spherical, input3_xyz},
        VgonioEvent,
    },
};
use glam::{IVec2, Vec3};
use winit::event_loop::EventLoopProxy;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum RayMode {
    Cartesian,
    Spherical,
}

pub(crate) struct RayTracingPane {
    ray_origin_cartesian: Vec3,
    ray_origin_spherical: Vec3,
    ray_target: Vec3,
    ray_mode: RayMode,
    max_bounces: u32,
    method: RtcMethod,
    prim_id: u32,
    cell_pos: IVec2,
    t: f32,
    event_loop: EventLoopProxy<VgonioEvent>,
}

impl RayTracingPane {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>) -> Self {
        Self {
            ray_origin_cartesian: Vec3::new(0.0, 5.0, 0.0),
            ray_origin_spherical: Vec3::new(5.0, 0.0, 0.0),
            ray_target: Default::default(),
            ray_mode: RayMode::Cartesian,
            max_bounces: 20,
            method: RtcMethod::Grid,
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
            #[cfg(feature = "embree")]
            ui.selectable_value(&mut self.method, RtcMethod::Standard, "Standard");
            ui.selectable_value(&mut self.method, RtcMethod::Grid, "Grid");
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
                    RayMode::Cartesian => Ray {
                        o: self.ray_origin_cartesian,
                        d: (self.ray_target - self.ray_origin_cartesian).normalize(),
                    },
                    RayMode::Spherical => {
                        let r = self.ray_origin_spherical.x;
                        let theta = self.ray_origin_spherical.y.to_radians();
                        let phi = self.ray_origin_spherical.z.to_radians();
                        let origin = Vec3::new(
                            r * theta.sin() * phi.cos(),
                            r * theta.cos(),
                            r * theta.sin() * phi.sin(),
                        );
                        Ray {
                            o: origin,
                            d: (self.ray_target - origin).normalize(),
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

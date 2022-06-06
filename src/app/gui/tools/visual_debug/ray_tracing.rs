use crate::acq::ray::Ray;
use crate::acq::tracing::RayTracingMethod;
use crate::app::gui::widgets::input3;
use crate::app::gui::VgonioEvent;
use glam::Vec3;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

pub(crate) struct RayTracingPane {
    ray: Ray,
    method: RayTracingMethod,
    event_loop: Arc<EventLoopProxy<VgonioEvent>>,
}

impl RayTracingPane {
    pub fn new(event_loop: Arc<EventLoopProxy<VgonioEvent>>) -> Self {
        Self {
            ray: Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            method: RayTracingMethod::Standard,
            event_loop,
        }
    }
}

impl egui::Widget for &mut RayTracingPane {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        egui::CollapsingHeader::new("ray")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("ray_grid").num_columns(2).show(ui, |ui| {
                    ui.label("origin");
                    ui.add(input3(&mut self.ray.o));
                    ui.end_row();

                    ui.label("direction");
                    ui.add(input3(&mut self.ray.d));
                    ui.end_row();

                    ui.label("energy");
                    ui.add(egui::DragValue::new(&mut self.ray.e).clamp_range(0.0..=1.0));
                    ui.end_row();
                })
            });

        ui.horizontal_wrapped(|ui| {
            ui.label("method");
            ui.selectable_value(&mut self.method, RayTracingMethod::Standard, "Standard");
            ui.selectable_value(&mut self.method, RayTracingMethod::Grid, "Grid");
            ui.selectable_value(&mut self.method, RayTracingMethod::Hybrid, "Hybrid");
        });

        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Trace").clicked() {
                let event = VgonioEvent::TraceRayDbg {
                    ray: Ray {
                        o: self.ray.o,
                        d: self.ray.d.normalize(),
                        e: self.ray.e,
                    },
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

use crate::acq::ray::Ray;
use glam::Vec3;
use crate::app::gui::widgets::input3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RayTracingMethod {
    Standard,
    Grid,
    Hybrid,
}

pub(crate) struct RayTracingPane {
    ray: Ray,
    method: RayTracingMethod,
}

impl Default for RayTracingPane {
    fn default() -> Self {
        Self {
            ray: Ray {
                o: Vec3::ZERO,
                d: Vec3::ZERO,
                e: 1.0,
            },
            method: RayTracingMethod::Standard,
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
                println!("Tracing ray");
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

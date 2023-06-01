//! Source code based on the official example of custom widget.

mod input;

use egui::{emath::Numeric, DragValue};
pub use input::{input, input3_spherical, input3_ui, input3_xyz, input_ui};

use crate::{
    units::{Angle, AngleUnit},
    RangeByStepSizeExclusive, RangeByStepSizeInclusive,
};

impl<T: Copy + Numeric> RangeByStepSizeInclusive<T> {
    pub fn ui(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.start).prefix("start: "));
            ui.add(egui::DragValue::new(&mut self.stop).prefix("stop: "));
            ui.add(egui::DragValue::new(&mut self.step_size).prefix("step: "));
        })
        .response
    }
}

fn drag_angle<'a, A: AngleUnit>(angle: &'a mut Angle<A>, prefix: &str) -> DragValue<'a> {
    let display_factor = A::FACTOR_TO_DEG;
    let storage_factor = A::FACTOR_FROM_DEG;
    DragValue::new(&mut angle.value)
        .prefix(prefix)
        .custom_formatter(move |val, _| format!("{:.2}°", val as f32 * display_factor))
        .custom_parser(move |val_str| {
            let val = val_str.parse::<f64>().unwrap_or(0.0);
            Some((val * storage_factor as f64) % A::TAU as f64)
        })
        .speed(storage_factor as f64)
}

impl<A: AngleUnit> RangeByStepSizeInclusive<Angle<A>> {
    /// Creates the UI for the range.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.add(drag_angle(&mut self.start, "start: "));
            ui.add(drag_angle(&mut self.stop, "stop: "));
            ui.add(drag_angle(&mut self.step_size, "step: "));
        })
        .response
    }
}

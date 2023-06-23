use egui::{emath::Numeric, DragValue};

use crate::{
    units::{Angle, AngleUnit, Length, LengthMeasurement},
    RangeByStepCountInclusive, RangeByStepSizeInclusive,
};

impl<T: Copy + Numeric> RangeByStepSizeInclusive<T> {
    /// Creates the UI for the range.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.start).prefix("start: "));
            ui.add(egui::DragValue::new(&mut self.stop).prefix("stop: "));
            ui.add(egui::DragValue::new(&mut self.step_size).prefix("step: "));
        })
        .response
    }
}

pub fn drag_angle<'a, A: AngleUnit>(angle: &'a mut Angle<A>, prefix: &str) -> DragValue<'a> {
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

#[allow(dead_code)]
fn drag_length<'a, L: LengthMeasurement>(length: &'a mut Length<L>, prefix: &str) -> DragValue<'a> {
    DragValue::new(&mut length.value)
        .prefix(prefix)
        .custom_formatter(move |val, _| format!("{:.2}m", val))
        .custom_parser(move |val_str| {
            let val = val_str.parse::<f64>().unwrap_or(0.0);
            Some(val)
        })
        .speed(0.1)
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

impl<A: AngleUnit> RangeByStepCountInclusive<Angle<A>> {
    /// Creates the UI for the range.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.add(drag_angle(&mut self.start, "start: "));
            ui.add(drag_angle(&mut self.stop, "stop: "));
            ui.add(DragValue::new(&mut self.step_count).prefix("steps: "));
        })
        .response
    }
}

impl<L: LengthMeasurement> RangeByStepSizeInclusive<Length<L>> {
    /// Creates the UI for the range.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.start.value)
                    .prefix("start: ")
                    .suffix(L::SYMBOL),
            );
            ui.add(
                egui::DragValue::new(&mut self.stop.value)
                    .prefix("stop: ")
                    .suffix(L::SYMBOL),
            );
            ui.add(
                egui::DragValue::new(&mut self.step_size.value)
                    .prefix("step: ")
                    .suffix(L::SYMBOL),
            );
        })
        .response
    }
}

use std::ops::RangeInclusive;

#[allow(dead_code)]
pub fn input_ui<T: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    value: &mut T,
    prefix: &str,
    range: Option<RangeInclusive<T>>,
) -> egui::Response {
    ui.add(match range {
        Some(range) => egui::DragValue::new(value)
            .prefix(prefix)
            .clamp_range(range),
        None => egui::DragValue::new(value).prefix(prefix),
    })
}

/// A wrapper that allows the more idiomatic usage of `input_vec3`.
///
/// ## Example:
/// ``` ignore
/// ui.add(input3(&mut my_vec3));
/// ```
#[allow(dead_code)]
pub fn input<'a, T: egui::emath::Numeric>(
    value: &'a mut T,
    prefix: &'a str,
    range: Option<RangeInclusive<T>>,
) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| input_ui(ui, value, prefix, range)
}

use glam::Vec3;

#[allow(dead_code)]
pub fn input3_ui(ui: &mut egui::Ui, value: &mut Vec3, prefixes: &[&str; 3]) -> egui::Response {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut value.x).prefix(prefixes[0]));
        ui.add(egui::DragValue::new(&mut value.y).prefix(prefixes[1]));
        ui.add(egui::DragValue::new(&mut value.z).prefix(prefixes[2]));
    })
    .response
}

/// A wrapper that allows the more idiomatic usage of `input_vec3`.
///
/// ## Example:
/// ``` ignore
/// ui.add(input3(&mut my_vec3));
/// ```
#[allow(dead_code)]
pub fn input3_xyz(value: &mut Vec3) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| input3_ui(ui, value, &["x: ", "y: ", "z: "])
}

#[allow(dead_code)]
pub fn input3_spherical(value: &mut Vec3) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| {
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut value.x).prefix("r: "));
            ui.add(
                egui::DragValue::new(&mut value.y)
                    .prefix("θ: ")
                    .suffix("°")
                    .clamp_range(0.0..=90.0)
                    .speed(0.01),
            );
            ui.add(
                egui::DragValue::new(&mut value.z)
                    .prefix("φ: ")
                    .suffix("°")
                    .clamp_range(0.0..=360.0)
                    .speed(0.01),
            );
        })
        .response
    }
}

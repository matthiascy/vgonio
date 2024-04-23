use egui::DragValue;

use base::range::{RangeByStepCountInclusive, RangeByStepSizeInclusive};

pub fn drag_angle<'a, A: AngleUnit>(angle: &'a mut Angle<A>, prefix: &str) -> DragValue<'a> {
    DragValue::new(angle.value_mut())
        .prefix(prefix)
        .custom_formatter(move |val, _| format!("{:6.2}°", val as f32 * A::FACTOR_TO_DEG))
        .custom_parser(move |val_str| {
            if val_str == "pi" {
                return Some(A::PI as f64);
            } else if val_str == "tau" || val_str == "2pi" {
                return Some(A::TAU as f64);
            }
            let val = val_str.parse::<f64>().unwrap_or(0.0);
            Some((val * A::FACTOR_FROM_DEG as f64) % A::TAU as f64)
        })
        .speed(A::FACTOR_FROM_DEG as f64)
}

pub fn range_step_size_inclusive_angle_ui<A: AngleUnit>(
    range: &mut RangeByStepSizeInclusive<Angle<A>>,
    ui: &mut egui::Ui,
) -> egui::Response {
    ui.horizontal(|ui| {
        ui.add(drag_angle(&mut range.start, ""));
        ui.label("..=");
        let start_val = range.start.value() as f64;
        ui.add(
            DragValue::new(range.stop.value_mut())
                .custom_formatter(move |val, _| format!("{:6.2}°", val as f32 * A::FACTOR_TO_DEG))
                .custom_parser(move |val_str| {
                    if val_str == "pi" {
                        return Some(A::PI as f64);
                    } else if val_str == "tau" || val_str == "2pi" {
                        return Some(A::TAU as f64);
                    }
                    let val = val_str.parse::<f64>().unwrap_or(0.0);
                    let wrapped = (val * A::FACTOR_FROM_DEG as f64) % A::TAU as f64;
                    if approx::relative_eq!(wrapped, start_val, epsilon = 1e-6) {
                        Some(val * A::FACTOR_FROM_DEG as f64)
                    } else {
                        Some(wrapped)
                    }
                })
                .speed(A::FACTOR_FROM_DEG as f64),
        );
        ui.label("per");
        ui.add(drag_angle(&mut range.step_size, ""));
    })
    .response
}

pub fn range_step_count_inclusive_angle_ui<A: AngleUnit>(
    range: &mut RangeByStepCountInclusive<Angle<A>>,
    ui: &mut egui::Ui,
) -> egui::Response {
    // Creates the UI for the range.
    ui.horizontal(|ui| {
        ui.add(drag_angle(&mut range.start, "start: "));
        ui.add(drag_angle(&mut range.stop, "stop: "));
        ui.add(DragValue::new(&mut range.step_count).prefix("steps: "));
    })
    .response
}

pub fn range_step_size_inclusive_length_ui<L: LengthMeasurement>(
    range: &mut RangeByStepSizeInclusive<Length<L>>,
    ui: &mut egui::Ui,
) -> egui::Response {
    // Creates the UI for the range.
    ui.horizontal(|ui| {
        ui.add(DragValue::new(range.start.value_mut()).suffix(L::SYMBOL));
        ui.label("..=");
        ui.add(DragValue::new(range.stop.value_mut()).suffix(L::SYMBOL));
        ui.label("per");
        ui.add(DragValue::new(range.step_size.value_mut()).suffix(L::SYMBOL));
    })
    .response
}

use base::{
    math::Vec3,
    units::{Angle, AngleUnit, Length, LengthMeasurement},
};
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
            ui.add(DragValue::new(&mut value.x).prefix("r: "));
            ui.add(
                DragValue::new(&mut value.y)
                    .prefix("θ: ")
                    .suffix("°")
                    .clamp_range(0.0..=90.0)
                    .speed(0.01),
            );
            ui.add(
                DragValue::new(&mut value.z)
                    .prefix("φ: ")
                    .suffix("°")
                    .clamp_range(0.0..=360.0)
                    .speed(0.01),
            );
        })
        .response
    }
}

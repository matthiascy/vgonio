use std::ops::RangeInclusive;

pub fn input_ui<T: egui::emath::Numeric>(ui: &mut egui::Ui, value: &mut T, prefix: &str, range: Option<RangeInclusive<T>>) -> egui::Response {
    ui.add(
        match range {
            Some(range) => egui::DragValue::new(value).prefix(prefix).clamp_range(range),
            None => egui::DragValue::new(value).prefix(prefix),
        }
    )
}

/// A wrapper that allows the more idiomatic usage of `input_vec3`.
///
/// ## Example:
/// ``` ignore
/// ui.add(input3(&mut my_vec3));
/// ```
pub fn input<'a, T: egui::emath::Numeric>(value: &'a mut T, prefix: &'a str, range: Option<RangeInclusive<T>>) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| input_ui(ui, value, prefix, range)
}
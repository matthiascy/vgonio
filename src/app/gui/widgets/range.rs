use crate::acq::desc::Range;

pub fn range_ui<T: Copy + egui::emath::Numeric>(ui: &mut egui::Ui, value: &mut Range<T>) -> egui::Response {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut value.start).prefix("start: "));
        ui.add(egui::DragValue::new(&mut value.stop).prefix("stop: "));
        ui.add(egui::DragValue::new(&mut value.step).prefix("step: "));
    })
    .response
}

/// A wrapper that allows the more idiomatic usage of `input_vec3`.
///
/// ## Example:
/// ``` ignore
/// ui.add(input3(&mut my_vec3));
/// ```
pub fn range<T: Copy + egui::emath::Numeric>(value: &mut Range<T>) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| range_ui(ui, value)
}

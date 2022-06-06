use glam::Vec3;

pub fn input3_ui(ui: &mut egui::Ui, value: &mut Vec3) -> egui::Response {
    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(&mut value.x)
                .prefix("x: ")
        );
        ui.add(egui::DragValue::new(&mut value.y)
            .prefix("y: "));
        ui.add(egui::DragValue::new(&mut value.z)
            .prefix("z: "));
    }).response
}

/// A wrapper that allows the more idiomatic usage of `input_vec3`.
///
/// ## Example:
/// ``` ignore
/// ui.add(input3(&mut my_vec3));
/// ```
pub fn input3(value: &mut Vec3) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| input3_ui(ui, value)
}
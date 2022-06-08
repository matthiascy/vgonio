use glam::Vec3;

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
pub fn input3_xyz(value: &mut Vec3) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| input3_ui(ui, value, &["x: ", "y: ", "z: "])
}

pub fn input3_spherical(value: &mut Vec3) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| {
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut value.x).prefix("r: "));
            ui.add(
                egui::DragValue::new(&mut value.y)
                    .prefix("θ: ")
                    .suffix("°")
                    .clamp_range(0.0..=90.0),
            );
            ui.add(
                egui::DragValue::new(&mut value.z)
                    .prefix("φ: ")
                    .suffix("°")
                    .clamp_range(0.0..=360.0),
            );
        })
        .response
    }
}

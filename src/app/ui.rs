pub(crate) fn ui(ctx: &egui::Context) {
    egui::TopBottomPanel::top("menubar-container").show(ctx, |ui| {
        ui.set_enabled(true);
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                ui.set_min_width(200.0);
                if ui.button("Preferences").clicked() {
                    ui.close_menu();
                }
            });
            ui.menu_button("Help", |ui| {
                ui.set_min_width(200.0);
                if ui.button("About ").clicked() {
                    ui.close_menu();
                }
                if ui.button("Support").clicked() {
                    ui.close_menu();
                }
            });
        });
    });
}
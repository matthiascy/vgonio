pub struct AnalysisWorkspace {}

impl epi::App for AnalysisWorkspace {
    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
        egui::SidePanel::right("Extraction panel").show(ctx, |ui| {
            ui.add(egui::Label::new("extraction panel"));
        });
    }

    fn name(&self) -> &str {
        "Analysis"
    }
}

pub struct AnalysisWorkspace {}

impl epi::App for AnalysisWorkspace {
    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        egui::SidePanel::right("Extraction panel").show(ctx, |ui| {
            egui::Label::new("extraction panel");
        });
    }

    fn name(&self) -> &str {
        "Analysis"
    }
}
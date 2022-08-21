use crate::app::gui::ui::Workspace;

pub struct AnalysisWorkspace {}

impl Workspace for AnalysisWorkspace {
    fn name(&self) -> &str { "Analysis" }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("Extraction panel").show(ctx, |ui| {
            ui.add(egui::Label::new("extraction panel"));
        });
    }
}

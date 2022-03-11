use egui::Context;
use epi::Frame;

pub struct Simulation {}

impl epi::App for Simulation {
    fn update(&mut self, ctx: &Context, frame: &Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Label::new("Simulation panel");
        });
    }

    fn name(&self) -> &str {
        "Simulation"
    }
}

pub struct Analysis {}

impl epi::App for Analysis {
    fn update(&mut self, ctx: &Context, frame: &Frame) {
        egui::SidePanel::right("Extraction panel").show(ctx, |ui| {
            egui::Label::new("extraction panel");
        });
    }

    fn name(&self) -> &str {
        "Analysis"
    }
}

use std::any::Any;
use egui::{Context, Ui};
use egui::plot::{Legend, Plot, Corner};
use crate::app::gui::tools::{Scratch, Tool};

pub struct Plotting {
    pub legend: Legend,
}

impl Default for Plotting {
    fn default() -> Self {
        let legend = Legend::default()
            .text_style(egui::TextStyle::Monospace)
            .background_alpha(1.0)
            .position(Corner::RightTop);
        Self {
            legend
        }
    }
}

impl Tool for Plotting {
    fn name(&self) -> &'static str {
        "Plot"
    }

    fn show(&mut self, ctx: &Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut Ui) {
        let plot = Plot::new("plotting")
            .legend(self.legend.clone())
            .data_aspect(1.0);
        plot.show(ui, |_| {});
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
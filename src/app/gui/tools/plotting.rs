use crate::app::gui::tools::Tool;
use egui::{
    plot::{Corner, Legend, Plot},
    Context, Ui,
};
use std::any::Any;

pub struct PlottingInspector {
    pub legend: Legend,
}

impl Default for PlottingInspector {
    fn default() -> Self {
        let legend = Legend::default()
            .text_style(egui::TextStyle::Monospace)
            .background_alpha(1.0)
            .position(Corner::RightTop);
        Self { legend }
    }
}

impl Tool for PlottingInspector {
    fn name(&self) -> &'static str { "Plotting" }

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

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

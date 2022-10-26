use crate::app::gui::tools::Tool;
use egui::{
    emath, Color32, Context, Frame, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Window,
};
use std::any::Any;

pub struct Scratch {
    lines: Vec<(Stroke, Vec<Pos2>)>,
    stroke: Stroke,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            lines: Default::default(),
            stroke: Stroke::new(1.0, Color32::from_rgb(25, 200, 100)),
        }
    }
}

impl Tool for Scratch {
    fn name(&self) -> &'static str { "Scratch" }

    fn show(&mut self, ctx: &Context, open: &mut bool) {
        Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut Ui) {
        self.ui_control(ui);
        Frame::canvas(ui.style()).show(ui, |ui| {
            self.ui_content(ui);
        });
    }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

impl Scratch {
    fn ui_control(&mut self, ui: &mut Ui) -> Response {
        ui.horizontal(|ui| {
            egui::stroke_ui(ui, &mut self.stroke, "");
            ui.separator();
            if ui.button("Clear").clicked() {
                self.lines.clear();
            }
        })
        .response
    }

    fn ui_content(&mut self, ui: &mut Ui) -> Response {
        let (mut response, painter) =
            ui.allocate_painter(ui.available_size_before_wrap(), Sense::drag());
        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, response.rect.square_proportions()),
            response.rect,
        );
        let from_screen = to_screen.inverse();

        if self.lines.is_empty() {
            self.lines.push((self.stroke, vec![]));
        }

        let current_line = self.lines.last_mut().unwrap();

        if let Some(pointer_pos) = response.interact_pointer_pos() {
            let canvas_pos = from_screen * pointer_pos;
            if current_line.1.last() != Some(&canvas_pos) {
                current_line.1.push(canvas_pos);
                response.mark_changed();
            }
        } else if !current_line.1.is_empty() {
            self.lines.push((self.stroke, vec![]));
            response.mark_changed();
        }

        let mut shapes = vec![];
        for line in &self.lines {
            if line.1.len() >= 2 {
                let points: Vec<Pos2> = line.1.iter().map(|p| to_screen * *p).collect();
                shapes.push(Shape::line(points, line.0));
            }
        }
        painter.extend(shapes);
        response
    }
}

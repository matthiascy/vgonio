use egui::{Stroke, Pos2};

pub struct Sketch {
    lines: Vec<(Stroke, Vec<Pos2>)>,
    stroke: Stroke,
}

impl Default for Sketch {
    fn default() -> Self {
        Self {
            lines: Default::default(),
            stroke: Stroke::new(1.0, egui::Color32::from_rgb(25, 200, 100)),
        }
    }
}

impl Sketch {
    fn ui_control(&mut self, ui: &mut egui::Ui) -> egui::Response {
        ui.horizontal(|ui| {
            egui::stroke_ui(ui, &mut self.stroke, "Stroke");
            ui.separator();
            if ui.button("Clear").clicked() {
                self.lines.clear();
            }
        })
        .response
    }

    fn ui_content(&mut self, ui: &mut egui::Ui) -> egui::Response {
        let (mut response, painter) =
            ui.allocate_painter(ui.available_size_before_wrap(), egui::Sense::drag());
        let to_screen = egui::emath::RectTransform::from_to(
            egui::Rect::from_min_size(egui::Pos2::ZERO, response.rect.square_proportions()),
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
                let points: Vec<egui::Pos2> = line.1.iter().map(|p| to_screen * *p).collect();
                shapes.push(egui::Shape::line(points, line.0));
            }
        }
        painter.extend(shapes);
        response
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.ui_control(ui);
        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            self.ui_content(ui);
        });
    }
}
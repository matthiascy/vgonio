use crate::app::ui::gizmo::VgonioGizmo;
use egui::epaint::CircleShape;
use egui_gizmo::GizmoMode;
use glam::Mat4;

pub struct SimulationWorkspace {
    is_sim_win_open: bool,
    is_view_gizmo_open: bool,
    sim_win: SimulationWindow,
    view_gizmo: VgonioGizmo,
}

impl Default for SimulationWorkspace {
    fn default() -> Self {
        Self {
            is_sim_win_open: true,
            is_view_gizmo_open: true,
            sim_win: Default::default(),
            view_gizmo: Default::default(),
        }
    }
}

impl epi::App for SimulationWorkspace {
    fn update(&mut self, ctx: &egui::Context, frame: &epi::Frame) {
        // egui::SidePanel::left("Left Panel").show(ctx, |ui| {
        //     egui::Label::new("Simulation panel");
        // });
        self.sim_win.show(ctx, &mut self.is_sim_win_open);
        self.view_gizmo.show(ctx, &mut self.is_view_gizmo_open);
    }

    fn name(&self) -> &str {
        "Simulation"
    }
}

impl SimulationWorkspace {
    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.view_gizmo.update_matrices(model, view, proj)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SensorShape {
    Rectangle,
    Circle,
}

pub struct SimulationWindow {
    sensor_shape: SensorShape,
}

impl Default for SimulationWindow {
    fn default() -> Self {
        Self {
            sensor_shape: SensorShape::Circle,
        }
    }
}

impl SimulationWindow {
    pub fn name(&self) -> &'static str {
        &"Simulation"
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .default_height(500.0)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Sensor", |ui| {
            egui::ComboBox::from_label("Take your pick")
                .selected_text(format!("{:?}", self.sensor_shape))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sensor_shape, SensorShape::Circle, "Circle");
                    ui.selectable_value(
                        &mut self.sensor_shape,
                        SensorShape::Rectangle,
                        "Rectangle",
                    );
                });
            ui.end_row();
        });
    }
}

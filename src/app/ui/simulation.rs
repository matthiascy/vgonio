use crate::app::ui::gizmo::VgonioGizmo;
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
    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
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
        // Grid of two columns.
        egui::Grid::new("sim_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.add(egui::Label::new("Sensor Shape:"));
                ui.horizontal(|ui| {
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

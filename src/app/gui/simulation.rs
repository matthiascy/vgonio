use crate::app::gui::gizmo::VgonioGizmo;
use crate::app::gui::{UserEvent, VisualDebugger};
use egui_gizmo::{GizmoMode, GizmoOrientation};
use glam::Mat4;
use std::sync::Arc;
use winit::event_loop::EventLoopProxy;

pub struct SimulationWorkspace {
    is_sim_win_open: bool,
    is_view_gizmo_open: bool,
    is_visual_debugger_open: bool,
    sim_win: SimulationWindow,
    view_gizmo: VgonioGizmo,
    visual_debugger: VisualDebugger,
}

impl epi::App for SimulationWorkspace {
    fn update(&mut self, ctx: &egui::Context, _frame: &epi::Frame) {
        // egui::SidePanel::left("Left Panel").show(ctx, |ui| {
        //     egui::Label::new("Simulation panel");
        // });
        self.sim_win.show(ctx, &mut self.is_sim_win_open);
        self.view_gizmo.show(ctx, &mut self.is_view_gizmo_open);
        self.visual_debugger
            .show(ctx, &mut self.is_visual_debugger_open);
    }

    fn name(&self) -> &str {
        "Simulation"
    }
}

impl SimulationWorkspace {
    pub fn new(evlp: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            is_sim_win_open: false,
            is_view_gizmo_open: false,
            is_visual_debugger_open: false,
            sim_win: SimulationWindow::new(evlp.clone()),
            view_gizmo: VgonioGizmo::new(GizmoMode::Translate, GizmoOrientation::Global),
            visual_debugger: VisualDebugger::new(evlp),
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.view_gizmo.update_matrices(model, view, proj)
    }

    pub fn open_visual_debugger(&mut self) {
        self.is_visual_debugger_open = true;
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SensorShape {
    Rectangle,
    Circle,
}

pub struct SimulationWindow {
    is_grid_enabled: bool,
    sensor_shape: SensorShape,
    evlp: Arc<EventLoopProxy<UserEvent>>,
}

impl SimulationWindow {
    pub fn new(evlp: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            is_grid_enabled: true,
            sensor_shape: SensorShape::Rectangle,
            evlp,
        }
    }

    pub fn name(&self) -> &'static str {
        "Simulation"
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .default_height(500.0)
            .show(ctx, |ui| self.ui(ui));

        egui::Area::new("controls")
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .show(ctx, |ui| {
                egui::Grid::new("controls_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new("Visual Grid:"));
                        let res = ui.add(super::widgets::toggle(&mut self.is_grid_enabled));
                        if res.changed() && self.evlp.send_event(UserEvent::ToggleGrid).is_err() {
                            log::warn!("[EVENT] Failed to send ToggleGrid event");
                        }
                        ui.end_row();
                    });
            });
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

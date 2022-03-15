use egui::{Align2, Color32, LayerId};
use egui_gizmo::{
    Gizmo, GizmoMode, GizmoOrientation, GizmoResult, GizmoVisuals, DEFAULT_SNAP_ANGLE,
    DEFAULT_SNAP_DISTANCE,
};
use glam::Mat4;

pub struct VgonioGizmo {
    model_matrix: Mat4,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    mode: GizmoMode,
    orientation: GizmoOrientation,
    last_gizmo_response: Option<GizmoResult>,
}

impl Default for VgonioGizmo {
    fn default() -> Self {
        Self {
            model_matrix: Mat4::IDENTITY,
            view_matrix: Mat4::IDENTITY,
            proj_matrix: Mat4::IDENTITY,
            mode: GizmoMode::Translate,
            orientation: GizmoOrientation::Global,
            last_gizmo_response: None,
        }
    }
}

impl VgonioGizmo {
    pub fn new(mode: GizmoMode, orientation: GizmoOrientation) -> Self {
        Self {
            model_matrix: Mat4::IDENTITY,
            view_matrix: Mat4::IDENTITY,
            proj_matrix: Mat4::IDENTITY,
            mode,
            orientation,
            last_gizmo_response: None,
        }
    }

    pub fn update_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.model_matrix = model;
        self.view_matrix = view;
        self.proj_matrix = proj;
    }

    pub fn name(&self) -> &'static str {
        &"Gizmo"
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Area::new("Viewport")
            // .fixed_pos((0.0, 0.0))
            .anchor(Align2::RIGHT_BOTTOM, (-150.0, -150.0))
            .interactable(false)
            .show(ctx, |ui| {
                // ui.with_layer_id(LayerId::background(), |ui| {
                // Snapping is enabled with ctrl key.
                // let snapping = ui.input().modifiers.command;
                // // Snap angle to use for rotation when snapping is enabled.
                // // Smaller snap angle is used when shift key is pressed.
                // let snap_angle = if ui.input().modifiers.shift {
                //     DEFAULT_SNAP_ANGLE / 2.0
                // } else {
                //     DEFAULT_SNAP_ANGLE
                // };

                // // Snap distance to use for translation when snapping is enabled.
                // // Smaller snap distance is used when shift key is pressed.
                // let snap_distance = if ui.input().modifiers.shift {
                //     DEFAULT_SNAP_DISTANCE / 2.0
                // } else {
                //     DEFAULT_SNAP_DISTANCE
                // };

                let visuals = GizmoVisuals {
                    stroke_width: 6.0,
                    x_color: Color32::from_rgb(255, 0, 148),
                    y_color: Color32::from_rgb(148, 255, 0),
                    z_color: Color32::from_rgb(0, 148, 255),
                    s_color: Color32::WHITE,
                    inactive_alpha: 0.5,
                    highlight_alpha: 1.0,
                    highlight_color: None,
                    gizmo_size: 75.0,
                };

                let gizmo = Gizmo::new("My gizmo")
                    .view_matrix(self.view_matrix.to_cols_array_2d())
                    .projection_matrix(self.proj_matrix.to_cols_array_2d())
                    .model_matrix(self.model_matrix.to_cols_array_2d())
                    .mode(self.mode)
                    .orientation(self.orientation)
                    // .snapping(snapping)
                    // .snap_angle(snap_angle)
                    // .snap_distance(snap_distance)
                    .visuals(visuals);

                self.last_gizmo_response = gizmo.interact(ui);

                if let Some(gizmo_response) = self.last_gizmo_response {
                    self.model_matrix = Mat4::from_cols_array_2d(&gizmo_response.transform);
                }
            });
        // });
    }
}

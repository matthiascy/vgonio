use egui::{Align2, Color32, LayerId};
use egui_gizmo::{Gizmo, GizmoMode, GizmoOrientation, GizmoResult, GizmoVisuals};
use vgcore::math::Mat4;

// TODO: customise gizmo visuals: implement Viewport (Navigation) Gizmo and
//       Object(Transform) Gizmo

pub struct NavigationGizmo {
    model_matrix: Mat4,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    mode: GizmoMode,
    orientation: GizmoOrientation,
    last_gizmo_response: Option<GizmoResult>,
}

impl Default for NavigationGizmo {
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

impl NavigationGizmo {
    pub fn new(orientation: GizmoOrientation) -> Self {
        log::info!("Creating NavigationGizmo");
        Self {
            model_matrix: Mat4::IDENTITY,
            view_matrix: Mat4::IDENTITY,
            proj_matrix: Mat4::IDENTITY,
            mode: GizmoMode::Translate,
            orientation,
            last_gizmo_response: None,
        }
    }

    pub fn update_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.model_matrix = model;
        self.view_matrix = view;
        self.proj_matrix = proj;
    }

    pub fn name(&self) -> &'static str { "Gizmo" }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
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
            .snapping(true)
            // .snap_angle(snap_angle)
            // .snap_distance(snap_distance)
            .visuals(visuals);

        self.last_gizmo_response = gizmo.interact(ui);

        if let Some(gizmo_response) = self.last_gizmo_response {
            self.model_matrix = gizmo_response.transform();
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        egui::Area::new(self.name())
            //.fixed_pos((0.0, 0.0))+
            .anchor(Align2::LEFT_TOP, (0.0, 0.0))
            .movable(true)
            .show(ctx, |ui| self.ui(ui));
    }
}

use super::InputState;
use crate::app::gfx::camera::{
    Camera, CameraController, CameraUniform, OrbitControls, Projection, ProjectionKind,
};

pub struct CameraState {
    pub(crate) camera: Camera,
    pub(crate) uniform: CameraUniform,
    pub(crate) controller: OrbitControls,
    pub(crate) projection: Projection,
}

impl CameraState {
    pub fn new(camera: Camera, projection: Projection, kind: ProjectionKind) -> Self {
        let uniform = CameraUniform::new(&camera, &projection, kind);
        let controller = OrbitControls::new(0.3, f32::INFINITY, 200.0, 400.0, 100.0);
        Self {
            camera,
            uniform,
            controller,
            projection,
        }
    }

    pub fn update_with_input_state(
        &mut self,
        input: &InputState,
        dt: std::time::Duration,
        kind: ProjectionKind,
    ) {
        self.controller.update(input, &mut self.camera, dt);
        self.uniform.update(&self.camera, &self.projection, kind);
    }
}

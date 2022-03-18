use super::InputState;
use crate::gfx::camera::{Camera, CameraController, CameraUniform, OrbitControls, Projection};

pub struct CameraState {
    pub(crate) camera: Camera,
    pub(crate) uniform: CameraUniform,
    pub(crate) controller: OrbitControls,
    pub(crate) projection: Projection,
}

impl CameraState {
    pub fn new(camera: Camera, projection: Projection) -> Self {
        let uniform = CameraUniform::new(&camera, &projection);
        let controller = OrbitControls::new(0.3, f32::INFINITY, 200.0, 400.0, 100.0);
        Self {
            camera,
            uniform,
            controller,
            projection,
        }
    }

    pub fn update(&mut self, input: &InputState, dt: std::time::Duration) {
        self.controller.update(input, &mut self.camera, dt);
        self.uniform.update(&self.camera, &self.projection);
    }
}

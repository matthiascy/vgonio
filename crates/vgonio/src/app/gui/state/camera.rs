use base::{input::InputState, math::Vec3};
use gxtk::camera::{
    Camera, CameraController, CameraUniform, OrbitControls, Projection, ProjectionKind,
};

pub struct CameraState {
    pub(crate) camera: Camera,
    pub(crate) uniform: CameraUniform,
    pub(crate) controller: OrbitControls,
    pub(crate) projection: Projection,
}

impl CameraState {
    /// Creates a new instance of the camera state with the default values,
    /// except for the size of the image plane.
    pub fn default_with_size(width: u32, height: u32) -> Self {
        let camera = Camera::new(Vec3::new(0.0, -10.0, 8.0), Vec3::ZERO);
        let projection = Projection::new(0.1, 100.0, 75.0f32.to_radians(), width, height);
        CameraState::new(camera, projection, ProjectionKind::Perspective)
    }

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

    pub fn update(&mut self, input: &InputState, dt: std::time::Duration, kind: ProjectionKind) {
        self.controller.update(input, &mut self.camera, dt);
        self.uniform.update(&self.camera, &self.projection, kind);
    }
}

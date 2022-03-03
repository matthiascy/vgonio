use crate::math::Vec3;

struct OrthonormalBasis {
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
}

impl OrthonormalBasis {
    pub fn update(&mut self, eye: Vec3, target: Vec3, up: Vec3) {
        self.axis_y = (target - eye).normalize();
        self.axis_x = self.axis_y.cross(up).normalize();
        self.axis_z = self.axis_x.cross(self.axis_y).normalize();
    }
}

struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
}

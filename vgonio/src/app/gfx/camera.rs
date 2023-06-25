use crate::app::gui::state::InputState;
use std::cmp::Ordering::Equal;
use vgcore::math::{Mat3, Mat4, Vec3, Vec4, Vec4Swizzles};
use winit::event::{MouseButton, VirtualKeyCode};

// pub struct OrthonormalBasis {
//     axis_x: Vec3,
//     axis_y: Vec3,
//     axis_z: Vec3,
// }
//
// impl OrthonormalBasis {
//     pub fn update(&mut self, eye: Vec3, target: Vec3, up: Vec3) {
//         self.axis_y = (target - eye).normalize();
//         self.axis_x = self.axis_y.cross(up).normalize();
//         self.axis_z = self.axis_x.cross(self.axis_y).normalize();
//     }
// }

// #[repr(u32)]
// #[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
// enum ReferenceFrame {
//     RightHandedZUp, // 3ds Max, Maya Z up, Blender
//     RightHandedYUp, // Maya Y up
//     LeftHandedZUp,  // Unreal 4
//     LeftHandedYUp,  // Unity, D3D, PBRT, OpenGL NDC, DX12 NDC
// }

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    Perspective,
    Orthographic,
}

/// Projection transformation
pub struct Projection {
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
    /// Vertical field of view
    pub fov: f32,
    /// Image width
    pub width: f32,
    /// Image height
    pub height: f32,
    /// Aspect ratio of camera image
    pub aspect: f32,
}

impl Default for Projection {
    fn default() -> Self {
        Self {
            near: 0.1,
            far: 100.0,
            fov: 45.0f32.to_radians(),
            width: 512.0,
            height: 288.0,
            aspect: 16.0 / 9.0,
        }
    }
}

impl Projection {
    pub fn new(near: f32, far: f32, fov: f32, width: u32, height: u32) -> Self {
        Self {
            near,
            far,
            fov,
            width: width as _,
            height: height as _,
            aspect: width as f32 / height as f32,
        }
    }

    pub fn orthographic_matrix(near: f32, far: f32, width: f32, height: f32) -> Mat4 {
        let half_width = width / 2.0;
        let half_height = height / 2.0;
        Mat4::orthographic_rh(
            -half_width,
            half_width,
            -half_height,
            half_height,
            near,
            far,
        )
    }

    pub fn matrix(&self, kind: ProjectionKind) -> Mat4 {
        let scale_depth = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 0.0),
            Vec4::new(0.0, 0.0, 0.5, 1.0),
        );
        scale_depth
            * match kind {
                ProjectionKind::Perspective => {
                    Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
                }
                ProjectionKind::Orthographic => Mat4::orthographic_rh(
                    -self.width / 2.0,
                    self.width / 2.0,
                    -self.height / 2.0,
                    self.height / 2.0,
                    self.near,
                    self.far,
                ),
            }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as _;
        self.height = height as _;
        self.aspect = width as f32 / height as f32;
    }
}

/// Localization: X-right, Y-forward, Z-up
/// Chirality: right-handed
pub struct Camera {
    /// Location of the view point
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3, up: Vec3) -> Self {
        let forward = (target - position).normalize();
        let right = {
            let cos = forward.dot(up.normalize());
            let up = if cos.partial_cmp(&1.0) == Some(Equal) {
                -Vec3::X
            } else if cos.partial_cmp(&-1.0) == Some(Equal) {
                Vec3::X
            } else {
                up
            };
            forward.cross(up).normalize()
        };
        let up = right.cross(forward).normalize();
        Self {
            eye: position,
            target,
            up,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        // let right = forward.cross(Vec3::Z);
        // let up = right.cross(forward);
        //
        // Mat4::from_cols(
        //     Vec4::new(right.x, up.x, forward.x, 0.0),
        //     Vec4::new(right.y, up.y, forward.y, 0.0),
        //     Vec4::new(right.z, up.z, forward.z, 0.0),
        //     Vec4::new(
        //         -right.dot(self.position),
        //         -up.dot(self.position),
        //         -forward.dot(self.position),
        //         1.0,
        //     ),
        // )
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    // pub fn view_proj_matrix(&self) -> Mat4 {
    //     // Scale depth from range [-1, 1] to range [0, 1]
    //     let scale_depth = Mat4::from_cols(
    //         Vec4::new(1.0, 0.0, 0.0, 0.0),
    //         Vec4::new(0.0, 1.0, 0.0, 0.0),
    //         Vec4::new(0.0, 0.0, 0.5, 0.0),
    //         Vec4::new(0.0, 0.0, 0.5, 1.0),
    //     );
    //     scale_depth * self.proj_matrix() * self.matrix()
    // }

    pub fn forward(&self) -> Vec3 { (self.target - self.eye).normalize() }

    pub fn right(&self) -> Vec3 { self.forward().cross(self.up).normalize() }

    pub fn up(&self) -> Vec3 { self.up }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewProjUniform {
    pub view: Mat4,
    pub proj: Mat4,
}

impl ViewProjUniform {
    pub const SIZE_IN_BYTES: usize = std::mem::size_of::<Self>();
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct CameraUniform {
    pub view_proj: ViewProjUniform,
    pub view_proj_inv: ViewProjUniform,
}

impl CameraUniform {
    pub fn new(camera: &Camera, projection: &Projection, kind: ProjectionKind) -> Self {
        let view_matrix = camera.view_matrix();
        let proj_matrix = projection.matrix(kind);
        Self {
            view_proj: ViewProjUniform {
                view: view_matrix,
                proj: proj_matrix,
            },
            view_proj_inv: ViewProjUniform {
                view: view_matrix.inverse(),
                proj: proj_matrix.inverse(),
            },
        }
    }

    pub fn update(&mut self, camera: &Camera, projection: &Projection, kind: ProjectionKind) {
        *self = Self::new(camera, projection, kind);
    }
}

pub trait CameraController {
    fn update(&mut self, input: &InputState, camera: &mut Camera, dt: std::time::Duration);
}

pub struct OrbitControls {
    /// How far you can zoom out
    max_zoom_dist: f32,
    /// How far you can zoom in
    min_zoom_dist: f32,
    /// Whether or not panning is enabled
    is_panning_enabled: bool,
    /// Whether or not rotation is enabled
    is_rotation_enabled: bool,
    /// Whether or not zooming is enabled
    is_zooming_enabled: bool,
    /// The panning speed
    pan_speed: f32,
    /// Rotation speed
    rotate_speed: f32,
    /// Zooming speed
    zoom_speed: f32,
}

impl Default for OrbitControls {
    fn default() -> Self {
        Self {
            max_zoom_dist: f32::INFINITY,
            min_zoom_dist: 0.0,
            is_panning_enabled: true,
            is_rotation_enabled: true,
            is_zooming_enabled: true,
            pan_speed: 0.3,
            rotate_speed: 1.0,
            zoom_speed: 0.4,
        }
    }
}

impl OrbitControls {
    pub fn new(
        min_zoom_dist: f32,
        max_zoom_dist: f32,
        pan_speed: f32,
        rotate_speed: f32,
        zoom_speed: f32,
    ) -> Self {
        Self {
            max_zoom_dist,
            min_zoom_dist,
            is_panning_enabled: true,
            is_rotation_enabled: true,
            is_zooming_enabled: true,
            pan_speed,
            rotate_speed,
            zoom_speed,
        }
    }

    pub fn toggle_panning(&mut self) { self.is_panning_enabled = !self.is_panning_enabled; }

    pub fn toggle_rotation(&mut self) { self.is_rotation_enabled = !self.is_rotation_enabled; }

    pub fn toggle_zooming(&mut self) { self.is_zooming_enabled = !self.is_zooming_enabled; }

    fn pan(&self, dx: f32, dy: f32, camera: &mut Camera) {
        let delta = camera.right() * dx * self.pan_speed + camera.up() * dy * self.pan_speed;
        camera.eye += delta;
        camera.target += delta;
    }

    fn rotate(&self, dx: f32, dy: f32, camera: &mut Camera) {
        let mut position = Vec4::from((camera.eye, 1.0));
        let pivot = Vec4::from((camera.target, 1.0));
        let forward = camera.forward();
        let up = camera.up();
        let right = camera.right();

        let cos = forward.dot(camera.up);

        if cos > 0.99 {
            camera.up = Mat3::from_axis_angle(right, 45.0f32.to_radians()) * up;
        } else if cos < -0.99 {
            camera.up = Mat3::from_axis_angle(-right, 45.0f32.to_radians()) * up;
        }

        let h_angle = dx * self.rotate_speed;
        let v_angle = dy * self.rotate_speed;

        let rot = Mat4::from_axis_angle(camera.right(), v_angle)
            * Mat4::from_axis_angle(camera.up(), h_angle);
        position = pivot + rot * (position - pivot);

        camera.eye = position.xyz();
    }

    fn zoom(&self, delta: f32, camera: &mut Camera) {
        let eye = camera.eye + camera.forward() * delta * self.zoom_speed;
        let distance = (camera.target - eye).length();
        if distance > self.min_zoom_dist && distance < self.max_zoom_dist {
            camera.eye = eye;
        }
    }
}

impl CameraController for OrbitControls {
    fn update(&mut self, input: &InputState, camera: &mut Camera, dt: std::time::Duration) {
        let is_middle_button_pressed = *input.mouse_map.get(&MouseButton::Middle).unwrap_or(&false);
        let is_shift_pressed = input.is_key_pressed(VirtualKeyCode::LShift)
            || input.is_key_pressed(VirtualKeyCode::RShift);
        let is_ctrl_pressed = input.is_key_pressed(VirtualKeyCode::LControl)
            || input.is_key_pressed(VirtualKeyCode::RControl);
        let [dx, dy] = input.cursor_delta;
        let scroll_delta = input.scroll_delta * 5.0;

        let dt = dt.as_secs_f32();

        if self.is_panning_enabled
            && is_middle_button_pressed
            && is_shift_pressed
            && !is_ctrl_pressed
        {
            // Shift + Middle --> panning
            self.pan(-dx * dt, dy * dt, camera);
        }

        if self.is_zooming_enabled
            && is_middle_button_pressed
            && !is_shift_pressed
            && is_ctrl_pressed
        {
            // Ctrl + Middle --> zooming
            self.zoom(-dy * dt * 5.0, camera);
        }

        if self.is_zooming_enabled && scroll_delta != 0.0 {
            // Wheel --> zooming
            self.zoom(-scroll_delta * dt, camera);
        }

        if self.is_rotation_enabled
            && is_middle_button_pressed
            && !is_shift_pressed
            && !is_ctrl_pressed
        {
            // Middle --> rotate
            self.rotate(dx * dt, dy * dt, camera);
        }
    }
}

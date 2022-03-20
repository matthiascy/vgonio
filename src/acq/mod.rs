use crate::gfx::camera::{Projection, ProjectionKind};
use glam::Vec3;

/// Light source used for acquisition of shadowing and masking function.
pub struct LightSource {
    pub pos: Vec3,
    pub proj: Projection,
    pub proj_kind: ProjectionKind,
}

/// Light space matrix used for generation of depth map.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightSourceRaw([f32; 16]);

impl LightSource {
    pub fn to_raw(&self) -> LightSourceRaw {
        let forward = -self.pos;
        let up = if forward == -Vec3::Y {
            Vec3::new(1.0, 1.0, 0.0).normalize()
        } else {
            Vec3::Y
        };
        let view = glam::Mat4::look_at_rh(self.pos, Vec3::ZERO, up);
        let proj = self.proj.matrix(self.proj_kind);
        LightSourceRaw((proj * view).to_cols_array())
    }
}

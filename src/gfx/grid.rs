use crate::math::{Mat4, Vec3, IDENTITY_MAT4};

pub struct VisualGrid {
    pub transform: Mat4,
    /// Extent of grid in X, Y plane
    pub grid_size: f32,
    /// Minimum size of one cell
    pub cell_size: f32,
    /// sRGB color and alpha of thin lines
    pub thin_lines_color: Vec3,
    /// sRGB color and alpha of thick lines (every tenth line)
    pub thick_lines_color: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisualGridUniforms {
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub view_inv: [f32; 16],
    pub proj_inv: [f32; 16],
}

impl Default for VisualGridUniforms {
    fn default() -> Self {
        Self {
            view: IDENTITY_MAT4,
            proj: IDENTITY_MAT4,
            view_inv: IDENTITY_MAT4,
            proj_inv: IDENTITY_MAT4,
        }
    }
}

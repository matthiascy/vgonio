use crate::math::{Mat4, Vec3};

pub struct VisualGrid {
    transform: Mat4,
    /// Extent of grid in X, Y plane
    grid_size: f32,
    /// Minimum size of one cell
    cell_size: f32,
    /// sRGB color and alpha of thin lines
    thin_lines_color: Vec3,
    /// sRGB color and alpha of thick lines (every tenth line)
    thick_lines_color: Vec3,
}

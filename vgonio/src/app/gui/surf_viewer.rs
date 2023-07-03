use crate::app::gui::{state::camera::CameraState, VisualGridState};

/// Surface viewer.
pub struct SurfViewer {
    camera: CameraState,
    /// State of the visual grid rendering, including the pipeline, binding
    /// groups, and buffers.
    visual_grid_state: VisualGridState,
}

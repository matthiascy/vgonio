mod analysis;
mod context;
mod gizmo;
mod simulation;
mod tools;
mod ui;
mod widgets;

pub use context::*;
pub(crate) use tools::VisualDebugTool;
pub use ui::VgonioGui;

/// User defined event.
#[derive(Debug)]
pub enum UserEvent {
    Quit,
    RequestRedraw,
    OpenFile(std::path::PathBuf),
    ToggleGrid,
    UpdateSurfaceScaleFactor(f32),
    UpdateDepthMap,
}

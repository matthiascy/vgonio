mod analysis;
mod context;
mod gizmo;
mod simulation;
mod ui;
mod widgets;

pub use context::*;
pub use ui::VgonioGui;

/// User defined event.
pub enum UserEvent {
    RequestRedraw,
    OpenFile(std::path::PathBuf),
    ToggleGrid,
}

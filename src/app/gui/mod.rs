mod analysis;
mod context;
mod gizmo;
mod simulation;
mod ui;

pub use context::*;
pub use ui::VgonioGui;

/// User defined event.
pub enum UserEvent {
    RequestRedraw,
    OpenFile(std::path::PathBuf),
}

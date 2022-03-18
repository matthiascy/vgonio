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

/// Repaint signal type that egui needs for requesting a repaint from another
/// thread. It sends the custom RequestRedraw event to the winit event loop.
pub struct RepaintSignal(pub(crate) std::sync::Mutex<winit::event_loop::EventLoopProxy<UserEvent>>);

impl epi::backend::RepaintSignal for RepaintSignal {
    fn request_repaint(&self) {
        self.0
            .lock()
            .unwrap()
            .send_event(UserEvent::RequestRedraw)
            .ok();
    }
}

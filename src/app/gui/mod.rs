mod analysis;
mod context;
mod gizmo;
mod simulation;
mod tools;
mod ui;
mod widgets;

pub use context::*;
pub use tools::*;
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

// /// Repaint signal type that egui needs for requesting a repaint from another
// /// thread. It sends the custom RequestRedraw event to the winit event loop.
// pub struct RepaintSignal(pub(crate)
// std::sync::Mutex<winit::event_loop::EventLoopProxy<UserEvent>>);
//
// impl epi::backend::RepaintSignal for RepaintSignal {
//     fn request_repaint(&self) {
//         self.0
//             .lock()
//             .unwrap()
//             .send_event(UserEvent::RequestRedraw)
//             .ok();
//     }
// }

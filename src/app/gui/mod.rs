mod analysis;
mod context;
mod gizmo;
mod plotter;
mod simulation;
mod tools;
mod ui;
mod widgets;

use crate::acq::{Ray, RtcMethod};
pub use context::*;
use glam::IVec2;
pub(crate) use tools::{trace_ray_grid_dbg, trace_ray_standard_dbg, VisualDebugger};
pub use ui::VgonioGui;

/// Events used by Vgonio application.
#[derive(Debug)]
pub enum VgonioEvent {
    Quit,
    RequestRedraw,
    OpenFile(std::path::PathBuf),
    ToggleGrid,
    UpdateSurfaceScaleFactor(f32),
    UpdateDepthMap,
    TraceRayDbg {
        ray: Ray,
        max_bounces: u32,
        method: RtcMethod,
    },
    ToggleDebugDrawing,
    ToggleSurfaceVisibility,
    UpdateDebugT(f32),
    UpdatePrimId(u32),
    UpdateCellPos(IVec2),
}

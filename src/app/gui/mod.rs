mod analysis;
mod context;
mod gizmo;
mod plotter;
mod simulation;
mod sketch;
mod tools;
mod ui;
mod widgets;

use crate::acq::{Ray, RayTracingMethod};
pub use context::*;
use glam::IVec2;
pub(crate) use tools::{trace_ray_grid_dbg, trace_ray_standard_dbg, VisualDebugTool};
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
        method: RayTracingMethod,
    },
    ToggleDebugDrawing,
    ToggleSurfaceVisibility,
    UpdateDebugT(f32),
    UpdatePrimId(u32),
    UpdateCellPos(IVec2),
}

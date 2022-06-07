mod analysis;
mod context;
mod gizmo;
mod simulation;
mod tools;
mod ui;
mod widgets;

use crate::acq::ray::Ray;
use crate::acq::tracing::RayTracingMethod;
pub use context::*;
pub(crate) use tools::VisualDebugTool;
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
}

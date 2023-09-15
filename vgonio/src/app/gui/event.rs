use crate::{
    app::{
        cache::Handle,
        gui::{notify::NotifyKind, theme::ThemeKind},
    },
    fitting::FittingProblemKind,
    measure::{
        bsdf::{
            detector::DetectorPatches,
            emitter::{EmitterSamples, MeasurementPoints},
            rtc::RtcMethod,
        },
        data::MeasurementData,
        params::{
            AdfMeasurementParams, BsdfMeasurementParams, MeasurementKind, MsfMeasurementParams,
        },
    },
};
use uuid::Uuid;
use vgcore::{
    math::{IVec2, Sph2},
    units::Degrees,
};
use vgsurf::{MicroSurface, MicroSurfaceMesh};

use super::outliner::OutlinerItem;

/// Event loop proxy with Vgonio events.
pub type EventLoopProxy = winit::event_loop::EventLoopProxy<VgonioEvent>;

/// Events used by Vgonio application.
#[derive(Debug)]
#[non_exhaustive]
pub enum VgonioEvent {
    Quit,
    RequestRedraw,
    OpenFiles(Vec<rfd::FileHandle>),
    ToggleSurfaceVisibility,
    CheckVisibleFacets {
        m_azimuth: Degrees,
        m_zenith: Degrees,
        opening_angle: Degrees,
    },
    BsdfViewer(BsdfViewerEvent),
    Debugging(DebuggingEvent),
    Measure(MeasureEvent),
    Notify {
        kind: NotifyKind,
        text: String,
        time: f32,
    },
    /// Update the theme.
    UpdateThemeKind(ThemeKind),
    SurfaceViewer(SurfaceViewerEvent),
    Outliner(OutlinerEvent),
    Graphing {
        kind: MeasurementKind,
        data: Handle<MeasurementData>,
        independent: bool,
    },
    Fitting {
        kind: FittingProblemKind,
        data: Handle<MeasurementData>,
    },
}

/// Events used by [`SurfaceViewer`].
#[derive(Debug)]
pub enum SurfaceViewerEvent {
    /// Notify the GUI backend that a surface viewer has been created.
    Create {
        /// ID of the surface viewer.
        uuid: Uuid,
        /// Texture ID of the surface viewer output.
        tex_id: egui::TextureId,
    },
    /// Notify the GUI backend that a surface viewer has been resized.
    Resize {
        /// ID of the surface viewer.
        uuid: Uuid,
        /// New size of the surface viewer.
        size: (u32, u32),
    },
    /// Notify the GUI backend that a surface viewer has been closed.
    Close {
        /// ID of the surface viewer.
        uuid: Uuid,
    },
    UpdateSurfaceList {
        /// List of surfaces to display.
        surfaces: Vec<Handle<MicroSurface>>,
    },
}

/// Events used by BSDF viewer.`
#[derive(Debug)]
pub enum BsdfViewerEvent {
    /// Enable/disable the rendering of a texture.
    ToggleView(egui::TextureId),
    /// Update the BSDF data buffer for a texture.
    UpdateBuffer {
        /// ID of the texture
        id: egui::TextureId,
        /// Buffer containing the BSDF data.
        buffer: Option<wgpu::Buffer>,
        /// Number of vertices in the buffer.
        count: u32,
    },
    Rotate {
        /// ID of the texture
        id: egui::TextureId,
        /// Rotation angle in radians.
        angle: f32,
    },
}

/// Events used by debugging tools.
#[derive(Debug)]
pub enum DebuggingEvent {
    ToggleDebugDrawing(bool),
    ToggleEmitterPointsDrawing(bool),
    ToggleEmitterRaysDrawing(bool),
    ToggleEmitterSamplesDrawing(bool),
    ToggleRayTrajectoriesDrawing {
        missed: bool,
        reflected: bool,
    },
    ToggleCollectedRaysDrawing(bool),
    MeasureOnce {
        method: RtcMethod,
        params: BsdfMeasurementParams,
        mesh: Handle<MicroSurfaceMesh>,
    },
    UpdateGridCellDrawing {
        pos: IVec2,
        status: bool,
    },
    UpdateCollectorDrawing {
        status: bool,
        patches: DetectorPatches,
    },
    ToggleSamplingRendering(bool),
    UpdateDepthMap,
    UpdateRayParams {
        t: f32,
    },
    UpdateMicroSurface {
        surf: Handle<MicroSurface>,
        mesh: Handle<MicroSurfaceMesh>,
    },
    UpdateEmitterSamples {
        samples: EmitterSamples,
    },
    UpdateEmitterPoints {
        points: MeasurementPoints,
    },
    UpdateEmitterPosition {
        position: Sph2,
    },
    UpdateSurfacePrimitiveId {
        surf: Option<Handle<MicroSurface>>,
        id: u32,
        status: bool,
    },
    ToggleSurfaceNormalDrawing,
    EmitRays,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum OutlinerEvent {
    SelectItem(OutlinerItem),
    RemoveItem(OutlinerItem),
}

#[derive(Debug)]
pub enum MeasureEvent {
    Madf {
        params: AdfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
    Mmsf {
        params: MsfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
    Bsdf {
        params: BsdfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
}

/// Response to an event.
#[derive(Debug)]
pub enum EventResponse {
    /// The event was consumed and should not be propagated.
    Handled,
    /// The event was ignored and should be propagated.
    Ignored(VgonioEvent),
}

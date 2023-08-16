use crate::{
    app::{
        cache::Handle,
        gfx::RenderableMesh,
        gui::{notify::NotifyKind, theme::ThemeKind},
    },
    fitting::{AreaDistributionFittingMode, ReflectionModelFamily},
    measure::{
        collector::CollectorPatches,
        emitter::EmitterSamples,
        measurement::{
            BsdfMeasurementParams, MadfMeasurementParams, MeasurementData, MeasurementKind,
            MmsfMeasurementParams,
        },
        CollectorScheme, RtcMethod,
    },
};
use uuid::Uuid;
use vgcore::{
    math::{IVec2, Vec3},
    units::{Degrees, Radians},
};
use vgsurf::{MicroSurface, MicroSurfaceMesh};

use super::outliner::Item;

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
    #[cfg(feature = "scaled-adf-fitting")]
    Fitting {
        kind: MeasurementKind,
        family: ReflectionModelFamily,
        data: Handle<MeasurementData>,
        mode: Option<AreaDistributionFittingMode>,
        scaled: bool,
    },
    #[cfg(not(feature = "scaled-adf-fitting"))]
    Fitting {
        kind: MeasurementKind,
        family: ReflectionModelFamily,
        data: Handle<MeasurementData>,
        mode: Option<AreaDistributionFittingMode>,
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
    ToggleCollectorDrawing {
        status: bool,
        scheme: CollectorScheme,
        patches: CollectorPatches,
        orbit_radius: f32,
        shape_radius: Option<f32>,
    },
    ToggleEmitterPointsDrawing(bool),
    ToggleEmitterRaysDrawing(bool),
    ToggleEmitterSamplesDrawing(bool),
    ToggleRayTrajectoriesDrawing {
        missed: bool,
        reflected: bool,
    },
    ToggleCollectedRaysDrawing(bool),
    MeasureOnePoint {
        method: RtcMethod,
        params: BsdfMeasurementParams,
        mesh: Handle<MicroSurfaceMesh>,
    },
    UpdateGridCellDrawing {
        pos: IVec2,
        status: bool,
    },
    ToggleSamplingRendering(bool),
    UpdateDepthMap,
    UpdateRayParams {
        t: f32,
        orbit_radius: f32,
        shape_radius: Option<f32>,
    },
    UpdateEmitterSamples {
        samples: EmitterSamples,
        orbit_radius: f32,
        shape_radius: Option<f32>,
    },
    UpdateEmitterPoints {
        points: Vec<Vec3>,
        orbit_radius: f32,
    },
    UpdateEmitterPosition {
        zenith: Radians,
        azimuth: Radians,
        orbit_radius: f32,
        shape_radius: Option<f32>,
    },
    UpdateSurfacePrimitiveId {
        mesh: Option<Handle<RenderableMesh>>,
        id: u32,
        status: bool,
    },
    EmitRays {
        orbit_radius: f32,
        shape_radius: Option<f32>,
    },
}

#[derive(Debug)]
#[non_exhaustive]
pub enum OutlinerEvent {
    SelectItem(Item),
}

#[derive(Debug)]
pub enum MeasureEvent {
    Madf {
        params: MadfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
    Mmsf {
        params: MmsfMeasurementParams,
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

#[cfg(feature = "fitting")]
use crate::fitting::FittingProblemKind;
use crate::{
    app::{
        cache::Handle,
        gui::{
            notify::NotifyKind,
            surf_viewer::{OverlayFlags, ShadingMode},
            theme::ThemeKind,
        },
    },
    io::OutputOptions,
    measure::{
        bsdf::emitter::{EmitterSamples, MeasurementPoints},
        params::MeasurementParams,
        Measurement,
    },
};
use base::{
    math::{IVec2, Sph2},
    partition::SphericalPartition,
    units::Degrees,
    MeasurementKind,
};
use std::path::PathBuf;
use surf::{MicroSurface, MicroSurfaceMesh};
use uuid::Uuid;

use super::outliner::OutlinerItem;

/// Event loop proxy with Vgonio events.
#[derive(Clone, Debug)]
pub struct EventLoopProxy(winit::event_loop::EventLoopProxy<VgonioEvent>);

impl EventLoopProxy {
    /// Send an event to the event loop.
    pub fn send_event(&self, event: VgonioEvent) {
        match self.0.send_event(event) {
            Ok(_) => {}
            Err(err) => {
                log::error!("Failed to send event: {}", err);
            }
        }
    }

    /// Create a new event loop proxy.
    pub fn new(event_loop: &winit::event_loop::EventLoop<VgonioEvent>) -> Self {
        Self(event_loop.create_proxy())
    }
}

/// Events used by Vgonio application.
#[derive(Debug)]
#[non_exhaustive]
pub enum VgonioEvent {
    Quit,
    RequestRedraw,
    OpenFiles(Vec<PathBuf>),
    ToggleSurfaceVisibility,
    CheckVisibleFacets {
        m_azimuth: Degrees,
        m_zenith: Degrees,
        opening_angle: Degrees,
    },
    BsdfViewer(BsdfViewerEvent),
    Debugging(DebuggingEvent),
    Measure {
        /// Parameters of the measurement.
        params: MeasurementParams,
        /// Surfaces to be measured.
        surfaces: Vec<Handle<MicroSurface>>,
        /// Output file format.
        output_opts: Option<OutputOptions>,
    },
    ExportMeasurement(Handle<Measurement>),
    Notify {
        kind: NotifyKind,
        text: String,
        time: f32,
    },
    UpdateThemeKind(ThemeKind),
    SurfaceViewer(SurfaceViewerEvent),
    Outliner(OutlinerEvent),
    Graphing {
        kind: MeasurementKind,
        data: Handle<Measurement>,
        independent: bool,
    },
    #[cfg(feature = "fitting")]
    Fitting {
        kind: FittingProblemKind,
        data: Handle<Measurement>,
        scale: f32,
    },
    // TODO
    SmoothSurface {
        surf: Handle<MicroSurface>,
        lod: u32,
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
    /// Notify the GUI backend that a surface viewer's overlay has been updated.
    UpdateOverlay {
        /// ID of the surface viewer.
        uuid: Uuid,
        /// New overlay flags.
        overlay: OverlayFlags,
    },
    /// Notify the GUI backend that a surface viewer's shading mode has been
    /// updated.
    UpdateShading {
        /// ID of the surface viewer.
        uuid: Uuid,
        /// New shading mode.
        shading: ShadingMode,
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
    ToggleMeasurementPointsDrawing(bool),
    ToggleEmitterRaysDrawing(bool),
    ToggleEmitterSamplesDrawing(bool),
    UpdateRayTrajectoriesDrawing {
        index: usize,
        missed: bool,
        reflected: bool,
    },
    FocusSurfaceViewer(Option<Uuid>),
    UpdateGridCellDrawing {
        pos: IVec2,
        status: bool,
    },

    ToggleCollectedRaysDrawing(bool),
    ToggleDetectorDomeDrawing(bool),
    UpdateDetectorPatches(SphericalPartition),
    ToggleSamplingRendering(bool),
    UpdateDepthMap,
    UpdateRayParams {
        t: f32,
    },
    UpdateMicroSurface {
        surf: Handle<MicroSurface>,
        mesh: Handle<MicroSurfaceMesh>,
    },
    UpdateEmitterSamples(EmitterSamples),
    UpdateMeasurementPoints(MeasurementPoints),
    UpdateEmitterPosition {
        position: Sph2,
    },
    UpdateFocusedSurface(Option<Handle<MicroSurface>>),
    UpdateSurfacePrimitiveId {
        id: u32,
        status: bool,
    },
    EmitRays,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum OutlinerEvent {
    SelectItem(OutlinerItem),
    RemoveItem(OutlinerItem),
}

/// Response to an event.
#[derive(Debug)]
pub enum EventResponse {
    /// The event was consumed and should not be propagated.
    Handled,
    /// The event was ignored and should be propagated.
    Ignored(VgonioEvent),
}

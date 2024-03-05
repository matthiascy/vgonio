mod brdf_viewer;
mod bsdf_viewer;
mod data;
mod docking;
mod event;
mod file_drop;
// mod gizmo;
mod icons;
mod measurement;
mod misc;
mod notify;
pub mod outliner;
mod plotter;
mod prop_inspector;
pub mod state;
mod surf_viewer;
mod theme;
mod tools;
mod ui;
mod visual_grid;
mod widgets;

// TODO: MSAA

use core::slice::SlicePattern;
use std::{
    default::Default,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

pub(crate) use tools::DebuggingInspector;
pub use ui::VgonioGui;

use crate::{
    app::{
        cache::RawCache,
        gui::{
            bsdf_viewer::BsdfViewer,
            event::{BsdfViewerEvent, DebuggingEvent, EventResponse, VgonioEvent},
            theme::ThemeState,
        },
    },
    error::RuntimeError,
    measure, RangeByStepSizeInclusive,
};
use base::{error::VgonioError, input::InputState};
use bxdf::brdf::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel};
use gfxkit::context::{GpuContext, WgpuConfig, WindowSurface};
use winit::{
    dpi::PhysicalSize,
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopWindowTarget},
    keyboard::PhysicalKey,
    window::{Window, WindowBuilder},
};

/// Initial window width.
const WIN_INITIAL_WIDTH: u32 = 1600;
/// Initial window height.
const WIN_INITIAL_HEIGHT: u32 = 900;

use self::tools::SamplingInspector;

use crate::{
    app::{
        cache::Cache,
        cli::BrdfModel,
        gui::{
            docking::WidgetKind,
            event::{EventLoopProxy, SurfaceViewerEvent},
            notify::NotifyKind,
            state::{debug::DebugDrawingState, GuiContext},
            surf_viewer::SurfaceViewerStates,
        },
        Config,
    },
    fitting::mse::compute_iso_brdf_mse,
    io::OutputOptions,
    measure::{
        data::{MeasuredData, MeasurementData},
        params::MeasurementParams,
    },
};

/// Launches Vgonio GUI application.
pub fn run(config: Config) -> Result<(), VgonioError> {
    let event_loop = EventLoopBuilder::<VgonioEvent>::with_user_event()
        .build()
        .unwrap();
    let (window, win_id) = {
        let window = WindowBuilder::new()
            .with_decorations(true)
            .with_resizable(true)
            .with_transparent(false)
            .with_inner_size(PhysicalSize {
                width: WIN_INITIAL_WIDTH,
                height: WIN_INITIAL_HEIGHT,
            })
            .with_title("vgonio")
            .build(&event_loop)
            .unwrap();
        let id = window.id();
        (Arc::new(window), id)
    };

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut vgonio = pollster::block_on(VgonioGuiApp::new(config, window.clone(), &event_loop))?;

    let mut last_frame_time = Instant::now();

    event_loop
        .run(move |event, elwt| {
            let now = Instant::now();
            let dt = now - last_frame_time;
            last_frame_time = now;

            match event {
                Event::UserEvent(event) => vgonio.on_user_event(event, elwt),
                Event::WindowEvent {
                    window_id,
                    ref event,
                } if window_id == win_id => {
                    vgonio.on_window_event(event, elwt);
                }
                Event::AboutToWait => {
                    vgonio.update_and_render(elwt, dt);
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .map_err(|err| VgonioError::new("Failed to run the application", Some(Box::new(err))))
}
// TODO: add MSAA

/// Vgonio client application with GUI.
pub struct VgonioGuiApp {
    /// The time when the application started.
    pub start_time: Instant,
    /// Gui context
    gui_ctx: GuiContext,
    /// Gpu context
    gpu_ctx: Arc<GpuContext>,
    /// The main window of the application.
    window: Arc<Window>,
    /// Surface for presenting rendered frames.
    canvas: WindowSurface,
    /// The GUI application state.
    ui: VgonioGui,
    /// The theme of the application.
    theme: ThemeState,
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,
    /// The cache of the application including preloaded datafiles. See
    /// [`Cache`].
    cache: Cache,
    /// Input states collected from the window.
    input: InputState,
    /// State of the micro surface rendering.
    surface_viewer_states: SurfaceViewerStates,
    /// State of the BSDF viewer, including the pipeline, binding groups, and
    /// buffers.
    bsdf_viewer: Arc<RwLock<BsdfViewer>>,
    /// Debug drawing state.
    dbg_drawing_state: DebugDrawingState,
    /// Event loop proxy for sending events to the application.
    event_loop_proxy: EventLoopProxy,
}

impl VgonioGuiApp {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(
        config: Config,
        window: Arc<Window>,
        event_loop: &EventLoop<VgonioEvent>,
    ) -> Result<Self, VgonioError> {
        let wgpu_config = WgpuConfig {
            device_descriptor: wgpu::DeviceDescriptor {
                label: Some("vgonio-wgpu-device"),
                required_features: wgpu::Features::POLYGON_MODE_LINE
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 256,
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        let event_loop_proxy = EventLoopProxy::new(event_loop);
        let (gpu_ctx, surface) = GpuContext::onscreen(window.clone(), &wgpu_config).await;
        let gpu_ctx = Arc::new(gpu_ctx);
        let canvas = WindowSurface::new(&gpu_ctx, &window, &wgpu_config, surface);
        let gui_ctx = GuiContext::new(
            &window,
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
            canvas.format(),
            1,
        );

        let config = Arc::new(config);
        let cache = {
            let mut inner = RawCache::new(config.cache_dir());
            inner.load_ior_database(&config);
            Cache::from_raw(inner)
        };

        // TODO: remove this
        let bsdf_viewer = Arc::new(RwLock::new(BsdfViewer::new(
            gpu_ctx.clone(),
            gui_ctx.renderer.clone(),
            event_loop_proxy.clone(),
        )));

        let ui = VgonioGui::new(
            EventLoopProxy::new(event_loop),
            config.clone(),
            gpu_ctx.clone(),
            gui_ctx.renderer.clone(),
            // bsdf_viewer.clone(),
            cache.clone(),
        );

        let input = InputState::default();

        let dbg_drawing_state = DebugDrawingState::new(
            &gpu_ctx,
            canvas.format(),
            event_loop_proxy.clone(),
            cache.clone(),
        );

        let surface_viewer_states = SurfaceViewerStates::new(&gpu_ctx, canvas.format());

        Ok(Self {
            start_time: Instant::now(),
            gpu_ctx,
            gui_ctx,
            config,
            ui,
            cache,
            input,
            dbg_drawing_state,
            canvas,
            bsdf_viewer,
            surface_viewer_states,
            theme: Default::default(),
            event_loop_proxy,
            window,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>, scale_factor: Option<f32>) {
        self.canvas.resize(
            &self.gpu_ctx.device,
            new_size.width,
            new_size.height,
            scale_factor,
        );
    }

    pub fn on_window_event(
        &mut self,
        event: &WindowEvent,
        elwt: &EventLoopWindowTarget<VgonioEvent>,
    ) {
        let _ = self.gui_ctx.on_window_event(&self.window, event);
        // Even if the event was consumed by the UI, we still need to update the
        // input state.
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state,
                        ..
                    },
                ..
            } => {
                self.input.update_key_map(*keycode, *state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.input.update_scroll_delta(*delta);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.update_mouse_map(*button, *state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.update_cursor_delta((*position).cast::<f32>());
            }
            WindowEvent::Resized(new_size) => {
                self.resize(*new_size, Some(self.window.scale_factor() as f32));
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.resize(self.window.inner_size(), Some(*scale_factor as f32));
            }
            WindowEvent::CloseRequested => elwt.exit(),
            _ => {}
        }
    }

    pub fn render_frame(&mut self, window: &Window, dt: Duration) -> Result<(), RuntimeError> {
        // Update the uniform buffers for the debug drawing state.
        if self.dbg_drawing_state.enabled && self.dbg_drawing_state.output_viewer.is_some() {
            let (lowest, highest, scale) = match self.dbg_drawing_state.microsurface {
                Some((surface, _)) => {
                    let properties = self.ui.properties.read().unwrap();
                    let prop = properties.surfaces.get(&surface).unwrap();
                    (prop.min, prop.max, prop.scale)
                }
                None => (0.0, 1.0, 1.0),
            };
            let viewer = self.dbg_drawing_state.output_viewer.unwrap();
            let view_proj = self
                .surface_viewer_states
                .viewer_state(viewer)
                .unwrap()
                .camera_view_proj();
            self.dbg_drawing_state.update_uniform_buffer(
                &self.gpu_ctx,
                &(view_proj.proj * view_proj.view),
                lowest,
                highest,
                scale,
            );
        }

        // Update GUI context.
        self.gui_ctx.update(window);

        // Update the renderings.
        self.render(window, dt)?;

        // Rendering the SamplingInspector is done after the main render pass.
        if self.dbg_drawing_state.sampling_debug_enabled {
            self.ui
                .tools
                .get_tool_mut::<SamplingInspector>()
                .unwrap()
                .render();
        }

        self.bsdf_viewer.write().unwrap().render();

        // Reset the input state. (mouse delta, scroll delta)
        self.input.reset();

        Ok(())
    }

    /// Render the frame to the surface.
    pub fn render(&mut self, window: &Window, dt: Duration) -> Result<(), wgpu::SurfaceError> {
        // Get the next frame (`SurfaceTexture`) to render to.
        let output_frame = self.canvas.get_current_texture()?;
        // Get a `TextureView` to the output frame's color attachment.
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (viewer_encoder, dbg_encoder) = {
            // Render first viewports for the surface viewers.
            let properties = self.ui.properties.read().unwrap();
            let surfaces = properties.visible_surfaces_with_props();
            let viewer_encoder = {
                let active_viewer = {
                    let viewers = self.ui.dock_space.surface_viewers();
                    if viewers.len() == 1 {
                        Some(viewers[0])
                    } else {
                        self.ui.dock_space.find_active_focused().and_then(|widget| {
                            match widget.1.dockable.kind() {
                                WidgetKind::SurfViewer => Some(widget.1.dockable.uuid()),
                                _ => None,
                            }
                        })
                    }
                };

                self.cache.read(|cache| {
                    self.surface_viewer_states.record_render_pass(
                        &self.gpu_ctx,
                        active_viewer,
                        &self.input,
                        dt,
                        &self.theme,
                        cache,
                        &surfaces,
                    )
                })
            };

            let dbg_drawing_encoder = match self.dbg_drawing_state.output_viewer {
                Some(viewer) => self.dbg_drawing_state.record_render_pass(
                    &self.gpu_ctx,
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self
                            .surface_viewer_states
                            .viewer_state(viewer)
                            .unwrap()
                            .colour_attachment()
                            .view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self
                            .surface_viewer_states
                            .viewer_state(viewer)
                            .unwrap()
                            .depth_attachment()
                            .view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                ),
                None => None,
            };
            (viewer_encoder, dbg_drawing_encoder)
        };

        // UI render pass recoding.
        let ui_render_output = self.gui_ctx.render(
            window,
            self.canvas.screen_descriptor(),
            &output_view,
            |ctx| {
                self.theme.update_context(ctx);
                self.ui.show(ctx, self.theme.kind());
            },
        );

        let cmds: Box<dyn Iterator<Item = wgpu::CommandBuffer>> = match dbg_encoder {
            Some(encoder) => Box::new(
                [viewer_encoder.finish(), encoder.finish()]
                    .into_iter()
                    .chain(ui_render_output.user_cmds)
                    .chain([ui_render_output.ui_cmd]),
            ),
            None => Box::new(
                std::iter::once(viewer_encoder.finish())
                    .chain(ui_render_output.user_cmds)
                    .chain([ui_render_output.ui_cmd]),
            ),
        };

        // Submit the command buffers to the GPU: first the user's command buffers, then
        // the main render pass, and finally the UI render pass.
        self.gpu_ctx.queue.submit(cmds);

        // Present the frame to the screen.
        output_frame.present();

        Ok(())
    }

    pub fn on_user_event(&mut self, event: VgonioEvent, elwt: &EventLoopWindowTarget<VgonioEvent>) {
        use VgonioEvent::*;

        // Handle events from the UI.
        match self.ui.on_user_event(event) {
            EventResponse::Handled => (),
            EventResponse::Ignored(event) => {
                match event {
                    Quit => {
                        elwt.exit();
                    }
                    RequestRedraw => {}
                    Debugging(event) => {
                        // TODO: handle events inside the DebuggingState.
                        match event {
                            DebuggingEvent::ToggleSamplingRendering(enabled) => {
                                self.dbg_drawing_state.sampling_debug_enabled = enabled;
                            }
                            DebuggingEvent::UpdateDepthMap => {
                                // self.depth_map.copy_to_buffer(
                                //     &self.gpu_ctx,
                                //     self.canvas.width(),
                                //     self.canvas.height(),
                                // );
                                // self.ui
                                //     .tools
                                //     .get_tool_mut::<DebuggingInspector>()
                                //     .unwrap()
                                //     .depth_map_pane
                                //     .update_depth_map(
                                //         &self.gpu_ctx,
                                //         &self.gui_ctx,
                                //         &self.depth_map.
                                // depth_attachment_storage,
                                //         self.depth_map.width,
                                //         self.canvas.height(),
                                //     );
                            }
                            DebuggingEvent::FocusSurfaceViewer(viewer) => {
                                self.dbg_drawing_state.output_viewer = viewer;
                            }
                            DebuggingEvent::UpdateEmitterSamples(samples) => {
                                log::trace!("Updating emitter samples: {:?}", samples.len());
                                self.dbg_drawing_state
                                    .update_emitter_samples(&self.gpu_ctx, samples);
                            }
                            DebuggingEvent::UpdateMeasurementPoints(points) => {
                                log::trace!("Updating measurement points: {:?}", points.len());
                                self.dbg_drawing_state
                                    .update_measurement_points(&self.gpu_ctx, points);
                            }
                            DebuggingEvent::ToggleMeasurementPointsDrawing(status) => {
                                log::trace!("Toggling measurement points drawing: {:?}", status);
                                self.dbg_drawing_state.emitter_points_drawing = status;
                            }
                            DebuggingEvent::UpdateEmitterPosition { position } => {
                                log::trace!("Updating emitter position: {:?}", position);
                                self.dbg_drawing_state
                                    .update_emitter_position(&self.gpu_ctx, position);
                            }
                            DebuggingEvent::EmitRays => {
                                log::trace!("Emitting rays");
                                self.dbg_drawing_state.emit_rays(&self.gpu_ctx);
                            }
                            DebuggingEvent::ToggleDebugDrawing(status) => {
                                log::trace!("Toggling debug drawing: {:?}", status);
                                self.dbg_drawing_state.enabled = status;
                            }
                            DebuggingEvent::ToggleDetectorDomeDrawing(status) => {
                                log::trace!("Toggling detector dome drawing: {:?}", status);
                                self.dbg_drawing_state.detector_dome_drawing = status;
                            }
                            DebuggingEvent::UpdateDetectorPatches(patches) => {
                                log::trace!(
                                    "Updating detector patches: {:?}",
                                    patches.num_patches()
                                );
                                self.dbg_drawing_state
                                    .update_detector_drawing(&self.gpu_ctx, patches);
                            }
                            DebuggingEvent::UpdateRayParams { t } => {
                                log::trace!("Updating ray params: {t}");
                                self.dbg_drawing_state.update_ray_params(&self.gpu_ctx, t);
                            }
                            DebuggingEvent::ToggleEmitterRaysDrawing(status) => {
                                log::trace!("Toggling emitter rays drawing: {:?}", status);
                                self.dbg_drawing_state.emitter_rays_drawing = status;
                            }
                            DebuggingEvent::ToggleEmitterSamplesDrawing(status) => {
                                log::trace!("Toggling emitter samples drawing: {:?}", status);
                                self.dbg_drawing_state.emitter_samples_drawing = status;
                            }
                            DebuggingEvent::UpdateRayTrajectoriesDrawing {
                                index,
                                missed,
                                reflected,
                            } => {
                                self.dbg_drawing_state.measurement_point_index =
                                    index % self.dbg_drawing_state.measurement_point_index_max;
                                log::debug!(
                                    "Updating ray trajectories drawing: {} - missed {:?}, \
                                     reflected {:?}",
                                    self.dbg_drawing_state.measurement_point_index,
                                    missed,
                                    reflected
                                );
                                self.dbg_drawing_state.ray_trajectories_drawing_reflected =
                                    reflected;
                                self.dbg_drawing_state.ray_trajectories_drawing_missed = missed;
                            }
                            DebuggingEvent::ToggleCollectedRaysDrawing(status) => {
                                self.dbg_drawing_state.detector_ray_hit_points_drawing = status;
                            }
                            DebuggingEvent::UpdateSurfacePrimitiveId { id, status } => {
                                log::trace!("Updating surface primitive id: {:?}", id);
                                self.dbg_drawing_state
                                    .update_surface_primitive_id(id, status);
                            }
                            DebuggingEvent::UpdateFocusedSurface(surf) => {
                                log::trace!("Updating focused surface: {:?}", surf);
                                self.dbg_drawing_state.update_focused_surface(surf);
                            }
                            DebuggingEvent::UpdateGridCellDrawing { .. } => {
                                todo!("UpdateGridCellDrawing")
                            }
                            DebuggingEvent::UpdateMicroSurface { surf, mesh } => {
                                self.dbg_drawing_state.microsurface = Some((surf, mesh));
                            }
                        }
                    }
                    BsdfViewer(event) => match event {
                        BsdfViewerEvent::ToggleView(id) => {
                            self.bsdf_viewer.write().unwrap().toggle_view(id);
                        }
                        BsdfViewerEvent::UpdateBuffer { id, buffer, count } => {
                            log::debug!("Updating buffer for id: {:?}", id);
                            self.bsdf_viewer
                                .write()
                                .unwrap()
                                .update_bsdf_data_buffer(id, buffer, count);
                        }
                        BsdfViewerEvent::Rotate { id, angle } => {
                            log::debug!("Rotating bsdf viewer id: {:?}", id);
                            self.bsdf_viewer.write().unwrap().rotate(id, angle);
                        }
                    },
                    Measure {
                        params,
                        surfaces,
                        output_opts,
                    } => {
                        let data = match params {
                            MeasurementParams::Adf(params) => self.cache.read(|cache| {
                                measure::microfacet::measure_area_distribution(
                                    params, &surfaces, cache,
                                )
                            }),
                            MeasurementParams::Msf(params) => self.cache.read(|cache| {
                                measure::microfacet::measure_masking_shadowing(
                                    params, &surfaces, cache,
                                )
                            }),
                            MeasurementParams::Bsdf(params) => {
                                if params.is_both_air_medium() {
                                    log::error!("Cannot measure BSDF for both air medium");
                                    self.event_loop_proxy.send_event(Notify {
                                        kind: NotifyKind::Error,
                                        text: "Cannot measure BSDF for both air medium".to_string(),
                                        time: 2.0,
                                    });
                                    return;
                                }

                                #[cfg(feature = "visu-dbg")]
                                {
                                    let measured = self.cache.read(|cache| {
                                        measure::bsdf::measure_bsdf_rt(
                                            params,
                                            &surfaces,
                                            params.sim_kind,
                                            cache,
                                        )
                                    });
                                    self.dbg_drawing_state.update_ray_trajectories(
                                        &self.gpu_ctx,
                                        &measured[0]
                                            .measured
                                            .as_bsdf()
                                            .as_ref()
                                            .unwrap()
                                            .trajectories(),
                                    );
                                    self.dbg_drawing_state.update_ray_hit_points(
                                        &self.gpu_ctx,
                                        &measured[0]
                                            .measured
                                            .as_bsdf()
                                            .as_ref()
                                            .unwrap()
                                            .hit_points(),
                                    );
                                    measured
                                }

                                #[cfg(not(feature = "visu-dbg"))]
                                self.cache.read(|cache| {
                                    measure::bsdf::measure_bsdf_rt(
                                        params,
                                        &surfaces,
                                        params.sim_kind,
                                        cache,
                                    )
                                })
                            }
                            MeasurementParams::Sdf(params) => self.cache.read(|cache| {
                                measure::microfacet::measure_slope_distribution(
                                    &surfaces, params, cache,
                                )
                            }),
                        };
                        if let Some(opts @ OutputOptions { .. }) = output_opts {
                            crate::io::write_measured_data_to_file(
                                &data,
                                &surfaces,
                                &self.cache,
                                &self.config,
                                opts,
                            )
                            .map_err(|err| {
                                log::error!("Failed to write measured data to file: {}", err);
                            })
                            .unwrap();
                        }
                        self.cache.write(|cache| {
                            let meas = Vec::from(data)
                                .into_iter()
                                .map(|measured| {
                                    cache.add_micro_surface_measurement(measured).unwrap()
                                })
                                .collect::<Vec<_>>();
                            let mut properties = self.ui.properties.write().unwrap();
                            properties.update_measurement_data(&meas, cache);
                        });
                    }
                    SurfaceViewer(event) => match event {
                        SurfaceViewerEvent::Create {
                            uuid,
                            tex_id: texture_id,
                        } => {
                            self.surface_viewer_states.allocate_viewer_resources(
                                uuid,
                                texture_id,
                                &self.gpu_ctx,
                                &self.gui_ctx.renderer,
                            );
                        }
                        SurfaceViewerEvent::Resize { uuid, size } => {
                            self.surface_viewer_states.resize_viewport(
                                uuid,
                                size.0,
                                size.1,
                                &self.gpu_ctx,
                                &self.gui_ctx.renderer,
                            );
                        }
                        SurfaceViewerEvent::Close { .. } => {
                            todo!("SurfaceViewerEvent::Close")
                        }
                        SurfaceViewerEvent::UpdateSurfaceList { surfaces } => {
                            self.surface_viewer_states.update_surfaces_list(&surfaces)
                        }
                        SurfaceViewerEvent::UpdateOverlay { uuid, overlay } => {
                            self.surface_viewer_states.update_overlay(uuid, overlay)
                        }
                        SurfaceViewerEvent::UpdateShading { uuid, shading } => {
                            self.surface_viewer_states.update_shading(uuid, shading)
                        }
                    },
                    UpdateThemeKind(kind) => {
                        self.theme.set_theme_kind(kind);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Update the state of the application then render the current frame.
    pub fn update_and_render(&mut self, elwt: &EventLoopWindowTarget<VgonioEvent>, dt: Duration) {
        let window = self.window.clone();
        match self.render_frame(&window, dt) {
            Ok(_) => {}
            Err(RuntimeError::Rhi(error)) => {
                if error.is_surface_error() {
                    if let Some(surface_error) = error.get::<wgpu::SurfaceError>() {
                        match surface_error {
                            // Reconfigure the surface if lost
                            wgpu::SurfaceError::Lost => {
                                self.canvas.reconfigure(&self.gpu_ctx.device)
                            }
                            // The system is out of memory, we should quit
                            wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                            // All other errors (Outdated, Timeout) should be resolved by the next
                            // frame
                            error => eprintln!("{error:?}"),
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

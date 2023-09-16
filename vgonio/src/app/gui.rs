mod brdf_viewer;
mod bsdf_viewer;
mod data;
mod docking;
mod event;
mod file_drop;
mod gizmo;
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

use std::{
    default::Default,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use uuid::Uuid;

pub(crate) use tools::DebuggingInspector;
pub use ui::VgonioGui;

use crate::{
    app::{
        cache::InnerCache,
        gfx::{GpuContext, WgpuConfig},
        gui::{
            bsdf_viewer::BsdfViewer,
            event::{BsdfViewerEvent, DebuggingEvent, EventResponse, MeasureEvent, VgonioEvent},
            state::{GuiContext, InputState},
            theme::ThemeState,
        },
    },
    error::RuntimeError,
    measure,
};
use vgcore::error::VgonioError;
use winit::{
    dpi::PhysicalSize,
    event::{Event, KeyboardInput, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder},
    window::{Window, WindowBuilder},
};

/// Initial window width.
const WIN_INITIAL_WIDTH: u32 = 1600;
/// Initial window height.
const WIN_INITIAL_HEIGHT: u32 = 900;

use self::tools::SamplingInspector;

use crate::app::{
    gfx::WindowSurface,
    gui::{
        docking::WidgetKind, event::SurfaceViewerEvent, state::debug::DebugDrawingState,
        surf_viewer::SurfaceViewerStates,
    },
    Config,
};

/// Launches Vgonio GUI application.
pub fn run(config: Config) -> Result<(), VgonioError> {
    let event_loop = EventLoopBuilder::<VgonioEvent>::with_user_event().build();
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

    let mut vgonio = pollster::block_on(VgonioGuiApp::new(config, &window, &event_loop))?;

    let mut last_frame_time = Instant::now();

    event_loop.run(move |event, _, control| {
        let now = Instant::now();
        let dt = now - last_frame_time;
        last_frame_time = now;

        match event {
            Event::UserEvent(event) => vgonio.on_user_event(event, control),
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                vgonio.on_window_event(&window, event, control);
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                vgonio.on_redraw_requested(&window, dt, control);
            }

            Event::MainEventsCleared => window.request_redraw(),

            _ => {}
        }
    })
}

/// Vgonio application context.
/// Contains all the resources needed for rendering.
pub struct Context {
    /// GPU context for rendering.
    gpu: Arc<GpuContext>,
    /// GUI context for rendering.
    gui: GuiContext,
}

// TODO: add MSAA

/// Vgonio client application with GUI.
pub struct VgonioGuiApp {
    /// The time when the application started.
    pub start_time: Instant,
    /// The application context.
    ctx: Context,
    /// Surface for presenting rendered frames.
    canvas: WindowSurface,
    /// The GUI application state.
    ui: VgonioGui,
    theme: ThemeState,
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,
    /// The cache of the application including preloaded datafiles. See
    /// [`VgonioCache`].
    cache: Arc<RwLock<InnerCache>>,
    /// Input states collected from the window.
    input: InputState,
    /// State of the micro surface rendering.
    surface_viewer_states: SurfaceViewerStates,
    /// State of the BSDF viewer, including the pipeline, binding groups, and
    /// buffers.
    bsdf_viewer: Arc<RwLock<BsdfViewer>>,
    /// Debug drawing state.
    dbg_drawing_state: DebugDrawingState,
}

impl VgonioGuiApp {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(
        config: Config,
        window: &Window,
        event_loop: &EventLoop<VgonioEvent>,
    ) -> Result<Self, VgonioError> {
        let wgpu_config = WgpuConfig {
            device_descriptor: wgpu::DeviceDescriptor {
                label: Some("vgonio-wgpu-device"),
                features: wgpu::Features::POLYGON_MODE_LINE
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::PUSH_CONSTANTS,
                limits: wgpu::Limits {
                    max_push_constant_size: 256,
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        let (gpu_ctx, surface) = GpuContext::new(window, &wgpu_config).await;
        let gpu_ctx = Arc::new(gpu_ctx);
        let canvas = WindowSurface::new(&gpu_ctx, window, &wgpu_config, surface);
        let gui_ctx = GuiContext::new(
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
            canvas.format(),
            event_loop,
            1,
        );

        let config = Arc::new(config);
        let cache = {
            let mut _cache = InnerCache::new(config.cache_dir());
            _cache.load_ior_database(&config);
            Arc::new(RwLock::new(_cache))
        };

        let bsdf_viewer = Arc::new(RwLock::new(BsdfViewer::new(
            gpu_ctx.clone(),
            gui_ctx.renderer.clone(),
            event_loop.create_proxy(),
        )));

        let ui = VgonioGui::new(
            event_loop.create_proxy(),
            config.clone(),
            gpu_ctx.clone(),
            gui_ctx.renderer.clone(),
            // bsdf_viewer.clone(),
            cache.clone(),
        );

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        let dbg_drawing_state = DebugDrawingState::new(
            &gpu_ctx,
            canvas.format(),
            event_loop.create_proxy(),
            cache.clone(),
        );

        let surface_viewer_states = SurfaceViewerStates::new(&gpu_ctx, canvas.format());

        Ok(Self {
            start_time: Instant::now(),
            ctx: Context {
                gpu: gpu_ctx,
                gui: gui_ctx,
            },
            config,
            ui,
            cache,
            input,
            dbg_drawing_state,
            canvas,
            bsdf_viewer,
            surface_viewer_states,
            theme: Default::default(),
        })
    }

    pub fn reconfigure_surface(&mut self) { self.canvas.reconfigure(&self.ctx.gpu.device); }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>, scale_factor: Option<f32>) {
        self.canvas.resize(
            &self.ctx.gpu.device,
            new_size.width,
            new_size.height,
            scale_factor,
        );
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
        control: &mut ControlFlow,
    ) {
        let _ = self.ctx.gui.on_window_event(event);
        // Even if the event was consumed by the UI, we still need to update the
        // input state.
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
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
                self.resize(*new_size, Some(window.scale_factor() as f32));
            }
            WindowEvent::ScaleFactorChanged {
                new_inner_size,
                scale_factor,
            } => {
                self.resize(**new_inner_size, Some(*scale_factor as f32));
            }
            WindowEvent::CloseRequested => *control = ControlFlow::Exit,
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
                &self.ctx.gpu,
                &(view_proj.proj * view_proj.view),
                lowest,
                highest,
                scale,
            );
        }

        // Update GUI context.
        self.ctx.gui.update(window);

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

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];

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
            let cache = self.cache.read().unwrap();
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

                self.surface_viewer_states.record_render_pass(
                    &self.ctx.gpu,
                    active_viewer,
                    &self.input,
                    dt,
                    &self.theme,
                    &cache,
                    &surfaces,
                )
            };

            let dbg_drawing_encoder = match self.dbg_drawing_state.output_viewer {
                Some(viewer) => self.dbg_drawing_state.record_render_pass(
                    &self.ctx.gpu,
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
                            store: true,
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
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                ),
                None => None,
            };
            (viewer_encoder, dbg_drawing_encoder)
        };

        // UI render pass recoding.
        let ui_render_output = self.ctx.gui.render(
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
        self.ctx.gpu.queue.submit(cmds);

        // Present the frame to the screen.
        output_frame.present();

        Ok(())
    }

    pub fn on_user_event(&mut self, event: VgonioEvent, control_flow: &mut ControlFlow) {
        use VgonioEvent::*;

        // Handle events from the UI.
        match self.ui.on_user_event(event) {
            EventResponse::Handled => return,
            EventResponse::Ignored(event) => {
                match event {
                    Quit => {
                        *control_flow = ControlFlow::Exit;
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
                                //     &self.ctx.gpu,
                                //     self.canvas.width(),
                                //     self.canvas.height(),
                                // );
                                // self.ui
                                //     .tools
                                //     .get_tool_mut::<DebuggingInspector>()
                                //     .unwrap()
                                //     .depth_map_pane
                                //     .update_depth_map(
                                //         &self.ctx.gpu,
                                //         &self.ctx.gui,
                                //         &self.depth_map.
                                // depth_attachment_storage,
                                //         self.depth_map.width,
                                //         self.canvas.height(),
                                //     );
                            }
                            DebuggingEvent::FocusSurfaceViewer(viewer) => {
                                self.dbg_drawing_state.output_viewer = viewer;
                            }
                            DebuggingEvent::UpdateEmitterSamples { samples } => {
                                self.dbg_drawing_state
                                    .update_emitter_samples(&self.ctx.gpu, samples);
                            }
                            DebuggingEvent::UpdateEmitterPoints { points } => {
                                self.dbg_drawing_state
                                    .update_measurement_points(&self.ctx.gpu, points);
                            }
                            DebuggingEvent::ToggleEmitterPointsDrawing(status) => {
                                self.dbg_drawing_state.emitter_points_drawing = status;
                            }
                            DebuggingEvent::UpdateEmitterPosition { position } => {
                                self.dbg_drawing_state
                                    .update_emitter_position(&self.ctx.gpu, position);
                            }
                            DebuggingEvent::EmitRays => {
                                self.dbg_drawing_state.emit_rays(&self.ctx.gpu);
                            }
                            DebuggingEvent::ToggleDebugDrawing(status) => {
                                log::trace!("Toggling debug drawing: {:?}", status);
                                self.dbg_drawing_state.enabled = status;
                            }
                            DebuggingEvent::UpdateCollectorDrawing { status, patches } => self
                                .dbg_drawing_state
                                .update_detector_drawing(&self.ctx.gpu, status, patches),
                            DebuggingEvent::UpdateRayParams { t } => {
                                self.dbg_drawing_state.update_ray_params(&self.ctx.gpu, t);
                            }
                            DebuggingEvent::ToggleEmitterRaysDrawing(status) => {
                                self.dbg_drawing_state.emitter_rays_drawing = status;
                            }
                            DebuggingEvent::ToggleEmitterSamplesDrawing(status) => {
                                self.dbg_drawing_state.emitter_samples_drawing = status;
                            }
                            DebuggingEvent::MeasureOnce {
                                method,
                                params,
                                mesh,
                            } => self.dbg_drawing_state.update_ray_trajectories(
                                &self.ctx.gpu,
                                method,
                                params,
                                mesh,
                            ),
                            DebuggingEvent::ToggleRayTrajectoriesDrawing { missed, reflected } => {
                                self.dbg_drawing_state.ray_trajectories_drawing_reflected =
                                    reflected;
                                self.dbg_drawing_state.ray_trajectories_drawing_missed = missed;
                            }
                            DebuggingEvent::ToggleCollectedRaysDrawing(status) => {
                                self.dbg_drawing_state.detector_ray_hit_points_drawing = status;
                            }
                            DebuggingEvent::UpdateSurfacePrimitiveId { surf, id, status } => {
                                log::debug!("Updating surface {:?}'s primitive id: {:?}", surf, id);
                                self.dbg_drawing_state
                                    .update_surface_primitive_id(surf, id, status);
                            }
                            DebuggingEvent::ToggleSurfaceNormalDrawing => {
                                self.dbg_drawing_state.toggle_surface_normals();
                            }
                            DebuggingEvent::UpdateGridCellDrawing { .. } => {
                                todo!("UpdateGridCellDrawing")
                            }
                            DebuggingEvent::UpdateMicroSurface { surf, mesh } => {
                                self.dbg_drawing_state.microsurface = Some((surf, mesh));
                                self.dbg_drawing_state
                                    .prepare_surface_normals_buffer(&self.ctx.gpu);
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
                    Measure(event) => match event {
                        MeasureEvent::Madf { params, surfaces } => {
                            println!("Measuring area distribution");
                            let measured = measure::microfacet::measure_area_distribution(
                                params,
                                &surfaces,
                                &self.cache.read().unwrap(),
                            );
                            {
                                let mut cache = self.cache.write().unwrap();
                                for meas in measured {
                                    cache.add_micro_surface_measurement(meas).unwrap();
                                }
                                todo!("Show the measurement in the outliner");
                            }
                        }
                        MeasureEvent::Mmsf { params, surfaces } => {
                            println!("Measuring masking/shadowing");
                            measure::microfacet::measure_masking_shadowing(
                                params,
                                &surfaces,
                                &self.cache.read().unwrap(),
                            );
                            todo!("Save area distribution to file or display it in a window");
                        }
                        MeasureEvent::Bsdf { params, surfaces } => {
                            println!("Measuring BSDF");
                            measure::bsdf::measure_bsdf_rt(
                                params,
                                &surfaces,
                                params.sim_kind,
                                &self.cache.read().unwrap(),
                            );
                            todo!("Save area distribution to file or display it in a window");
                        }
                    },
                    SurfaceViewer(event) => match event {
                        SurfaceViewerEvent::Create {
                            uuid,
                            tex_id: texture_id,
                        } => {
                            self.surface_viewer_states.allocate_viewer_resources(
                                uuid,
                                texture_id,
                                &self.ctx.gpu,
                                &self.ctx.gui.renderer,
                            );
                        }
                        SurfaceViewerEvent::Resize { uuid, size } => {
                            self.surface_viewer_states.resize_viewport(
                                uuid,
                                size.0,
                                size.1,
                                &self.ctx.gpu,
                                &self.ctx.gui.renderer,
                            );
                        }
                        SurfaceViewerEvent::Close { .. } => {
                            todo!("SurfaceViewerEvent::Close")
                        }
                        SurfaceViewerEvent::UpdateSurfaceList { surfaces } => {
                            self.surface_viewer_states.update_surfaces_list(&surfaces)
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
    pub fn on_redraw_requested(
        &mut self,
        window: &Window,
        dt: Duration,
        control: &mut ControlFlow,
    ) {
        match self.render_frame(window, dt) {
            Ok(_) => {}
            Err(RuntimeError::Rhi(error)) => {
                if error.is_surface_error() {
                    if let Some(surface_error) = error.get::<wgpu::SurfaceError>() {
                        match surface_error {
                            // Reconfigure the surface if lost
                            wgpu::SurfaceError::Lost => self.reconfigure_surface(),
                            // The system is out of memory, we should quit
                            wgpu::SurfaceError::OutOfMemory => *control = ControlFlow::Exit,
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

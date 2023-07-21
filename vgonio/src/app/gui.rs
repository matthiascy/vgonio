mod brdf_viewer;
mod bsdf_viewer;
mod docking;
mod event;
mod file_drop;
mod gizmo;
mod icons;
mod misc;
pub mod outliner;
mod simulations;
pub mod state;
mod surf_viewer;
mod theme;
mod tools;
mod ui;
mod visual_grid;
mod widgets;

// TODO: MSAA

use egui::Align2;
use egui_toast::{Toast, ToastKind, ToastOptions, Toasts};
use std::{
    default::Default,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

pub(crate) use tools::DebuggingInspector;
pub use ui::VgonioGui;

use crate::{
    app::{
        cache::Cache,
        gfx::{
            camera::{Camera, Projection, ProjectionKind},
            GpuContext, WgpuConfig,
        },
        gui::{
            bsdf_viewer::BsdfViewer,
            event::{BsdfViewerEvent, DebuggingEvent, MeasureEvent, VgonioEvent},
            state::{camera::CameraState, DebugDrawingState, DepthMap, GuiContext, InputState},
            theme::ThemeState,
        },
    },
    error::RuntimeError,
    measure::{
        self,
        collector::CollectorPatches,
        emitter::EmitterSamples,
        measurement::{BsdfMeasurementParams, MadfMeasurementParams, MmsfMeasurementParams},
        CollectorScheme, RtcMethod,
    },
};
use vgcore::{
    error::VgonioError,
    math::{Handedness, IVec2, Mat4, Vec3},
    units::{Degrees, Radians},
};
use vgsurf::{MicroSurface, MicroSurfaceMesh};
use winit::{
    dpi::PhysicalSize,
    event::{Event, KeyboardInput, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy},
    window::{Window, WindowBuilder},
};

/// Initial window width.
const WIN_INITIAL_WIDTH: u32 = 1600;
/// Initial window height.
const WIN_INITIAL_HEIGHT: u32 = 900;

use self::{
    surf_viewer::{MicroSurfaceState, MicroSurfaceUniforms},
    tools::{PlottingWidget, SamplingInspector},
    visual_grid::VisualGridState,
};

use crate::app::{gfx::WindowSurface, Config};

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
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,
    /// The cache of the application including preloaded datafiles. See
    /// [`VgonioCache`].
    cache: Arc<RwLock<Cache>>,
    /// Input states collected from the window.
    input: InputState,
    /// Camera state including the view and projection matrices.
    camera: CameraState,
    /// State of the micro surface rendering, including the pipeline, binding
    /// groups, and buffers.
    msurf_rdr_state: MicroSurfaceState,
    /// State of the visual grid rendering, including the pipeline, binding
    /// groups, and buffers.
    visual_grid_state: VisualGridState,
    /// State of the BSDF viewer, including the pipeline, binding groups, and
    /// buffers.
    bsdf_viewer: Arc<RwLock<BsdfViewer>>,

    theme: ThemeState,

    /// Depth map of the scene. TODO: refactor
    depth_map: DepthMap,
    // TODO: add MSAA
    /// Debug drawing state.
    dbg_drawing_state: DebugDrawingState,

    /// Notification manager.
    toasts: Toasts,
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
        let depth_map = DepthMap::new(&gpu_ctx, canvas.width(), canvas.height());
        let camera = {
            let camera = Camera::new(Vec3::new(0.0, 4.0, 10.0), Vec3::ZERO, Vec3::Y);
            let projection = Projection::new(
                0.1,
                100.0,
                75.0f32.to_radians(),
                canvas.width(),
                canvas.height(),
            );
            CameraState::new(camera, projection, ProjectionKind::Perspective)
        };

        let visual_grid_state = VisualGridState::new(&gpu_ctx, canvas.format());

        let gui_ctx = GuiContext::new(
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
            canvas.format(),
            event_loop,
            1,
        );

        let config = Arc::new(config);
        let cache = {
            let mut _cache = Cache::new(config.cache_dir());
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
            bsdf_viewer.clone(),
            cache.clone(),
        );

        let theme = ThemeState::default();

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
        let msurf_rdr_state = MicroSurfaceState::new(&gpu_ctx, canvas.format());

        let toasts = Toasts::new()
            .anchor(Align2::LEFT_BOTTOM, (10.0, -10.0))
            .direction(egui::Direction::BottomUp);

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
            depth_map,
            dbg_drawing_state,
            camera,
            canvas,
            msurf_rdr_state,
            visual_grid_state,
            bsdf_viewer,
            toasts,
            theme,
        })
    }

    #[inline]
    pub fn surface_width(&self) -> u32 { self.canvas.width() }

    #[inline]
    pub fn surface_height(&self) -> u32 { self.canvas.height() }

    pub fn reconfigure_surface(&mut self) { self.canvas.reconfigure(&self.ctx.gpu.device); }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>, scale_factor: Option<f32>) {
        if self.canvas.resize(
            &self.ctx.gpu.device,
            new_size.width,
            new_size.height,
            scale_factor,
        ) {
            self.depth_map
                .resize(&self.ctx.gpu, new_size.width, new_size.height);
            self.camera
                .projection
                .resize(new_size.width, new_size.height);
        }
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
        control: &mut ControlFlow,
    ) {
        let response = self.ctx.gui.on_window_event(event);
        // Note: `consumed` is `true` if the event was handled by egui.
        if !response.consumed {
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
    }

    pub fn update(&mut self, window: &Window, dt: Duration) -> Result<(), RuntimeError> {
        // Update camera uniform.
        self.camera
            .update(&self.input, dt, ProjectionKind::Perspective);
        self.ui.update_gizmo_matrices(
            Mat4::IDENTITY,
            Mat4::look_at_rh(self.camera.camera.eye, Vec3::ZERO, self.camera.camera.up),
            Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
        );

        let view_proj = self.camera.uniform.view_proj;
        let view_proj_inv = self.camera.uniform.view_proj_inv;
        self.visual_grid_state.update_uniforms(
            &self.ctx.gpu,
            &view_proj,
            &view_proj_inv,
            self.theme.visuals().grid_line_color,
            self.theme.kind(),
        );
        let dbg_tool = self.ui.tools.get_tool::<DebuggingInspector>().unwrap();
        let (lowest, highest, scale) = match dbg_tool.brdf_debugging.selected_surface() {
            Some(surface) => {
                let state = self.ui.outliner().surfaces().get(&surface).unwrap();
                (state.1.min, state.1.max, state.1.scale)
            }
            None => (0.0, 1.0, 1.0),
        };
        self.dbg_drawing_state.update_uniform_buffer(
            &self.ctx.gpu,
            &(view_proj.proj * view_proj.view),
            lowest,
            highest,
            scale,
        );

        // Update uniform buffer for all visible surfaces.
        let visible_surfaces = self.ui.outliner().visible_surfaces();
        if !visible_surfaces.is_empty() {
            self.ctx.gpu.queue.write_buffer(
                &self.msurf_rdr_state.global_uniform_buffer,
                0,
                bytemuck::bytes_of(&view_proj),
            );
            // Update per-surface uniform buffer.
            let aligned_size = MicroSurfaceUniforms::aligned_size(&self.ctx.gpu.device);
            for (hdl, state) in visible_surfaces.iter() {
                let mut buf = [0.0; 20];
                let local_uniform_buf_index = self
                    .msurf_rdr_state
                    .locals_lookup
                    .iter()
                    .position(|h| *h == **hdl)
                    .unwrap();
                buf[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                buf[16..20].copy_from_slice(&[
                    state.min + state.height_offset,
                    state.max + state.height_offset,
                    state.max - state.min,
                    state.scale,
                ]);
                self.ctx.gpu.queue.write_buffer(
                    &self.msurf_rdr_state.local_uniform_buffer,
                    local_uniform_buf_index as u64 * aligned_size as u64,
                    bytemuck::cast_slice(&buf),
                );
            }
        }

        // Update GUI context.
        self.ctx.gui.update(window);

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];

        // Update the renderings.
        self.render(window)?;

        // Rendering the SamplingInspector is done after the main render pass.
        if self.dbg_drawing_state.sampling_debug_enabled {
            self.ui
                .tools
                .get_tool_mut::<SamplingInspector>()
                .unwrap()
                .render();
        }

        self.bsdf_viewer.write().unwrap().render();

        Ok(())
    }

    /// Render the frame to the surface.
    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        // Get the next frame (`SurfaceTexture`) to render to.
        let output_frame = self.canvas.get_current_texture()?;
        // Get a `TextureView` to the output frame's color attachment.
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Command encoders for the current frame.
        let mut main_encoder =
            self.ctx
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_render_encoder"),
                });

        {
            let cache = self.cache.read().unwrap();
            let visible_surfaces = self.ui.outliner().visible_surfaces();
            {
                let mut render_pass = main_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Main Render Pass"),
                    color_attachments: &[Some(
                        // This is what [[location(0)]] in the fragment shader targets
                        wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            // This is the texture that will receive the resolved output; will be
                            // the same as `view` unless multisampling.
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(self.theme.visuals().clear_color),
                                store: true,
                            },
                        },
                    )],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_map.depth_attachment.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });

                let aligned_micro_surface_uniform_size =
                    MicroSurfaceUniforms::aligned_size(&self.ctx.gpu.device);

                if !visible_surfaces.is_empty() {
                    render_pass.set_pipeline(&self.msurf_rdr_state.pipeline);
                    render_pass.set_bind_group(0, &self.msurf_rdr_state.globals_bind_group, &[]);

                    for (hdl, _) in visible_surfaces.iter() {
                        let renderable =
                            cache.get_micro_surface_renderable_mesh_by_surface_id(**hdl);
                        if renderable.is_none() {
                            log::debug!(
                                "Failed to get renderable mesh for surface {:?}, skipping.",
                                hdl
                            );
                            continue;
                        }
                        let buf_index = self
                            .msurf_rdr_state
                            .locals_lookup
                            .iter()
                            .position(|x| x == *hdl)
                            .unwrap();
                        let renderable = renderable.unwrap();
                        render_pass.set_bind_group(
                            1,
                            &self.msurf_rdr_state.locals_bind_group,
                            &[buf_index as u32 * aligned_micro_surface_uniform_size],
                        );
                        render_pass.set_vertex_buffer(0, renderable.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            renderable.index_buffer.slice(..),
                            renderable.index_format,
                        );
                        render_pass.draw_indexed(0..renderable.indices_count, 0, 0..1);
                    }
                }

                self.visual_grid_state.render(&mut render_pass);
            }
        }

        let dbg_drawing_encoder = self.dbg_drawing_state.record_render_pass(
            &self.ctx.gpu,
            Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }),
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_map.depth_attachment.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        );

        // UI render pass recoding.
        let ui_render_output = self.ctx.gui.render(
            window,
            self.canvas.screen_descriptor(),
            &output_view,
            |ctx| {
                self.theme.update(ctx);
                self.ui
                    .show(ctx, self.theme.kind(), &mut self.visual_grid_state.visible);
                self.toasts.show(ctx);
            },
        );

        let command_buffers: Box<dyn Iterator<Item = wgpu::CommandBuffer>> =
            match dbg_drawing_encoder {
                None => Box::new(
                    ui_render_output
                        .user_cmds
                        .into_iter()
                        .chain([main_encoder.finish(), ui_render_output.ui_cmd]),
                ),
                Some(dbg_encoder) => Box::new(ui_render_output.user_cmds.into_iter().chain([
                    main_encoder.finish(),
                    dbg_encoder.finish(),
                    ui_render_output.ui_cmd,
                ])),
            };

        // Submit the command buffers to the GPU: first the user's command buffers, then
        // the main render pass, and finally the UI render pass.
        self.ctx.gpu.queue.submit(command_buffers);

        self.depth_map
            .copy_to_buffer(&self.ctx.gpu, self.canvas.width(), self.canvas.height());

        // Present the frame to the screen.
        output_frame.present();

        Ok(())
    }

    pub fn on_user_event(&mut self, event: VgonioEvent, control_flow: &mut ControlFlow) {
        use VgonioEvent::*;
        match event {
            Quit => {
                *control_flow = ControlFlow::Exit;
            }
            RequestRedraw => {}
            OpenFiles(files) => self.open_files(files),
            Debugging(event) => {
                // TODO: handle events inside the DebuggingState.
                match event {
                    DebuggingEvent::ToggleSamplingRendering(enabled) => {
                        self.dbg_drawing_state.sampling_debug_enabled = enabled;
                    }
                    DebuggingEvent::UpdateDepthMap => {
                        self.depth_map.copy_to_buffer(
                            &self.ctx.gpu,
                            self.canvas.width(),
                            self.canvas.height(),
                        );
                        self.ui
                            .tools
                            .get_tool_mut::<DebuggingInspector>()
                            .unwrap()
                            .depth_map_pane
                            .update_depth_map(
                                &self.ctx.gpu,
                                &self.ctx.gui,
                                &self.depth_map.depth_attachment_storage,
                                self.depth_map.width,
                                self.canvas.height(),
                            );
                    }
                    DebuggingEvent::UpdateEmitterSamples {
                        samples,
                        orbit_radius,
                        shape_radius,
                    } => {
                        self.dbg_drawing_state.update_emitter_samples(
                            &self.ctx.gpu,
                            samples,
                            orbit_radius,
                            shape_radius,
                        );
                    }
                    DebuggingEvent::UpdateEmitterPoints {
                        points,
                        orbit_radius,
                    } => {
                        self.dbg_drawing_state.update_emitter_points(
                            &self.ctx.gpu,
                            points,
                            orbit_radius,
                        );
                    }
                    DebuggingEvent::ToggleEmitterPointsDrawing(status) => {
                        self.dbg_drawing_state.emitter_points_drawing = status;
                    }
                    DebuggingEvent::UpdateEmitterPosition {
                        zenith,
                        azimuth,
                        orbit_radius,
                        shape_radius,
                    } => {
                        self.dbg_drawing_state.update_emitter_position(
                            &self.ctx.gpu,
                            zenith,
                            azimuth,
                            orbit_radius,
                            shape_radius,
                        );
                    }
                    DebuggingEvent::EmitRays {
                        orbit_radius,
                        shape_radius,
                    } => {
                        self.dbg_drawing_state
                            .emit_rays(&self.ctx.gpu, orbit_radius, shape_radius);
                    }
                    DebuggingEvent::ToggleDebugDrawing(status) => {
                        self.dbg_drawing_state.enabled = status;
                    }
                    DebuggingEvent::ToggleCollectorDrawing {
                        status,
                        scheme,
                        patches,
                        orbit_radius,
                        shape_radius,
                    } => self.dbg_drawing_state.update_collector_drawing(
                        &self.ctx.gpu,
                        status,
                        Some(scheme),
                        patches,
                        orbit_radius,
                        shape_radius,
                    ),
                    DebuggingEvent::UpdateRayParams {
                        t,
                        orbit_radius,
                        shape_radius,
                    } => {
                        self.dbg_drawing_state.update_ray_params(
                            &self.ctx.gpu,
                            t,
                            orbit_radius,
                            shape_radius,
                        );
                    }
                    DebuggingEvent::ToggleEmitterRaysDrawing(status) => {
                        self.dbg_drawing_state.emitter_rays_drawing = status;
                    }
                    DebuggingEvent::ToggleEmitterSamplesDrawing(status) => {
                        self.dbg_drawing_state.emitter_samples_drawing = status;
                    }
                    DebuggingEvent::MeasureOnePoint {
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
                        self.dbg_drawing_state.ray_trajectories_drawing_reflected = reflected;
                        self.dbg_drawing_state.ray_trajectories_drawing_missed = missed;
                    }
                    DebuggingEvent::ToggleCollectedRaysDrawing(status) => {
                        self.dbg_drawing_state.collector_ray_hit_points_drawing = status;
                    }
                    DebuggingEvent::UpdateSurfacePrimitiveId { mesh, id, status } => {
                        self.dbg_drawing_state
                            .update_surface_primitive_id(mesh, id, status);
                    }
                    DebuggingEvent::UpdateGridCellDrawing { .. } => {
                        todo!("UpdateGridCellDrawing")
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
                    measure::microfacet::measure_area_distribution(
                        params,
                        &surfaces,
                        &self.cache.read().unwrap(),
                    );
                    todo!("Save area distribution to file or display it in a window");
                }
                MeasureEvent::Mmsf { params, surfaces } => {
                    println!("Measuring masking/shadowing");
                    measure::microfacet::measure_masking_shadowing(
                        params,
                        &surfaces,
                        &self.cache.read().unwrap(),
                        Handedness::RightHandedYUp,
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
            Notify { kind, text, time } => {
                self.toasts.add(Toast {
                    kind,
                    text: text.into(),
                    options: ToastOptions::default()
                        .duration_in_seconds(time as f64)
                        .show_progress(true)
                        .show_icon(true),
                });
            }
            UpdateThemeKind(kind) => self.theme.set_theme_kind(kind),
            _ => {}
        }
    }

    /// Update the state of the application then render the current frame.
    pub fn on_redraw_requested(
        &mut self,
        window: &Window,
        dt: Duration,
        control: &mut ControlFlow,
    ) {
        match self.update(window, dt) {
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

// VgonioEvent handling
impl VgonioGuiApp {
    fn open_files(&mut self, files: Vec<rfd::FileHandle>) {
        let mut surfaces = vec![];
        let mut measurements = vec![];
        // TODO: handle other file types
        for file in files {
            let path: PathBuf = file.into();
            let ext = match path.extension() {
                None => None,
                Some(s) => s.to_str().map(|s| s.to_lowercase()),
            };

            if let Some(ext) = ext {
                match ext.as_str() {
                    "vgmo" => {
                        // Micro-surface measurement data
                        log::debug!("Opening micro-surface measurement output: {:?}", path);
                        match self
                            .cache
                            .write()
                            .unwrap()
                            .load_micro_surface_measurement(&self.config, &path)
                        {
                            Ok(hdl) => {
                                measurements.push(hdl);
                            }
                            Err(e) => {
                                log::error!("Failed to load micro surface measurement: {:?}", e);
                            }
                        }
                    }
                    "vgms" | "txt" => {
                        // Micro-surface profile
                        log::debug!("Opening micro-surface profile: {:?}", path);
                        let mut locked_cache = self.cache.write().unwrap();
                        match locked_cache.load_micro_surface(&self.config, &path) {
                            Ok((surf, _)) => {
                                let _ = locked_cache
                                    .create_micro_surface_renderable_mesh(
                                        &self.ctx.gpu.device,
                                        surf,
                                    )
                                    .unwrap();
                                surfaces.push(surf)
                            }
                            Err(e) => {
                                log::error!("Failed to load micro surface: {:?}", e);
                            }
                        }
                    }
                    "spd" => {
                        todo!()
                    }
                    "csv" => {
                        todo!()
                    }
                    _ => {}
                }
            } else {
                log::warn!("File {:?} has no extension, ignoring", path);
            }
        }
        let cache = self.cache.read().unwrap();
        self.msurf_rdr_state.update_locals_lookup(&surfaces);
        self.ui.update_loaded_surfaces(&surfaces, &cache);
        self.ui
            .outliner_mut()
            .update_measurement_data(&measurements, &cache);
    }
}

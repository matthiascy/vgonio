mod gizmo;
mod plotter;
mod simulation;
pub mod state;
mod tools;
mod ui;
mod widgets;

use crate::{
    common::Handedness,
    error::Error,
    measure::{Ray, RtcMethod},
    units::degrees,
};
use glam::{IVec2, Mat4, Vec3};
use std::{
    cell::RefCell, collections::HashMap, default::Default, ops::Deref, path::PathBuf, sync::Arc,
    time::Instant,
};

#[cfg(feature = "embree")]
pub(crate) use tools::trace_ray_standard_dbg;

pub(crate) use tools::{trace_ray_grid_dbg, VisualDebugger};
pub use ui::VgonioUi;
use wgpu::util::DeviceExt;

use crate::{
    app::{
        cache::{Cache, Handle},
        gfx::{
            camera::{Camera, Projection, ProjectionKind},
            GpuContext, RenderPass, RenderableMesh, Texture, WgpuConfig,
            DEFAULT_BIND_GROUP_LAYOUT_DESC,
        },
        gui::state::{camera::CameraState, DebugState, DepthMap, GuiState, InputState},
    },
    measure::rtc::GridRayTracing,
    msurf::{AxisAlignment, MicroSurface, MicroSurfaceMesh},
};
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

/// Event processing state.
pub enum EventResponse {
    /// Event wasn't handled, continue processing.
    Ignored,
    /// Event was handled, stop processing events.
    Consumed,
}

impl EventResponse {
    /// Returns true if the event was consumed.
    pub fn is_consumed(&self) -> bool {
        match self {
            EventResponse::Ignored => false,
            EventResponse::Consumed => true,
        }
    }
}

impl From<egui_winit::EventResponse> for EventResponse {
    fn from(resp: egui_winit::EventResponse) -> Self {
        match resp.consumed {
            true => EventResponse::Consumed,
            false => EventResponse::Ignored,
        }
    }
}

/// Events used by Vgonio application.
#[derive(Debug)]
pub enum VgonioEvent {
    Quit,
    RequestRedraw,
    OpenFile(rfd::FileHandle),
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
    UpdateSamplingDebugger {
        count: u32,
        azimuth: (f32, f32),
        zenith: (f32, f32),
    },
}

use self::tools::SamplingDebugger;

use super::{gfx::WindowSurface, Config};

/// Launches Vgonio GUI application.
pub fn run(config: Config) -> Result<(), Error> {
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

    let mut vgonio = pollster::block_on(VgonioApp::new(config, &window, &event_loop))?;

    let mut last_frame_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = Instant::now();
        let dt = now - last_frame_time;
        last_frame_time = now;

        match event {
            Event::UserEvent(VgonioEvent::Quit) => {
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(event) => vgonio.on_user_event(event),
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                if !vgonio.on_event(event).is_consumed() {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                        WindowEvent::Resized(new_size) => {
                            vgonio.resize(*new_size, Some(window.scale_factor() as f32))
                        }

                        WindowEvent::ScaleFactorChanged {
                            new_inner_size,
                            scale_factor,
                        } => vgonio.resize(**new_inner_size, Some(*scale_factor as f32)),

                        _ => {}
                    }
                }
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                vgonio.update(&window, dt);
                match vgonio.render(&window) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => vgonio.reconfigure_surface(),
                    // The system is out of memory, we should quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{e:?}"),
                }
            }

            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually request it.
                window.request_redraw()
            }

            _ => {}
        }
    })
}

/// Micro-surface presentation view.
#[derive(Debug)]
struct MicroSurfaceView {
    /// Handle to the micro-surface.
    surf: Handle<MicroSurface>,
    /// Handle to the micro-surface mesh.
    mesh: Handle<MicroSurfaceMesh>,
    /// Handle to the micro-surface's renderable mesh.
    renderable: Handle<RenderableMesh>,
    /// Scale factor when rendering the micro-surface.
    scale_factor: f32,
    /// Visibility of the micro-surface in the scene.
    visible: bool,
}

/// Vgonio client application with GUI.
pub struct VgonioApp {
    /// GPU context for rendering.
    gpu_ctx: GpuContext,

    /// Surface for presenting rendered frames.
    win_surf: WindowSurface,

    /// GUI context and state for rendering.
    gui_state: GuiState,

    /// The GUI application state.
    ui: VgonioUi,

    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// The cache of the application including preloaded datafiles. See
    /// [`VgonioCache`].
    cache: Arc<RefCell<Cache>>,

    input: InputState,
    camera: CameraState,

    /// Different pipelines with its binding groups used for rendering.
    passes: HashMap<&'static str, RenderPass>,

    depth_map: DepthMap,
    debug_drawing: DebugState,
    debug_drawing_enabled: bool,

    msurf: Option<MicroSurfaceView>,

    // surface: Option<Rc<MicroSurface>>,
    // surface_mesh: Option<Rc<MicroSurfaceMesh>>,
    // surface_mesh_view: Option<RenderableMesh>,
    // surface_scale_factor: f32,
    // opened_surfaces: Vec<PathBuf>,
    pub start_time: Instant,
    pub prev_frame_time: Option<f32>,

    pub demos: egui_demo_lib::DemoWindows,
    pub visual_grid_enabled: bool,
    // pub surface_visible: bool,
}

impl VgonioApp {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(
        config: Config,
        window: &Window,
        event_loop: &EventLoop<VgonioEvent>,
    ) -> Result<Self, Error> {
        let wgpu_config = WgpuConfig {
            device_descriptor: wgpu::DeviceDescriptor {
                label: Some("vgonio-wgpu-device"),
                features: wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::TIMESTAMP_QUERY,
                limits: wgpu::Limits::default(),
            },
            ..Default::default()
        };
        let (gpu_ctx, surface) = GpuContext::new(window, &wgpu_config).await;
        let win_surf = WindowSurface::new(&gpu_ctx, window, &wgpu_config, surface);
        let depth_map = DepthMap::new(&gpu_ctx, win_surf.width(), win_surf.height());
        let camera = {
            let camera = Camera::new(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
            let projection = Projection::new(
                0.1,
                100.0,
                60.0f32.to_radians(),
                win_surf.width(),
                win_surf.height(),
            );
            CameraState::new(camera, projection, ProjectionKind::Perspective)
        };

        let heightfield_pass = create_heightfield_pass(&gpu_ctx, win_surf.format());
        let visual_grid_pass = create_visual_grid_pass(&gpu_ctx, win_surf.format());

        let mut passes = HashMap::new();
        passes.insert("visual_grid", visual_grid_pass);
        passes.insert("heightfield", heightfield_pass);

        let mut gui_state = GuiState::new(
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
            win_surf.format(),
            event_loop,
            1,
        );

        let config = Arc::new(config);
        let cache = {
            let mut _cache = Cache::new(config.cache_dir());
            _cache.load_ior_database(&config);
            Arc::new(RefCell::new(_cache))
        };

        let gui = VgonioUi::new(
            event_loop.create_proxy(),
            config.clone(),
            cache.clone(),
            &gpu_ctx,
            &mut gui_state.renderer,
        );

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        let debug_drawing = DebugState::new(&gpu_ctx, win_surf.format());

        Ok(Self {
            gpu_ctx,
            gui_state,
            config,
            ui: gui,
            cache,
            input,
            passes,
            depth_map,
            debug_drawing,
            debug_drawing_enabled: true,
            // surface: None,
            // surface_mesh: None,
            // surface_mesh_view: None,
            msurf: None,
            camera,
            start_time: Instant::now(),
            prev_frame_time: None,
            demos: egui_demo_lib::DemoWindows::default(),
            visual_grid_enabled: true,
            // opened_surfaces: vec![],
            // surface_visible: true,
            win_surf,
        })
    }

    #[inline]
    pub fn surface_width(&self) -> u32 { self.win_surf.width() }

    #[inline]
    pub fn surface_height(&self) -> u32 { self.win_surf.height() }

    pub fn reconfigure_surface(&mut self) { self.win_surf.reconfigure(&self.gpu_ctx.device); }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>, scale_factor: Option<f32>) {
        if self.win_surf.resize(
            &self.gpu_ctx.device,
            new_size.width,
            new_size.height,
            scale_factor,
        ) {
            self.depth_map
                .resize(&self.gpu_ctx, new_size.width, new_size.height);
            self.camera
                .projection
                .resize(new_size.width, new_size.height);
        }
    }

    pub fn on_event(&mut self, event: &WindowEvent) -> EventResponse {
        match self.gui_state.on_event(event) {
            EventResponse::Ignored => match event {
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
                    EventResponse::Consumed
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    self.input.update_scroll_delta(*delta);
                    EventResponse::Consumed
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    self.input.update_mouse_map(*button, *state);
                    EventResponse::Consumed
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.input.update_cursor_delta((*position).cast::<f32>());
                    EventResponse::Consumed
                }
                _ => EventResponse::Ignored,
            },
            EventResponse::Consumed => EventResponse::Consumed,
        }
    }

    pub fn update(&mut self, window: &Window, dt: std::time::Duration) {
        // Update camera uniform.
        self.camera
            .update(&self.input, dt, ProjectionKind::Perspective);
        self.ui.update_gizmo_matrices(
            Mat4::IDENTITY,
            Mat4::look_at_rh(self.camera.camera.eye, Vec3::ZERO, self.camera.camera.up),
            Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
        );

        let (view, proj) = (
            self.camera.uniform.view_matrix,
            self.camera.uniform.proj_matrix,
        );

        self.gpu_ctx.queue.write_buffer(
            self.passes
                .get("visual_grid")
                .unwrap()
                .uniform_buffer
                .as_ref()
                .unwrap(),
            0,
            bytemuck::cast_slice(&[
                view,
                proj,
                self.camera.uniform.view_inv_matrix,
                self.camera.uniform.proj_inv_matrix,
            ]),
        );

        if let Some(msurf_view) = &self.msurf {
            let cache = self.cache.deref().borrow();
            let msurf = cache.micro_surface(msurf_view.surf).unwrap();
            let mut uniform = [0.0f32; 16 * 3 + 4];
            // Displace visually the heightfield.
            uniform[0..16].copy_from_slice(
                &Mat4::from_translation(Vec3::new(
                    0.0,
                    -(msurf.max + msurf.min) * 0.5 * msurf_view.scale_factor,
                    0.0,
                ))
                .to_cols_array(),
            );
            uniform[16..32].copy_from_slice(&self.camera.uniform.view_matrix.to_cols_array());
            uniform[32..48].copy_from_slice(&self.camera.uniform.proj_matrix.to_cols_array());
            uniform[48..52].copy_from_slice(&[
                msurf.min,
                msurf.max,
                msurf.max - msurf.min,
                msurf_view.scale_factor,
            ]);
            self.gpu_ctx.queue.write_buffer(
                self.passes
                    .get("heightfield")
                    .unwrap()
                    .uniform_buffer
                    .as_ref()
                    .unwrap(),
                0,
                bytemuck::cast_slice(&uniform),
            );

            self.gpu_ctx.queue.write_buffer(
                &self.debug_drawing.prim_uniform_buffer,
                0,
                bytemuck::cast_slice(&uniform),
            );

            // Update the uniform buffer for debug drawing.
            if self.debug_drawing_enabled {
                self.debug_drawing
                    .update_uniform_buffer(&self.gpu_ctx, &uniform);
            }
        }

        self.gui_state.update(window);

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];
    }

    /// Render the frame to the surface.
    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let cache = self.cache.deref().borrow();
        // Get the next frame (`SurfaceTexture`) to render to.
        let output_frame = self.win_surf.get_current_texture()?;
        // Get a `TextureView` to the output frame's color attachment.
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Command encoders for the current frame.
        let mut encoders = [
            self.gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_render_encoder"),
                }),
            self.gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_render_debug_encoder"),
                }),
        ];

        // Main render pass
        {
            let mut render_pass = encoders[0].begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        // This is the texture that will receive the resolved output; will be the
                        // same as `view` unless multisampling.
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.046,
                                g: 0.046,
                                b: 0.046,
                                a: 1.0,
                            }),
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

            if let Some(ref msurf_view) = self.msurf {
                if msurf_view.visible {
                    let renderable = cache
                        .micro_surface_renderable_mesh(msurf_view.renderable)
                        .expect("Failed to get renderable mesh");

                    let pass = self.passes.get("heightfield").unwrap();
                    render_pass.set_pipeline(&pass.pipeline);
                    render_pass.set_bind_group(0, &pass.bind_groups[0], &[]);
                    render_pass.set_vertex_buffer(0, renderable.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        renderable.index_buffer.slice(..),
                        renderable.index_format,
                    );
                    render_pass.draw_indexed(0..renderable.indices_count, 0, 0..1);
                }
            }

            if self.visual_grid_enabled {
                let pass = self.passes.get("visual_grid").unwrap();
                render_pass.set_pipeline(&pass.pipeline);
                render_pass.set_bind_group(0, &pass.bind_groups[0], &[]);
                render_pass.draw(0..6, 0..1);
            }
        }

        if self.debug_drawing_enabled {
            let mut render_pass = encoders[1].begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.debug_drawing.render_pass_rd.pipeline);
            render_pass.set_bind_group(0, &self.debug_drawing.render_pass_rd.bind_groups[0], &[]);
            render_pass.set_vertex_buffer(0, self.debug_drawing.vert_buffer.slice(..));
            render_pass.draw(0..self.debug_drawing.vert_count, 0..1);

            render_pass.set_pipeline(&self.debug_drawing.prim_pipeline);
            render_pass.set_bind_group(0, &self.debug_drawing.prim_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.debug_drawing.prim_vert_buffer.slice(..));
            render_pass.draw(0..3, 0..1)
        }

        // UI render pass recoding.
        let ui_render_output = self.gui_state.render(
            window,
            self.win_surf.screen_descriptor(),
            &output_view,
            |ctx| {
                // self.demos.ui(ctx);
                self.ui.show(ctx);
            },
        );

        // Submit the command buffers to the GPU: first the user's command buffers, then
        // the main render pass, and finally the UI render pass.
        self.gpu_ctx.queue.submit(
            ui_render_output.user_cmds.into_iter().chain(
                encoders
                    .into_iter()
                    .map(|enc| enc.finish())
                    .chain(std::iter::once(ui_render_output.ui_cmd)),
            ),
        );

        self.depth_map
            .copy_to_buffer(&self.gpu_ctx, self.win_surf.width(), self.win_surf.height());

        // Present the frame to the screen.
        output_frame.present();

        Ok(())
    }

    pub fn on_user_event(&mut self, event: VgonioEvent) {
        match event {
            VgonioEvent::RequestRedraw => {}
            VgonioEvent::OpenFile(handle) => {
                // TODO: deal with different file types
                let path: PathBuf = handle.into();
                log::debug!("Opening file: {:?}", path);
                let cache = self.cache.deref();
                let (surf, mesh) = cache
                    .borrow_mut()
                    .load_micro_surface(&self.config, &path, Some(AxisAlignment::XZ))
                    .unwrap();
                self.ui
                    .simulation_workspace
                    .update_surface_list(&cache.borrow().loaded_micro_surface_paths().unwrap());
                let renderable = cache
                    .borrow_mut()
                    .create_micro_surface_renderable_mesh(&self.gpu_ctx.device, surf)
                    .unwrap();
                self.msurf = Some(MicroSurfaceView {
                    surf,
                    mesh,
                    renderable,
                    scale_factor: 1.0,
                    visible: true,
                });
            }
            VgonioEvent::ToggleGrid => {
                self.visual_grid_enabled = !self.visual_grid_enabled;
            }
            VgonioEvent::UpdateDepthMap => {
                self.depth_map.copy_to_buffer(
                    &self.gpu_ctx,
                    self.win_surf.width(),
                    self.win_surf.height(),
                );
                self.ui
                    .tools
                    .get_tool::<VisualDebugger>("Visual Debugger")
                    .unwrap()
                    .shadow_map_pane
                    .update_depth_map(
                        &self.gpu_ctx,
                        &self.gui_state,
                        &self.depth_map.depth_attachment_storage,
                        self.depth_map.width,
                        self.win_surf.height(),
                    );
            }
            VgonioEvent::UpdateSurfaceScaleFactor(factor) => {
                if let Some(msurf_view) = &mut self.msurf {
                    msurf_view.scale_factor = factor;
                }
            }
            VgonioEvent::UpdateDebugT(t) => {
                self.debug_drawing.ray_t = t;
            }
            VgonioEvent::UpdatePrimId(prim_id) => {
                if let Some(msurf_view) = &self.msurf {
                    let cache = self.cache.deref().borrow();
                    let msurf = cache.micro_surface_mesh(msurf_view.mesh).unwrap();
                    let id = prim_id as usize;
                    if id < msurf.facets.len() {
                        let indices = [
                            msurf.facets[id * 3],
                            msurf.facets[id * 3 + 1],
                            msurf.facets[id * 3 + 2],
                        ];
                        let vertices = [
                            msurf.verts[indices[0] as usize],
                            msurf.verts[indices[1] as usize],
                            msurf.verts[indices[2] as usize],
                        ];
                        log::debug!(
                            "query prim {}: {:?}\n    p0: {:?}\n    p1: {:?}\n    p2: {:?}",
                            prim_id,
                            indices,
                            vertices[0],
                            vertices[1],
                            vertices[2]
                        );
                        self.gpu_ctx.queue.write_buffer(
                            &self.debug_drawing.prim_vert_buffer,
                            0,
                            bytemuck::cast_slice(&vertices),
                        );
                    }
                }
            }
            VgonioEvent::TraceRayDbg {
                ray,
                method,
                max_bounces,
            } => {
                log::debug!("= = = = [Debug Ray Tracing] = = = =\n  => {:?}", ray);
                if self.msurf.is_none() {
                    log::warn!("No micro-surface loaded, can't trace ray!");
                } else {
                    match method {
                        #[cfg(feature = "embree")]
                        RtcMethod::Embree => {
                            log::debug!("  => [Standard Ray Tracing]");
                            self.debug_drawing.rays = trace_ray_standard_dbg(
                                ray,
                                max_bounces,
                                self.surface_mesh.as_ref().unwrap(),
                            );
                        }
                        #[cfg(feature = "optix")]
                        RtcMethod::Optix => {
                            todo!()
                        }
                        RtcMethod::Grid => {
                            log::debug!("  => [Grid Ray Tracing]");
                            let cache = self.cache.deref().borrow();
                            let grid_rt = GridRayTracing::new(
                                cache
                                    .micro_surface(self.msurf.as_ref().unwrap().surf)
                                    .unwrap(),
                                cache
                                    .micro_surface_mesh(self.msurf.as_ref().unwrap().mesh)
                                    .unwrap(),
                            );
                            self.debug_drawing.rays =
                                trace_ray_grid_dbg(ray, max_bounces, &grid_rt);
                        }
                    }
                    if self.debug_drawing_enabled {
                        let mut content = self
                            .debug_drawing
                            .rays
                            .iter()
                            .map(|r| r.o)
                            .collect::<Vec<_>>();
                        let last_ray = self.debug_drawing.rays[self.debug_drawing.rays.len() - 1];
                        content.push(last_ray.o + last_ray.d * self.debug_drawing.ray_t);
                        log::debug!("content: {:?}", content);
                        self.debug_drawing.vert_count = self.debug_drawing.rays.len() as u32 + 1;
                        self.gpu_ctx.queue.write_buffer(
                            &self.debug_drawing.vert_buffer,
                            0,
                            bytemuck::cast_slice(&content),
                        );
                    }
                }
            }
            VgonioEvent::ToggleDebugDrawing => {
                self.debug_drawing_enabled = !self.debug_drawing_enabled;
                println!("Debug drawing: {}", self.debug_drawing_enabled);
            }
            VgonioEvent::ToggleSurfaceVisibility => {
                if let Some(msurf_view) = &mut self.msurf {
                    msurf_view.visible = !msurf_view.visible;
                }
            }
            VgonioEvent::UpdateSamplingDebugger {
                count,
                azimuth,
                zenith,
            } => {
                let samples = crate::measure::emitter::uniform_sampling_on_unit_sphere(
                    count as usize,
                    degrees!(zenith.0).into(),
                    degrees!(zenith.1).into(),
                    degrees!(azimuth.0).into(),
                    degrees!(azimuth.1).into(),
                    Handedness::RightHandedYUp,
                );
                let mut encoder =
                    self.gpu_ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sampling Debugger Encoder"),
                        });
                self.ui
                    .tools
                    .get_tool::<SamplingDebugger>("Sampling Debugger")
                    .unwrap()
                    .record_render_pass(&self.gpu_ctx, &mut encoder, &samples);
                self.gpu_ctx.queue.submit(Some(encoder.finish()));
            }
            _ => {}
        }
    }

    // fn load_height_field(&mut self, path: &Path, alignment:
    // Option<AxisAlignment>) {     match MicroSurface::from_file(path, None,
    // alignment) {         Ok(mut hf) => {
    //             log::info!(
    //                 "Heightfield loaded: {}, rows = {}, cols = {}, du = {}, dv =
    // {}",                 path.display(),
    //                 hf.rows,
    //                 hf.cols,
    //                 hf.du,
    //                 hf.dv
    //             );
    //             let mesh = hf.triangulate();
    //             self.surface = Some(Rc::new(hf));
    //             self.surface_mesh = Some(Rc::new(mesh));
    //         }
    //         Err(err) => {
    //             log::error!("HeightField loading error: {}", err);
    //         }
    //     }
    // }
}

fn create_heightfield_pass(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> RenderPass {
    // Load shader
    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("height_field_shader_module"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("./gui/assets/shaders/wgsl/heightfield.wgsl").into(),
            ),
        });

    // Create uniform buffer for rendering height field
    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform-buffer"),
            contents: bytemuck::cast_slice(&[0.0f32; 16 * 3 + 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("height_field_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("height_field_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    // Create height field render pipeline
    let render_pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("height_field_render_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("height_field_render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Line,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, /* tells when to discard a
                                                             * new pixel */
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

    RenderPass {
        pipeline,
        bind_groups: vec![bind_group],
        uniform_buffer: Some(uniform_buffer),
    }
}

fn create_visual_grid_pass(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> RenderPass {
    // let grid_vert_shader = ctx
    //     .device
    //     .create_shader_module(wgpu::include_spirv!("../assets/shaders/spirv/grid.
    // vert.spv")); let grid_frag_shader = ctx
    //     .device
    //     .create_shader_module(wgpu::include_spirv!("../assets/shaders/spirv/grid.
    // frag.spv"));
    let grid_shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("./gui/assets/shaders/wgsl/grid.wgsl"));
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&DEFAULT_BIND_GROUP_LAYOUT_DESC);
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid_render_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid_uniform_buffer"),
            contents: bytemuck::bytes_of(&crate::app::gfx::VisualGridUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grid_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, /* tells when to discard a
                                                             * new pixel */
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    // blend: Some(wgpu::BlendState {
                    //     color: wgpu::BlendComponent {
                    //         src_factor: wgpu::BlendFactor::SrcAlpha,
                    //         dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    //         operation: wgpu::BlendOperation::Add,
                    //     },
                    //     alpha: wgpu::BlendComponent {
                    //         src_factor: wgpu::BlendFactor::One,
                    //         dst_factor: wgpu::BlendFactor::Zero,
                    //         operation: wgpu::BlendOperation::Add,
                    //     },
                    // }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

    RenderPass {
        pipeline,
        bind_groups: vec![bind_group],
        uniform_buffer: Some(uniform_buffer),
    }
}

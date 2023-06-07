mod gizmo;
mod icons;
mod misc;
pub mod outliner;
mod simulations;
pub mod state;
mod tools;
mod ui;
mod widgets;

// TODO: MSAA

use crate::{
    error::Error,
    measure,
    measure::{rtc::Ray, RtcMethod},
    units::degrees,
    Handedness,
};
use glam::{IVec2, Mat4, Vec3, Vec4};
use std::{
    any::Any,
    default::Default,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

// #[cfg(feature = "embree")]
// pub(crate) use tools::trace_ray_standard_dbg;

pub(crate) use tools::DebuggingInspector;
pub use ui::VgonioUi;
use wgpu::util::DeviceExt;

use crate::{
    app::{
        cache::{Cache, Handle},
        gfx::{
            camera::{Camera, Projection, ProjectionKind, ViewProjUniform},
            GpuContext, Texture, VisualGridUniforms, WgpuConfig, DEFAULT_BIND_GROUP_LAYOUT_DESC,
        },
        gui::{
            state::{camera::CameraState, DebugDrawingState, DepthMap, GuiContext, InputState},
            ui::Theme,
        },
    },
    measure::measurement::{BsdfMeasurementParams, MadfMeasurementParams, MmsfMeasurementParams},
    msurf::MicroSurface,
    units::Degrees,
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
#[non_exhaustive]
pub enum VgonioEvent {
    Quit,
    RequestRedraw,
    OpenFiles(Vec<rfd::FileHandle>),
    UpdateDepthMap,
    TraceRayDbg {
        ray: Ray,
        max_bounces: u32,
        method: RtcMethod,
    },
    ToggleSurfaceVisibility,
    UpdateDebugT(f32),
    UpdatePrimId(u32),
    UpdateCellPos(IVec2),
    UpdateSamplingDebugger {
        count: u32,
        azimuth: (f32, f32),
        zenith: (f32, f32),
    },
    MeasureAreaDistribution {
        params: MadfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
    MeasureMaskingShadowing {
        params: MmsfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
    MeasureBsdf {
        params: BsdfMeasurementParams,
        surfaces: Vec<Handle<MicroSurface>>,
    },
    CheckVisibleFacets {
        m_azimuth: Degrees,
        m_zenith: Degrees,
        opening_angle: Degrees,
    },
}

use self::tools::SamplingInspector;

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
    gpu: GpuContext,
    /// GUI context for rendering.
    gui: GuiContext,
}

/// Rendering resources for loaded [`MicroSurface`].
struct MicroSurfaceRenderingState {
    /// Render pipeline for rendering micro surfaces.
    pipeline: wgpu::RenderPipeline,
    /// Bind group containing global uniform buffer.
    globals_bind_group: wgpu::BindGroup,
    /// Bind group containing local uniform buffer.
    locals_bind_group: wgpu::BindGroup,
    /// Uniform buffer containing only view and projection matrices.
    global_uniform_buffer: wgpu::Buffer,
    /// Uniform buffer containing data subject to each loaded micro surface.
    local_uniform_buffer: wgpu::Buffer,
    /// Lookup table linking [`MicroSurface`] to its offset in the local uniform
    /// buffer.
    locals_lookup: Vec<Handle<MicroSurface>>,
}

impl MicroSurfaceRenderingState {
    pub const INITIAL_MICRO_SURFACE_COUNT: usize = 64;

    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("micro_surface_shader_module"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./gui/assets/shaders/wgsl/micro_surface.wgsl").into(),
                ),
            });
        let global_uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro_surface_global_uniform_buffer"),
            size: std::mem::size_of::<ViewProjUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let aligned_locals_size = MicroSurfaceUniforms::aligned_size(&ctx.device);
        let local_uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro_surface_local_uniform_buffer"),
            size: aligned_locals_size as u64 * Self::INITIAL_MICRO_SURFACE_COUNT as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("micro_surface_globals_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ViewProjUniform::SIZE_IN_BYTES as u64,
                            ),
                        },
                        count: None,
                    }],
                });

        let locals_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("micro_surface_locals_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(aligned_locals_size as u64),
                        },
                        count: None,
                    }],
                });

        let globals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_globals_bind_group"),
            layout: &globals_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        let locals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_locals_bind_group"),
            layout: &locals_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &local_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(aligned_locals_size as u64),
                }),
            }],
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("micro_surface_pipeline_layout"),
                bind_group_layouts: &[&globals_bind_group_layout, &locals_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("micro_surface_render_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 3 * std::mem::size_of::<f32>() as u64,
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
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Line,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
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
        Self {
            pipeline,
            globals_bind_group,
            locals_bind_group,
            global_uniform_buffer,
            local_uniform_buffer,
            locals_lookup: Default::default(),
        }
    }

    /// Only updates the lookup table. The actual data is updated in the
    /// [`VgonioGuiApp::update_old`] method.
    pub fn update_locals_lookup(&mut self, surfs: &[Handle<MicroSurface>]) {
        for hdl in surfs {
            if self.locals_lookup.contains(hdl) {
                continue;
            }
            self.locals_lookup.push(*hdl);
        }
    }
}

struct VisualGridState {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
}

impl VisualGridState {
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let vert_shader = ctx
            .device
            .create_shader_module(wgpu::include_spirv!(concat!(
                env!("OUT_DIR"),
                "/visual_grid.vert.spv"
            )));
        let frag_shader = ctx
            .device
            .create_shader_module(wgpu::include_spirv!(concat!(
                env!("OUT_DIR"),
                "/visual_grid.frag.spv"
            )));
        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&DEFAULT_BIND_GROUP_LAYOUT_DESC);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visual_grid_render_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visual_grid_uniform_buffer"),
                contents: bytemuck::bytes_of(&VisualGridUniforms::default()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visual_grid_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visual_grid_render_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vert_shader,
                    entry_point: "main",
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
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &frag_shader,
                    entry_point: "main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });
        Self {
            pipeline,
            bind_group,
            uniform_buffer,
        }
    }

    pub fn update_uniforms(
        &self,
        ctx: &GpuContext,
        view_proj: &ViewProjUniform,
        view_proj_inv: &ViewProjUniform,
        color: wgpu::Color,
        is_dark_mode: bool,
    ) {
        ctx.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&VisualGridUniforms {
                view: view_proj.view.to_cols_array(),
                proj: view_proj.proj.to_cols_array(),
                view_inv: view_proj_inv.view.to_cols_array(),
                proj_inv: view_proj_inv.proj.to_cols_array(),
                grid_line_color: [
                    color.r as f32,
                    color.g as f32,
                    color.b as f32,
                    if is_dark_mode { 1.0 } else { 0.0 },
                ],
            }),
        );
    }
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
    ui: VgonioUi,
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
    msurf_rdr_state: MicroSurfaceRenderingState,
    /// State of the visual grid rendering, including the pipeline, binding
    /// groups, and buffers.
    visual_grid_state: VisualGridState,
    /// Depth map of the scene. TODO: refactor
    depth_map: DepthMap,
    // TODO: add MSAA
    /// Debug drawing state.
    dbg_drawing_state: DebugDrawingState,
}

impl VgonioGuiApp {
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

        let mut gui_ctx = GuiContext::new(
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

        let mut ui = VgonioUi::new(
            event_loop.create_proxy(),
            config.clone(),
            &gpu_ctx,
            &mut gui_ctx.renderer,
            cache.clone(),
        );

        ui.set_theme(Theme::Light);

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        let dbg_drawing_state = DebugDrawingState::new(&gpu_ctx, canvas.format());
        let msurf_rdr_state = MicroSurfaceRenderingState::new(&gpu_ctx, canvas.format());

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
        match self.ctx.gui.on_window_event(event) {
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
            },
            EventResponse::Consumed => {}
        }
    }

    pub fn update(&mut self, window: &Window, dt: Duration) -> Result<(), Error> {
        // Update camera uniform.
        self.camera
            .update(&self.input, dt, ProjectionKind::Perspective);
        self.ui.update_gizmo_matrices(
            Mat4::IDENTITY,
            Mat4::look_at_rh(self.camera.camera.eye, Vec3::ZERO, self.camera.camera.up),
            Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
        );
        let current_theme_visuals = self.ui.current_theme_visuals();
        let view_proj = self.camera.uniform.view_proj;
        let view_proj_inv = self.camera.uniform.view_proj_inv;
        self.visual_grid_state.update_uniforms(
            &self.ctx.gpu,
            &view_proj,
            &view_proj_inv,
            current_theme_visuals.grid_line_color,
            current_theme_visuals.egui_visuals.dark_mode,
        );
        let dbg_tool = self.ui.tools.get_tool::<DebuggingInspector>().unwrap();
        let dbg_selected_surface = dbg_tool.brdf_pane.selected_surface;
        let (dome_radius, lowest, highest, scale) = match dbg_selected_surface {
            Some(surface) => {
                let radius = dbg_tool.brdf_pane.dome_radius;
                let state = self.ui.outliner().surfaces().get(&surface).unwrap();
                (radius, state.1.min, state.1.max, state.1.scale)
            }
            None => (1.0, 0.0, 1.0, 1.0),
        };
        self.dbg_drawing_state.update_uniform_buffer(
            &self.ctx.gpu,
            &view_proj.view,
            &view_proj.proj,
            dome_radius,
            lowest,
            highest,
            scale,
        );

        // Update uniform buffer for all visible surfaces.
        let visible_surfaces = self.ui.outliner().visible_surfaces();
        if !visible_surfaces.is_empty() {
            // Update global uniform buffer.
            log::trace!("Updating global uniform buffer.");
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
                buf[0..16].copy_from_slice(
                    &Mat4::from_translation(Vec3::new(0.0, state.y_offset * state.scale, 0.0))
                        .to_cols_array(),
                );
                buf[16..20].copy_from_slice(&[
                    state.min,
                    state.max,
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

        self.ctx.gui.update(window);

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];

        self.render(window)?;

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
        let mut encoders = [
            self.ctx
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_render_encoder"),
                }),
            self.ctx
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_dbg_render_encoder"),
                }),
        ];

        let cache = self.cache.read().unwrap();
        let visible_surfaces = self.ui.outliner().visible_surfaces();
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
                            load: wgpu::LoadOp::Clear(self.ui.current_theme_visuals().clear_color),
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
                    let renderable = cache.get_micro_surface_renderable_mesh_by_surface_id(**hdl);
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

            if self.ui.visual_grid_enabled {
                render_pass.set_pipeline(&self.visual_grid_state.pipeline);
                render_pass.set_bind_group(0, &self.visual_grid_state.bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }

        let dbg_inspector = self.ui.tools.get_tool::<DebuggingInspector>().unwrap();
        if dbg_inspector.debug_drawing_enabled {
            let mut render_pass = encoders[1].begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("debug_render_pass"),
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

            if dbg_inspector.brdf_pane.show_dome {
                render_pass.set_pipeline(&self.dbg_drawing_state.pipeline);
                render_pass.set_bind_group(0, &self.dbg_drawing_state.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.dbg_drawing_state.vertices.data_slice(..));
                render_pass.set_index_buffer(
                    self.dbg_drawing_state.indices.data_slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                self.dbg_drawing_state
                    .indices
                    .subslices()
                    .iter()
                    .enumerate()
                    .for_each(|(i, index_range)| {
                        let count = ((index_range.end - index_range.start) / 4) as u32;
                        let index_offset = (index_range.start / 4) as u32;
                        let vertex_offset =
                            self.dbg_drawing_state.vertices.subslices()[i].start / 12;
                        render_pass.draw_indexed(
                            index_offset..index_offset + count,
                            vertex_offset as i32,
                            0..1,
                        );
                    });
            }
        }

        // if self.debug_drawing_enabled {
        // if let Some(ref msurf_view) = self.msurf {
        //     let renderable = cache
        //         .micro_surface_renderable_mesh(msurf_view.renderable)
        //         .expect("Failed to get renderable mesh");
        //     let mut render_pass =
        // encoders[1].begin_render_pass(&wgpu::RenderPassDescriptor {
        //         label: Some("Render Pass"),
        //         color_attachments: &[Some(wgpu::RenderPassColorAttachment
        // {             view: &output_view,
        //             resolve_target: None,
        //             ops: wgpu::Operations {
        //                 load: wgpu::LoadOp::Load,
        //                 store: true,
        //             },
        //         })],
        //         depth_stencil_attachment: None,
        //     });
        //     if !self.debug_drawing.drawing_msurf_prims {
        //         render_pass.set_pipeline(&self.debug_drawing.rays_rp.
        // pipeline);         render_pass.set_bind_group(0,
        // &self.debug_drawing.rays_rp.bind_groups[0], &[]);
        //         render_pass.set_vertex_buffer(0,
        // self.debug_drawing.rays_vertex_buf.slice(..));
        //         render_pass.draw(0..self.debug_drawing.rays_vertex_count,
        // 0..1);     } else {
        //         render_pass.set_pipeline(&self.debug_drawing.
        // msurf_prim_rp.pipeline);         render_pass.
        // set_bind_group(             0,
        //             &self.debug_drawing.msurf_prim_rp.bind_groups[0],
        //             &[],
        //         );
        //         render_pass.set_vertex_buffer(0,
        // renderable.vertex_buffer.slice(..));
        //         render_pass.set_index_buffer(
        //             self.debug_drawing.msurf_prim_index_buf.slice(..),
        //             renderable.index_format,
        //         );
        //         if self.debug_drawing.multiple_prims {
        //             render_pass.draw_indexed(
        //                 3..3 + self.debug_drawing.msurf_prim_index_count,
        //                 0,
        //                 0..1,
        //             );
        //         } else {
        //             render_pass.draw_indexed(0..3, 0, 0..1);
        //         }
        //     }
        // }
        // }

        // UI render pass recoding.
        let ui_render_output = self.ctx.gui.render(
            window,
            self.canvas.screen_descriptor(),
            &output_view,
            |ctx| self.ui.show(ctx),
        );

        // Submit the command buffers to the GPU: first the user's command buffers, then
        // the main render pass, and finally the UI render pass.
        self.ctx.gpu.queue.submit(
            ui_render_output.user_cmds.into_iter().chain(
                encoders
                    .into_iter()
                    .map(|enc| enc.finish())
                    .chain(std::iter::once(ui_render_output.ui_cmd)),
            ),
        );

        self.depth_map
            .copy_to_buffer(&self.ctx.gpu, self.canvas.width(), self.canvas.height());

        // Present the frame to the screen.
        output_frame.present();

        Ok(())
    }

    pub fn on_user_event(&mut self, event: VgonioEvent, control_flow: &mut ControlFlow) {
        match event {
            VgonioEvent::Quit => {
                *control_flow = ControlFlow::Exit;
            }
            VgonioEvent::RequestRedraw => {}
            VgonioEvent::OpenFiles(files) => self.open_files(files),
            VgonioEvent::UpdateDepthMap => {
                self.depth_map.copy_to_buffer(
                    &self.ctx.gpu,
                    self.canvas.width(),
                    self.canvas.height(),
                );
                self.ui
                    .tools
                    .get_tool_mut::<DebuggingInspector>()
                    .unwrap()
                    .shadow_map_pane
                    .update_depth_map(
                        &self.ctx.gpu,
                        &self.ctx.gui,
                        &self.depth_map.depth_attachment_storage,
                        self.depth_map.width,
                        self.canvas.height(),
                    );
            }
            VgonioEvent::UpdateDebugT(t) => {
                self.dbg_drawing_state.ray_t = t;
            }
            // VgonioEvent::UpdatePrimId(prim_id) => {
            // if let Some(msurf_view) = &self.msurf {
            //     let cache = self.cache.deref().borrow();
            //     let msurf =
            // cache.micro_surface_mesh(msurf_view.mesh).unwrap();
            //     let id = prim_id as usize;
            //     if id < msurf.facets.len() {
            //         let indices = [
            //             msurf.facets[id * 3],
            //             msurf.facets[id * 3 + 1],
            //             msurf.facets[id * 3 + 2],
            //         ];
            //         let vertices = [
            //             msurf.verts[indices[0] as usize],
            //             msurf.verts[indices[1] as usize],
            //             msurf.verts[indices[2] as usize],
            //         ];
            //         log::debug!(
            //             "query prim {}: {:?}\n    p0: {:?}\n    p1:
            // {:?}\n    p2: {:?}",             prim_id,
            //             indices,
            //             vertices[0],
            //             vertices[1],
            //             vertices[2]
            //         );
            //         let mut encoder =
            // self.ctx.gpu.device.create_command_encoder(
            //             &wgpu::CommandEncoderDescriptor {
            //                 label: Some("copy-index-buffer-encoder"),
            //             },
            //         );
            //         self.ctx.gpu.queue.write_buffer(
            //             &self.debug_drawing.msurf_prim_index_buf,
            //             0,
            //             bytemuck::cast_slice(&indices),
            //         );
            //     }
            //     self.debug_drawing.drawing_msurf_prims = true;
            //     self.debug_drawing.multiple_prims = false;
            // }
            // }
            // VgonioEvent::TraceRayDbg {
            //     ray,
            //     method,
            //     max_bounces,
            // } => {
            //     log::debug!("= = = = [Debug Ray Tracing] = = = =\n  => {:?}", ray);
            //     if self.msurf.is_none() {
            //         log::warn!("No micro-surface loaded, can't trace ray!");
            //     } else {
            //         match method {
            //             #[cfg(feature = "embree")]
            //             RtcMethod::Embree => {
            //                 log::debug!("  => [Standard Ray Tracing]");
            //                 todo!();
            //                 // self.debug_drawing.rays = trace_ray_standard_dbg(
            //                 //     ray,
            //                 //     max_bounces,
            //                 //     self.surface_mesh.as_ref().unwrap(),
            //                 // );
            //             }
            //             #[cfg(feature = "optix")]
            //             RtcMethod::Optix => {
            //                 todo!()
            //             }
            //             RtcMethod::Grid => {
            //                 log::debug!("  => [Grid Ray Tracing]");
            //                 // let cache = self.cache.deref().borrow();
            //                 // let grid_rt = Grid::new(
            //                 //     cache
            //                 //         .micro_surface(self.msurf.as_ref().unwrap().surf)
            //                 //         .unwrap(),
            //                 //     cache
            //                 //         .micro_surface_mesh(self.msurf.as_ref().unwrap().mesh)
            //                 //         .unwrap(),
            //                 // );
            //                 // self.debug_drawing.rays =
            //                 //     trace_ray_grid_dbg(ray, max_bounces, &grid_rt);
            //                 todo!()
            //             }
            //         }
            //         if self.debug_drawing_enabled {
            //             let mut content = self
            //                 .debug_drawing
            //                 .rays
            //                 .iter()
            //                 .map(|r| r.org)
            //                 .collect::<Vec<_>>();
            //             let last_ray = self.debug_drawing.rays[self.debug_drawing.rays.len() -
            // 1];             content.push(last_ray.org + last_ray.dir *
            // self.debug_drawing.ray_t);             log::debug!("content: {:?}",
            // content);             self.debug_drawing.rays_vertex_count =
            //                 self.debug_drawing.rays.len() as u32 + 1;
            //             self.debug_drawing.drawing_msurf_prims = false;
            //             self.ctx.gpu.queue.write_buffer(
            //                 &self.debug_drawing.rays_vertex_buf,
            //                 0,
            //                 bytemuck::cast_slice(&content),
            //             );
            //         }
            //     }
            // }
            VgonioEvent::UpdateSamplingDebugger {
                count,
                azimuth,
                zenith,
            } => {
                let samples = measure::emitter::uniform_sampling_on_unit_sphere(
                    count as usize,
                    degrees!(zenith.0).into(),
                    degrees!(zenith.1).into(),
                    degrees!(azimuth.0).into(),
                    degrees!(azimuth.1).into(),
                    Handedness::RightHandedYUp,
                );
                let mut encoder =
                    self.ctx
                        .gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sampling Debugger Encoder"),
                        });
                self.ui
                    .tools
                    .get_tool_mut::<SamplingInspector>()
                    .unwrap()
                    .record_render_pass(&self.ctx.gpu, &mut encoder, &samples);
                self.ctx.gpu.queue.submit(Some(encoder.finish()));
            }
            // VgonioEvent::CheckVisibleFacets {
            //     m_azimuth,
            //     m_zenith,
            //     opening_angle,
            // } => match &self.msurf {
            //     None => {}
            //     Some(msurf) => {
            //         let cache = self.cache.deref().borrow();
            //         let msurf_mesh = cache.micro_surface_mesh(msurf.mesh).unwrap();
            //         let half_opening_angle_cos = (opening_angle / 2.0).cos();
            //         log::debug!(
            //             "zenith: {}, azimuth: {}, opening angle: {}, half opening angle cos: {}",
            //             m_zenith,
            //             m_azimuth,
            //             opening_angle,
            //             half_opening_angle_cos
            //         );
            //         // Right-handed Y-up
            //         let view_dir = Vec3::new(
            //             m_zenith.sin() * m_azimuth.cos(),
            //             m_zenith.cos(),
            //             m_zenith.sin() * m_azimuth.sin(),
            //         )
            //         .normalize();
            //         log::debug!("View direction: {:?}", view_dir);
            //         log::debug!("normals: {:?}", msurf_mesh.facet_normals);
            //         let visible_facets_indices = msurf_mesh
            //             .facet_normals
            //             .iter()
            //             .enumerate()
            //             .inspect(|(idx, normal)| {
            //                 log::debug!(
            //                     "facet {}: {:?}, dot {}",
            //                     idx,
            //                     normal,
            //                     normal.dot(view_dir)
            //                 );
            //             })
            //             .filter_map(|(idx, normal)| {
            //                 if normal.dot(view_dir) >= half_opening_angle_cos {
            //                     Some(idx)
            //                 } else {
            //                     None
            //                 }
            //             })
            //             .flat_map(|idx| {
            //                 [
            //                     msurf_mesh.facets[idx * 3],
            //                     msurf_mesh.facets[idx * 3 + 1],
            //                     msurf_mesh.facets[idx * 3 + 2],
            //                 ]
            //             })
            //             .collect::<Vec<_>>();
            //         log::debug!("Visible facets count: {}", visible_facets_indices.len() / 3);
            //
            //         if visible_facets_indices.len() >= 3 {
            //             // reallocate buffer if needed
            //             if visible_facets_indices.len() * std::mem::size_of::<u32>()
            //                 > self.debug_drawing.msurf_prim_index_buf.size() as usize - 3
            //             {
            //                 log::debug!(
            //                     "Reallocating visible facets index buffer to {} bytes",
            //                     visible_facets_indices.len() * std::mem::size_of::<u32>() + 3
            //                 );
            //                 self.debug_drawing.msurf_prim_index_buf =
            //                     self.ctx.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            //                         label: Some("Visible Facets Index Buffer"),
            //                         size: (visible_facets_indices.len()
            //                             * std::mem::size_of::<u32>()
            //                             + 3) as u64,
            //                         usage: wgpu::BufferUsages::COPY_DST
            //                             | wgpu::BufferUsages::INDEX
            //                             | wgpu::BufferUsages::COPY_SRC,
            //                         mapped_at_creation: false,
            //                     });
            //             }
            //             self.debug_drawing.msurf_prim_index_count =
            //                 visible_facets_indices.len() as u32;
            //             log::debug!("Updating visible facets index buffer");
            //             self.ctx.gpu.queue.write_buffer(
            //                 &self.debug_drawing.msurf_prim_index_buf,
            //                 3 * std::mem::size_of::<u32>() as u64,
            //                 bytemuck::cast_slice(&visible_facets_indices),
            //             );
            //             self.debug_drawing.drawing_msurf_prims = true;
            //             self.debug_drawing.multiple_prims = true;
            //         }
            //     }
            // },
            VgonioEvent::MeasureAreaDistribution { params, surfaces } => {
                println!("Measuring area distribution");
                measure::microfacet::measure_area_distribution(
                    params,
                    &surfaces,
                    &self.cache.read().unwrap(),
                );
                todo!("Save area distribution to file or display it in a window");
            }
            VgonioEvent::MeasureMaskingShadowing { params, surfaces } => {
                println!("Measuring masking/shadowing");
                measure::microfacet::measure_masking_shadowing(
                    params,
                    &surfaces,
                    &self.cache.read().unwrap(),
                    Handedness::RightHandedYUp,
                );
                todo!("Save area distribution to file or display it in a window");
            }
            VgonioEvent::MeasureBsdf { params, surfaces } => {
                println!("Measuring BSDF");
                measure::bsdf::measure_bsdf_rt(
                    params,
                    &surfaces,
                    params.sim_kind,
                    &self.cache.read().unwrap(),
                );
                todo!("Save area distribution to file or display it in a window");
            }
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
            Err(Error::Rhi(error)) => {
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

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MicroSurfaceUniforms {
    model: Mat4,
    info: Vec4,
}

impl MicroSurfaceUniforms {
    fn aligned_size(device: &wgpu::Device) -> u32 {
        let alignment = device.limits().min_uniform_buffer_offset_alignment;
        let size = std::mem::size_of::<MicroSurfaceUniforms>() as u32;
        let remainder = size % alignment;
        if remainder == 0 {
            size
        } else {
            size + alignment - remainder
        }
    }
}

pub enum PlottingMode {
    None,
    Adf,
    Bsdf,
    Msf,
}

pub trait Plottable {
    fn mode(&self) -> PlottingMode { PlottingMode::None }
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl Plottable for () {
    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

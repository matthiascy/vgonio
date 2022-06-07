mod camera;
mod input;

pub use input::InputState;

use crate::app::gui::{GuiContext, VgonioEvent, VgonioGui, WindowSize};
use crate::gfx::{
    GpuContext, MeshView, RdrPass, SizedBuffer, Texture, VertexLayout,
    DEFAULT_BIND_GROUP_LAYOUT_DESC,
};
use camera::CameraState;

use crate::acq::tracing::RayTracingMethod;
use crate::acq::{MicroSurfaceView, OcclusionEstimationPass};
use crate::app::VgonioConfig;
use crate::error::Error;
use crate::gfx::camera::{Camera, Projection, ProjectionKind};
use crate::htfld::Heightfield;
use glam::{Mat4, Vec3};
use std::{
    collections::HashMap,
    default::Default,
    io::{BufWriter, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    time::Instant
};
use wgpu::util::DeviceExt;
use wgpu::{VertexFormat, VertexStepMode};
use winit::event::{KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoopProxy;

const AZIMUTH_BIN_SIZE_DEG: usize = 5;
const ZENITH_BIN_SIZE_DEG: usize = 2;
const AZIMUTH_BIN_SIZE_RAD: f32 = (AZIMUTH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
const ZENITH_BIN_SIZE_RAD: f32 = (ZENITH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
const NUM_AZIMUTH_BINS: usize = ((2.0 * std::f32::consts::PI) / AZIMUTH_BIN_SIZE_RAD) as _;
const NUM_ZENITH_BINS: usize = ((0.5 * std::f32::consts::PI) / ZENITH_BIN_SIZE_RAD) as _;

// TODO: fix blending.

/// Stores the content of depth buffer.
struct DepthMap {
    depth_attachment: Texture,
    depth_attachment_storage: wgpu::Buffer, // used to store depth attachment
}

impl DepthMap {
    pub fn new(ctx: &GpuContext) -> Self {
        let depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            ctx.surface_config.width,
            ctx.surface_config.height,
            None,
            Some("depth-texture"),
        );
        let depth_attachment_storage_size = (std::mem::size_of::<f32>()
            * (ctx.surface_config.width * ctx.surface_config.height) as usize)
            as wgpu::BufferAddress;
        let depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            depth_attachment,
            depth_attachment_storage,
        }
    }

    pub fn resize(&mut self, ctx: &GpuContext) {
        self.depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            ctx.surface_config.width,
            ctx.surface_config.height,
            None,
            Some("depth-texture"),
        );
        let depth_map_buffer_size = (std::mem::size_of::<f32>()
            * (ctx.surface_config.width * ctx.surface_config.height) as usize)
            as wgpu::BufferAddress;
        self.depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_map_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    }

    pub fn copy_to_buffer(&mut self, ctx: &GpuContext) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                // texture: &self.depth_attachment.raw,
                texture: &self.depth_attachment.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                // buffer: &self.depth_attachment_storage,
                buffer: &self.depth_attachment_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(
                        std::mem::size_of::<f32>() as u32 * ctx.surface_config.width,
                    ),
                    rows_per_image: NonZeroU32::new(ctx.surface_config.height),
                },
            },
            wgpu::Extent3d {
                width: ctx.surface_config.width,
                height: ctx.surface_config.height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Save current depth buffer content to a PNG file.
    pub fn save_to_image(&mut self, ctx: &GpuContext, path: &Path) {
        let mut image = image::GrayImage::new(ctx.surface_config.width, ctx.surface_config.height);
        {
            let buffer_slice = self.depth_attachment_storage.slice(..);

            let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
            ctx.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async { mapping.await.unwrap() });

            let buffer_view_f32 = buffer_slice.get_mapped_range();
            let data_u8 = unsafe {
                let (_, data, _) = buffer_view_f32.align_to::<f32>();
                data.iter()
                    .map(|d| (remap_depth(*d, 0.1, 100.0) * 255.0) as u8)
                    .collect::<Vec<u8>>()
            };

            image.copy_from_slice(&data_u8);
            image.save(path).unwrap();
        }
        self.depth_attachment_storage.unmap();
    }
}

struct DebugDrawing {
    /// Vertex buffer storing all vertices.
    pub vert_buffer: SizedBuffer,

    pub vert_count: u32,

    pub rays: Vec<embree::Ray>,

    /// Render pass for rays drawing.
    pub render_pass_rd: RdrPass,

    pub ray_t: f32,
}

impl DebugDrawing {
    pub fn new(ctx: &GpuContext) -> Self {
        let vert_layout = VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None);
        let vert_buffer_layout = vert_layout.buffer_layout(wgpu::VertexStepMode::Vertex);
        let shader_module = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("debug-drawing-rays-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../assets/shaders/wgsl/rays.wgsl").into(),
                ),
            });
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("debug-drawing-rays-bind-group-layout"),
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
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug-drawing-rays-uniform-buffer"),
                contents: bytemuck::cast_slice(&[0.0f32; 16 * 3 + 4]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            vert_buffer: SizedBuffer::new(
                &ctx.device,
                std::mem::size_of::<f32>() * 1024, // initial capacity of 1024 rays
                wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                Some("debug-rays-vert-buf"),
            ),
            vert_count: 0,
            rays: vec![],
            render_pass_rd: RdrPass {
                pipeline: ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("debug-rays-pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: Some("debug-rays-pipeline-layout"),
                                bind_group_layouts: &[&bind_group_layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        vertex: wgpu::VertexState {
                            module: &shader_module,
                            entry_point: "vs_main",
                            buffers: &[vert_buffer_layout],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::LineStrip,
                            strip_index_format: None,
                            front_face: wgpu::FrontFace::Ccw,
                            cull_mode: Some(wgpu::Face::Back),
                            unclipped_depth: false,
                            polygon_mode: wgpu::PolygonMode::Line,
                            conservative: false,
                        },
                        depth_stencil: None,
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &shader_module,
                            entry_point: "fs_main",
                            targets: &[wgpu::ColorTargetState {
                                format: ctx.surface_config.format,
                                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            }],
                        }),
                        multiview: None,
                    }),
                bind_groups: vec![ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("debug-rays-bind-group"),
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                })],
                uniform_buffer: Some(uniform_buffer),
            },
            ray_t: 10.0
        }
    }

    pub fn update_uniform_buffer(&mut self, ctx: &GpuContext, uniform: &[f32; 16 * 3 + 4]) {
        ctx.queue.write_buffer(
            self.render_pass_rd.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(uniform),
        );
    }
}

pub struct VgonioApp {
    gpu_ctx: GpuContext,
    gui_ctx: GuiContext,
    gui: VgonioGui,
    input: InputState,
    camera: CameraState,
    passes: HashMap<&'static str, RdrPass>,
    depth_map: DepthMap,
    debug_drawing: DebugDrawing,
    debug_drawing_enabled: bool,

    // Shadow pass used for measurement of shadowing and masking function.
    //shadow_pass: ShadowPass,
    occlusion_estimation_pass: OcclusionEstimationPass,

    heightfield: Option<Box<Heightfield>>,
    heightfield_mesh_view: Option<MeshView>,
    heightfield_micro_view: Option<MicroSurfaceView>,
    surface_scale_factor: f32,

    opened_surfaces: Vec<PathBuf>,

    pub start_time: Instant,
    pub prev_frame_time: Option<f32>,

    //pub demo_ui: egui_demo_lib::DemoWindows,
    pub visual_grid_enabled: bool,
    pub surface_visible: bool,
}

impl VgonioApp {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(
        config: VgonioConfig,
        window: &winit::window::Window,
        event_loop: EventLoopProxy<VgonioEvent>,
    ) -> Result<Self, Error> {
        let gpu_ctx = GpuContext::new(window).await;
        let depth_map = DepthMap::new(&gpu_ctx);
        // Camera
        let camera = {
            let camera = Camera::new(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
            let projection = Projection::new(
                0.1,
                100.0,
                60.0f32.to_radians(),
                gpu_ctx.surface_config.width,
                gpu_ctx.surface_config.height,
            );
            CameraState::new(camera, projection, ProjectionKind::Perspective)
        };

        let heightfield_pass = create_heightfield_pass(&gpu_ctx);
        let visual_grid_pass = create_visual_grid_pass(&gpu_ctx);

        let mut passes = HashMap::new();
        passes.insert("visual_grid", visual_grid_pass);
        passes.insert("heightfield", heightfield_pass);

        // let shadow_pass = ShadowPass::new(
        //     &gpu_ctx,
        //     gpu_ctx.surface_config.width,
        //     gpu_ctx.surface_config.height,
        //     true,
        //     true,
        // );

        let gui_ctx = GuiContext::new(window, &gpu_ctx.device, gpu_ctx.surface_config.format, 1);

        //let ui = egui_demo_lib::WrapApp::default();
        let gui = VgonioGui::new(event_loop, config);

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        let occlusion_estimation_pass = OcclusionEstimationPass::new(&gpu_ctx, 512, 512);
        let debug_drawing = DebugDrawing::new(&gpu_ctx);

        Ok(Self {
            gpu_ctx,
            gui_ctx,
            gui,
            input,
            passes,
            depth_map,
            // shadow_pass,
            debug_drawing,
            debug_drawing_enabled: true,
            occlusion_estimation_pass,
            heightfield: None,
            heightfield_mesh_view: None,
            heightfield_micro_view: None,
            camera,
            start_time: Instant::now(),
            prev_frame_time: None,
            // demo_ui: ui,
            visual_grid_enabled: true,
            surface_scale_factor: 1.0,
            opened_surfaces: vec![],
            surface_visible: true
        })
    }

    #[inline]
    pub fn surface_width(&self) -> u32 {
        self.gpu_ctx.surface_config.width
    }

    #[inline]
    pub fn surface_height(&self) -> u32 {
        self.gpu_ctx.surface_config.height
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.gpu_ctx.surface_config.width = new_size.width;
            self.gpu_ctx.surface_config.height = new_size.height;
            self.gpu_ctx
                .surface
                .configure(&self.gpu_ctx.device, &self.gpu_ctx.surface_config);
            self.depth_map.resize(&self.gpu_ctx);
            self.camera
                .projection
                .resize(new_size.width, new_size.height);

            // self.shadow_pass
            //     .resize(&self.gpu_ctx.device, new_size.width,
            // new_size.height);
        }
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        if !self.gui_ctx.handle_event(event) {
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
                    if self.input.is_key_pressed(VirtualKeyCode::K) {
                        println!("Measuring geometric term ...");
                        let now = Instant::now();
                        self.measure_micro_surface_geometric_term();
                        println!(
                            "Geometric term measurement finished in {} secs",
                            (Instant::now() - now).as_secs_f32()
                        );
                    }
                    true
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    self.input.update_scroll_delta(*delta);
                    true
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    self.input.update_mouse_map(*button, *state);
                    true
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.input.update_cursor_delta((*position).cast::<f32>());
                    true
                }
                _ => false,
            }
        } else {
            true
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        // Update camera uniform.
        self.camera
            .update(&self.input, dt, ProjectionKind::Perspective);
        self.gui.update_gizmo_matrices(
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

        // self.shadow_pass
        //     .update_uniforms(&self.gpu_ctx.queue, Mat4::IDENTITY, view, proj);

        if let Some(hf) = &self.heightfield {
            let mut uniform = [0.0f32; 16 * 3 + 4];
            // Displace visually the heightfield.
            uniform[0..16].copy_from_slice(
                &Mat4::from_translation(Vec3::new(
                    0.0,
                    -(hf.max + hf.min) * 0.5 * self.surface_scale_factor,
                    0.0,
                ))
                .to_cols_array(),
            );
            uniform[16..32].copy_from_slice(&self.camera.uniform.view_matrix.to_cols_array());
            uniform[32..48].copy_from_slice(&self.camera.uniform.proj_matrix.to_cols_array());
            uniform[48..52].copy_from_slice(&[
                hf.min,
                hf.max,
                hf.max - hf.min,
                self.surface_scale_factor,
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

            // Update the uniform buffer for debug drawing.
            if self.debug_drawing_enabled {
                self.debug_drawing
                    .update_uniform_buffer(&self.gpu_ctx, &uniform);
            }
        }

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];
    }

    pub fn render(&mut self, window: &winit::window::Window) -> Result<(), wgpu::SurfaceError> {
        let output_frame = self.gpu_ctx.surface.get_current_texture()?;
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
            self.gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_ui_render_encoder"),
                }),
        ];

        if self.gui.current_workspace_name() == "Simulation" {
            let mut render_pass = encoders[0].begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &output_view,
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
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    // view: &self.depth_attachment.view,
                    view: &self.depth_map.depth_attachment.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            if self.surface_visible {
                if let Some(mesh) = &self.heightfield_mesh_view {
                    let pass = self.passes.get("heightfield").unwrap();
                    render_pass.set_pipeline(&pass.pipeline);
                    render_pass.set_bind_group(0, &pass.bind_groups[0], &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);
                    render_pass.draw_indexed(0..mesh.indices_count, 0, 0..1);
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
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.debug_drawing.render_pass_rd.pipeline);
            render_pass.set_bind_group(0, &self.debug_drawing.render_pass_rd.bind_groups[0], &[]);
            render_pass.set_vertex_buffer(0, self.debug_drawing.vert_buffer.raw.slice(..));
            render_pass.draw(0..self.debug_drawing.vert_count, 0..1);
        }
        {
            // Record UI
            let ui_start_time = Instant::now();
            self.gui_ctx.take_input(window);
            self.gui_ctx.begin_frame();
            // self.demo_ui.ui(self.gui_ctx.egui_context());
            self.gui.show(&self.gui_ctx);
            let ui_output_frame = self.gui_ctx.egui_context().end_frame();
            let meshes = self
                .gui_ctx
                .egui_context()
                .tessellate(ui_output_frame.shapes);
            let frame_time = (Instant::now() - ui_start_time).as_secs_f64() as f32;
            self.prev_frame_time = Some(frame_time);
            let win_size = WindowSize {
                physical_width: self.gpu_ctx.surface_config.width,
                physical_height: self.gpu_ctx.surface_config.height,
                scale_factor: window.scale_factor() as f32,
            };
            self.gui_ctx
                .update_textures(
                    &self.gpu_ctx.device,
                    &self.gpu_ctx.queue,
                    ui_output_frame.textures_delta,
                )
                .unwrap();
            self.gui_ctx.update_buffers(
                &self.gpu_ctx.device,
                &self.gpu_ctx.queue,
                &meshes,
                &win_size,
            );
            self.gui_ctx
                .render(&mut encoders[2], &output_view, &meshes, &win_size, None)
                .unwrap();
        }

        self.gpu_ctx.queue.submit(encoders.map(|enc| enc.finish()));

        self.depth_map.copy_to_buffer(&self.gpu_ctx);

        output_frame.present();

        Ok(())
    }

    pub fn handle_user_event(&mut self, event: VgonioEvent) {
        match event {
            VgonioEvent::RequestRedraw => {}
            VgonioEvent::OpenFile(path) => {
                self.load_height_field(path.as_path());
                if !self.opened_surfaces.contains(&path) {
                    self.opened_surfaces.push(path);
                }
                self.gui
                    .workspaces
                    .simulation
                    .update_surface_list(&self.opened_surfaces);
            }
            VgonioEvent::ToggleGrid => {
                self.visual_grid_enabled = !self.visual_grid_enabled;
            }
            VgonioEvent::UpdateDepthMap => {
                self.depth_map.copy_to_buffer(&self.gpu_ctx);
                self.gui
                    .workspaces
                    .simulation
                    .visual_debug_tool
                    .shadow_map_pane
                    .update_depth_map(&self.gpu_ctx, &self.depth_map.depth_attachment_storage);
            }
            VgonioEvent::UpdateSurfaceScaleFactor(factor) => {
                self.surface_scale_factor = factor;
            }
            VgonioEvent::UpdateDebugT(t) => {
                self.debug_drawing.ray_t = t;
                // let content = self.debug_drawing.rays.iter().flat_map(|r| {
                //     [r.org_x, r.org_y, r.org_z,
                //         r.org_x + self.debug_drawing.ray_t * r.dir_x,
                //         r.org_y + self.debug_drawing.ray_t * r.dir_y,
                //         r.org_z + self.debug_drawing.ray_t * r.dir_z]
                // }).collect::<Vec<_>>();
                // self.debug_drawing.vert_count = (self.debug_drawing.rays.len() * 2) as u32;
                // self.gpu_ctx.queue.write_buffer(
                //     &self.debug_drawing.vert_buffer.raw,
                //     0,
                //     bytemuck::cast_slice(&content),
                // );
            }
            VgonioEvent::TraceRayDbg {
                ray,
                method,
                max_bounces,
            } => {
                if self.heightfield.is_none() {
                    log::warn!("No heightfield loaded, can't trace ray!");
                } else {
                    match method {
                        RayTracingMethod::Standard => {
                            self.debug_drawing.rays = crate::acq::tracing::trace_ray_standard(
                                ray,
                                max_bounces,
                                self.heightfield.as_ref().unwrap(),
                            );
                            if self.debug_drawing_enabled {
                                let mut content = vec![0.0f32; (self.debug_drawing.rays.len() + 1) * 3];
                                for (i, r) in self.debug_drawing.rays.iter().enumerate() {
                                    content[i * 3 + 0] = r.org_x;
                                    content[i * 3 + 1] = r.org_y;
                                    content[i * 3 + 2] = r.org_z;
                                    if i == self.debug_drawing.rays.len() - 1 {
                                        content[i * 3 + 3] = r.org_x + self.debug_drawing.ray_t * r.dir_x;
                                        content[i * 3 + 4] = r.org_y + self.debug_drawing.ray_t * r.dir_y;
                                        content[i * 3 + 5] = r.org_z + self.debug_drawing.ray_t * r.dir_z;
                                    }
                                }
                                println!("rays: {:?}", content);
                                self.debug_drawing.vert_count = self.debug_drawing.rays.len() as u32 + 1;
                                self.gpu_ctx.queue.write_buffer(
                                    &self.debug_drawing.vert_buffer.raw,
                                    0,
                                    bytemuck::cast_slice(&content),
                                );
                            }
                        }
                        RayTracingMethod::Grid => {
                            crate::acq::tracing::trace_ray_grid(
                                ray,
                                self.heightfield.as_ref().unwrap(),
                            );
                        }
                        RayTracingMethod::Hybrid => {
                            crate::acq::tracing::trace_ray_hybrid(
                                ray,
                                self.heightfield.as_ref().unwrap(),
                            );
                        }
                    }
                }
            }
            VgonioEvent::ToggleDebugDrawing => {
                self.debug_drawing_enabled = !self.debug_drawing_enabled;
                println!("Debug drawing: {}", self.debug_drawing_enabled);
            }
            VgonioEvent::ToggleSurfaceVisibility => {
                self.surface_visible = !self.surface_visible;
            }
            _ => {}
        }
    }

    fn load_height_field(&mut self, path: &Path) {
        match Heightfield::read_from_file(path, None, None) {
            Ok(mut hf) => {
                log::info!(
                    "Heightfield loaded: {}, rows = {}, cols = {}",
                    path.display(),
                    hf.rows,
                    hf.cols
                );
                hf.fill_holes();
                let mesh_view = MeshView::from_height_field(&self.gpu_ctx.device, &hf);
                let micro_view = MicroSurfaceView::from_height_field(&self.gpu_ctx.device, &hf);
                self.heightfield = Some(Box::new(hf));
                self.heightfield_mesh_view = Some(mesh_view);
                self.heightfield_micro_view = Some(micro_view);
            }
            Err(err) => {
                log::error!("HeightField loading error: {}", err);
            }
        }
    }

    /// Measure the geometric masking/shadowing function of a micro-surface.
    fn measure_micro_surface_geometric_term(&mut self) {
        if let Some(mesh) = &self.heightfield_micro_view {
            let radius = (mesh.extent.max - mesh.extent.min).max_element();
            let near = 0.1f32;
            let far = radius * 2.0;

            // let cs_module = self.gpu_ctx.device.create_shader_module(
            //     &wgpu::include_spirv!("../assets/shaders/spirv/visibility.comp.spv"),
            // );
            //
            // let faces_visibility = vec![0u32; mesh.faces.len()];
            // let faces_visibility_buffer = self
            //     .gpu_ctx
            //     .device
            //     .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //         label: None,
            //         contents: faces_visibility.as_bytes(),
            //         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            //     });
            //
            // let normals_buffer = self
            //     .gpu_ctx
            //     .device
            //     .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //         label: None,
            //         contents: bytemuck::cast_slice(&mesh.facet_normals),
            //         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            //     });
            //
            // let uniform_buffer =
            // self.gpu_ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            //     label: None,
            //     contents: bytemuck::cast_slice(&[0.0f32; 4]),
            //     usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            // });

            let proj = {
                let projection = Projection::new(
                    near,
                    far,
                    70.0f32.to_radians(),
                    (radius * 1.414) as u32,
                    (radius * 1.414) as u32,
                );
                projection.matrix(ProjectionKind::Orthographic)
            };

            let file = std::fs::File::create("measured_geometric_term_v2.txt").unwrap();
            let writer = &mut BufWriter::new(file);
            writer.write_all("phi theta ratio\n".as_bytes()).unwrap();

            // self.shadow_pass.resize(&self.gpu_ctx.device, 512, 512);

            for i in 0..NUM_AZIMUTH_BINS {
                for j in 0..NUM_ZENITH_BINS {
                    // Camera is located at the center of each bin.
                    let phi = ((2 * i + 1) as f32) * AZIMUTH_BIN_SIZE_RAD * 0.5; // azimuth
                    let theta = ((2 * j + 1) as f32) * ZENITH_BIN_SIZE_RAD * 0.5; // zenith
                    let (sin_theta, cos_theta) = theta.sin_cos();
                    let (sin_phi, cos_phi) = phi.sin_cos();
                    let view_pos = Vec3::new(
                        radius * sin_theta * cos_phi,
                        radius * cos_theta,
                        radius * sin_theta * sin_phi,
                    );
                    let view_dir = view_pos.normalize();
                    let camera = Camera::new(view_pos, Vec3::ZERO, Vec3::Y);
                    let visible_facets = mesh
                        .facets
                        .iter()
                        .enumerate()
                        .filter_map(|(i, f)| {
                            if mesh.facet_normals[i].dot(view_dir) > 0.0 {
                                Some(*f)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    let ratio = if visible_facets.is_empty() {
                        0.0
                    } else {
                        self.occlusion_estimation_pass.update_uniforms(
                            &self.gpu_ctx.queue,
                            Mat4::IDENTITY,
                            proj * camera.matrix(),
                        );

                        let index_buffer = &self.gpu_ctx.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("measurement_vertex_buffer"),
                                contents: bytemuck::cast_slice(&visible_facets),
                                usage: wgpu::BufferUsages::INDEX,
                            },
                        );

                        let indices_count = visible_facets.len() as u32 * 3;

                        self.occlusion_estimation_pass.run_once(
                            &self.gpu_ctx.device,
                            &self.gpu_ctx.queue,
                            &mesh.vertex_buffer,
                            index_buffer,
                            indices_count,
                            MicroSurfaceView::INDEX_FORMAT,
                        );

                        // self.occlusion_estimation_pass
                        //     .save_depth_attachment(
                        //         &self.gpu_ctx.device,
                        //         near,
                        //         far,
                        //         format!(
                        //             "occ_pass_depth_{:.2}_{:.2}.png",
                        //             phi.to_degrees(),
                        //             theta.to_degrees()
                        //         )
                        //         .as_ref(),
                        //     )
                        //     .unwrap();
                        //
                        // self.occlusion_estimation_pass
                        //     .save_color_attachment(
                        //         &self.gpu_ctx.device,
                        //         format!(
                        //             "occ_pass_color_{:.2}_{:.2}.png",
                        //             phi.to_degrees(),
                        //             theta.to_degrees()
                        //         )
                        //         .as_ref(),
                        //     )
                        //     .unwrap();

                        self.occlusion_estimation_pass
                            .calculate_ratio(&self.gpu_ctx.device)
                    };

                    writer
                        .write_all(
                            format!(
                                "{:<6.2} {:<5.2} {:.6}\n",
                                phi.to_degrees(),
                                theta.to_degrees(),
                                ratio
                            )
                            .as_bytes(),
                        )
                        .unwrap();
                }
            }
        }
    }
}

fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    (2.0 * near * far) / (far + near - depth * (far - near))
}

pub fn remap_depth(depth: f32, near: f32, far: f32) -> f32 {
    linearize_depth(depth, near, far) / (far - near)
}

fn create_heightfield_pass(ctx: &GpuContext) -> RdrPass {
    // Load shader
    let shader_module = ctx
        .device
        .create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("height_field_shader_module"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/wgsl/heightfield.wgsl").into(),
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
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: VertexFormat::Float32x3,
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
                targets: &[wgpu::ColorTargetState {
                    format: ctx.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

    RdrPass {
        pipeline,
        bind_groups: vec![bind_group],
        uniform_buffer: Some(uniform_buffer),
    }
}

fn create_visual_grid_pass(ctx: &GpuContext) -> RdrPass {
    let grid_vert_shader = ctx.device.create_shader_module(&wgpu::include_spirv!(
        "../assets/shaders/spirv/grid.vert.spv"
    ));
    let grid_frag_shader = ctx.device.create_shader_module(&wgpu::include_spirv!(
        "../assets/shaders/spirv/grid.frag.spv"
    ));
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
            contents: bytemuck::bytes_of(&crate::gfx::VisualGridUniforms::default()),
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
                module: &grid_vert_shader,
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
                module: &grid_frag_shader,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: ctx.surface_config.format,
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
                }],
            }),
            multiview: None,
        });

    RdrPass {
        pipeline,
        bind_groups: vec![bind_group],
        uniform_buffer: Some(uniform_buffer),
    }
}

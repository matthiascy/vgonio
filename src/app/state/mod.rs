mod camera;
mod input;

pub use input::InputState;

use crate::app::gfx::{GpuContext, Texture, Vertex, VertexLayout};
use crate::app::gui::{GuiContext, UserEvent, VgonioGui, WindowSize};
use camera::CameraState;

use crate::app::gfx::camera::{Camera, Projection};
use crate::error::Error;
use epi::App;
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::default::Default;
use std::time::Instant;
use wgpu::util::DeviceExt;
use wgpu::{include_spirv, VertexFormat};
use winit::event::{KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoopProxy;
use winit::window::Window;

const NUM_INSTANCES_PER_ROW: u32 = 9;
const NUM_INSTANCES_PER_COL: u32 = 9;

// TODO: fix blending.

pub struct VgonioApp {
    gpu_ctx: GpuContext,
    gui_ctx: GuiContext,
    gui: VgonioGui,
    input: InputState,
    camera: CameraState,

    graphics_pipelines: HashMap<&'static str, wgpu::RenderPipeline>,

    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_vertices: u32,
    pub num_indices: u32,
    pub texture_bind_groups: [wgpu::BindGroup; 2],
    pub current_texture_index: usize,

    pub object_model_matrix: glam::Mat4,
    pub instancing_buffer: wgpu::Buffer,
    pub instancing_transforms: Vec<Mat4>,
    pub depth_texture: Texture,
    pub start_time: Instant,
    pub prev_frame_time: Option<f32>,

    pub demo_ui: egui_demo_lib::WrapApp,
}

/// Repaint signal type that egui needs for requesting a repaint from another
/// thread. It sends the custom RequestRedraw event to the winit event loop.
pub struct RepaintSignal(pub(crate) std::sync::Mutex<winit::event_loop::EventLoopProxy<UserEvent>>);

impl epi::backend::RepaintSignal for RepaintSignal {
    fn request_repaint(&self) {
        self.0
            .lock()
            .unwrap()
            .send_event(UserEvent::RequestRedraw)
            .ok();
    }
}

impl VgonioApp {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(
        window: &Window,
        event_loop: EventLoopProxy<UserEvent>,
    ) -> Result<Self, Error> {
        let gpu_ctx = GpuContext::new(window).await;
        let num_vertices = VERTICES.len() as u32;
        // Create texture
        let sampler =
            std::sync::Arc::new(gpu_ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        let textures = [
            Texture::create_from_bytes(
                &gpu_ctx.device,
                &gpu_ctx.queue,
                include_bytes!("../assets/damascus001.jpg"),
                sampler.clone(),
                Some("damascus-texture-001"),
            ),
            Texture::create_from_bytes(
                &gpu_ctx.device,
                &gpu_ctx.queue,
                include_bytes!("../assets/damascus002.jpg"),
                sampler,
                Some("damascus-texture-002"),
            ),
        ];

        let depth_texture = Texture::create_depth_texture(
            &gpu_ctx.device,
            &gpu_ctx.surface_config,
            "depth-texture",
        );

        // Descriptor Sets
        // [`BindGroup`] describes a set of resources and how they can be accessed by a
        // shader.
        let texture_bind_group_layout =
            gpu_ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("texture-bind-group-layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                // SamplerBindingType::Comparison is only for
                                // TextureSampleType::Depth
                                // SamplerBindingType::Filtering if the sample_type of the texture
                                // is:     TextureSampleType::Float
                                // { filterable: true }
                                // Otherwise you'll get an error.
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            count: None,
                        },
                    ],
                });

        let texture_bind_groups = [
            gpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("diffuse-texture-bind-group-001"),
                    layout: &texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&textures[0].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&textures[0].sampler),
                        },
                    ],
                }),
            gpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("diffuse-texture-bind-group-002"),
                    layout: &texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&textures[1].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&textures[1].sampler),
                        },
                    ],
                }),
        ];

        // Camera
        let object_model_matrix = glam::Mat4::IDENTITY;
        let camera = {
            let camera = Camera::new(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
            let projection = Projection::new(
                0.1,
                100.0,
                45.0f32.to_radians(),
                gpu_ctx.surface_config.width,
                gpu_ctx.surface_config.height,
            );
            CameraState::new(&gpu_ctx.device, camera, projection)
        };

        let instance_displacement = Vec3::new(
            NUM_INSTANCES_PER_ROW as f32 * 0.5,
            NUM_INSTANCES_PER_COL as f32 * 0.5,
            0.0,
        );
        // Instancing
        let instancing_transforms = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|u| {
                (0..NUM_INSTANCES_PER_COL).map(move |v| {
                    let pos = Vec3::new(u as f32, v as f32, 0.0) - instance_displacement;
                    let rot = if pos.abs_diff_eq(Vec3::ZERO, 0.0001f32) {
                        Quat::from_axis_angle(Vec3::Z, 0.0)
                    } else {
                        Quat::from_axis_angle(pos.normalize(), 45.0f32.to_radians())
                    };
                    Mat4::from_translation(pos) * Mat4::from_quat(rot)
                })
            })
            .collect::<Vec<_>>();
        let instancing_buffer =
            gpu_ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("object-instancing-buffer"),
                    contents: bytemuck::cast_slice(&instancing_transforms),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

        let shader = gpu_ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("default-vertex-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../assets/shaders/wgsl/shader.wgsl").into(),
                ),
            });

        let grid_vert_shader = gpu_ctx
            .device
            .create_shader_module(&include_spirv!("../assets/shaders/spirv/grid.vert.spv"));

        let grid_frag_shader = gpu_ctx
            .device
            .create_shader_module(&include_spirv!("../assets/shaders/spirv/grid.frag.spv"));

        // Pipeline layout
        let default_render_pipeline_layout =
            gpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("default-pipeline-layout"),
                    bind_group_layouts: &[&texture_bind_group_layout, &camera.bind_group_layout],
                    push_constant_ranges: &[],
                });

        let grid_graphics_pipeline_layout =
            gpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("grid_graphics_pipeline_layout"),
                    bind_group_layouts: &[&camera.bind_group_layout],
                    push_constant_ranges: &[],
                });

        let instancing_vertex_layout = VertexLayout::new(
            &[
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
                VertexFormat::Float32x4,
            ],
            Some(5),
        );

        let default_render_pipeline =
            gpu_ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&default_render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[
                            Vertex::layout().buffer_layout(wgpu::VertexStepMode::Vertex),
                            instancing_vertex_layout.buffer_layout(wgpu::VertexStepMode::Instance),
                        ],
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
                        module: &shader,
                        entry_point: "fs_main",
                        // targets: &[gpu_ctx.surface_config.format.into()],
                        targets: &[wgpu::ColorTargetState {
                            format: gpu_ctx.surface_config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        }],
                    }),
                    multiview: None,
                });

        let grid_render_pipeline =
            gpu_ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Grid Render Pipeline"),
                    layout: Some(&grid_graphics_pipeline_layout),
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
                            format: gpu_ctx.surface_config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            // blend: Some(wgpu::BlendState {
                            //     color: wgpu::BlendComponent {
                            //         src_factor: wgpu::BlendFactor::One,
                            //         dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            //         operation: wgpu::BlendOperation::Add,
                            //     },
                            //     alpha: wgpu::BlendComponent {
                            //         src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            //         dst_factor: wgpu::BlendFactor::One,
                            //         operation: wgpu::BlendOperation::Add,
                            //     },
                            // }),
                            write_mask: wgpu::ColorWrites::ALL,
                        }],
                    }),
                    multiview: None,
                });

        let mut graphics_pipelines = HashMap::new();
        graphics_pipelines.insert("default", default_render_pipeline);
        graphics_pipelines.insert("grid", grid_render_pipeline);

        let vertex_buffer = gpu_ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex-buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = gpu_ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index-buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            });

        let num_indices = INDICES.len() as u32;

        let gui_ctx = GuiContext::new(window, &gpu_ctx.device, gpu_ctx.surface_config.format, 1);
        let ui = egui_demo_lib::WrapApp::default();
        let gui = VgonioGui::new(event_loop);

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        Ok(Self {
            gpu_ctx,
            gui_ctx,
            graphics_pipelines,
            vertex_buffer,
            index_buffer,
            num_vertices,
            num_indices,
            texture_bind_groups,
            current_texture_index: 0,
            camera,
            object_model_matrix,
            instancing_buffer,
            instancing_transforms,
            depth_texture,
            start_time: Instant::now(),
            prev_frame_time: None,
            // gui_ctx,
            demo_ui: ui,
            gui,
            input,
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
            self.depth_texture = Texture::create_depth_texture(
                &self.gpu_ctx.device,
                &self.gpu_ctx.surface_config,
                "depth_texture",
            );
            self.camera
                .projection
                .resize(new_size.width, new_size.height);
        }
    }

    pub fn collect_input(&mut self, event: &WindowEvent) -> bool {
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
        self.camera.update(&self.input, dt);
        self.gui.update_gizmo_matrices(
            Mat4::IDENTITY,
            Mat4::look_at_rh(self.camera.camera.eye, Vec3::ZERO, self.camera.camera.up),
            Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
        );

        if self.input.is_key_pressed(VirtualKeyCode::Space) {
            self.current_texture_index += 1;
            self.current_texture_index %= 2;
        }

        self.gpu_ctx.queue.write_buffer(
            &self.camera.uniform_buffer,
            0,
            bytemuck::cast_slice(&[
                self.object_model_matrix,
                self.camera.uniform.view_matrix,
                self.camera.uniform.proj_matrix,
                self.camera.uniform.view_inv_matrix,
                self.camera.uniform.proj_inv_matrix,
            ]),
        );

        for m in &mut self.instancing_transforms {
            *m *= Mat4::from_rotation_x(0.2f32.to_radians());
        }

        self.gpu_ctx.queue.write_buffer(
            &self.instancing_buffer,
            0,
            bytemuck::cast_slice(&self.instancing_transforms),
        );

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];
    }

    pub fn render(
        &mut self,
        window: &winit::window::Window,
        repaint_signal: std::sync::Arc<RepaintSignal>,
    ) -> Result<(), wgpu::SurfaceError> {
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
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(self.graphics_pipelines.get("default").unwrap());
            render_pass.set_bind_group(
                0,
                &self.texture_bind_groups[self.current_texture_index],
                &[],
            );
            render_pass.set_bind_group(1, &self.camera.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instancing_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.num_indices,
                0,
                0..self.instancing_transforms.len() as _,
            );

            render_pass.set_pipeline(self.graphics_pipelines.get("grid").unwrap());
            render_pass.set_bind_group(0, &self.camera.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        {
            // Record UI
            let ui_start_time = Instant::now();
            let input = self.gui_ctx.egui_state_mut().take_egui_input(window);

            self.gui_ctx.egui_context_mut().begin_frame(input);
            let egui_output = epi::backend::AppOutput::default();
            let ui_frame = epi::Frame::new(epi::backend::FrameData {
                info: epi::IntegrationInfo {
                    name: "vgonio-gui",
                    web_info: None,
                    prefer_dark_mode: None,
                    cpu_usage: self.prev_frame_time,
                    native_pixels_per_point: Some(window.scale_factor() as f32),
                },
                output: egui_output,
                repaint_signal,
            });
            // self.ui.update(self.ui_state.egui_context(), &ui_frame);
            self.gui.update(self.gui_ctx.egui_context(), &ui_frame);
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
                .render(&mut encoders[1], &output_view, &meshes, &win_size, None)
                .unwrap();
        }

        self.gpu_ctx.queue.submit(encoders.map(|enc| enc.finish()));

        output_frame.present();

        Ok(())
    }

    pub fn handle_event(&mut self, event: UserEvent) {
        match event {
            UserEvent::RequestRedraw => {}
            UserEvent::OpenFile(filepath) => {
                println!("____________________________________{:?}", filepath);
            }
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.4131759, 0.99240386],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.0048659444, 0.56958647],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.28081453, 0.05060294],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.85967, 0.1526709],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.9414737, 0.7347359],
    }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];

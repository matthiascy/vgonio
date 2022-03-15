use crate::app::ui::VgonioUi;
use crate::app::{texture::Texture, ui};
use crate::error::Error;
use epi::App;
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::default::Default;
use std::time::Instant;
use wgpu::{util::DeviceExt, VertexFormat};
use winit::event::{DeviceEvent, ModifiersState, MouseButton, MouseScrollDelta};
use winit::{
    event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent},
    window::Window,
};

pub(crate) mod gfx_state;
pub(crate) mod input;
pub(crate) mod ui_state;

use crate::app::gfx::camera::{Camera, CameraController, CameraUniform, OrbitControls, Projection};
use crate::app::state::input::InputState;
use gfx_state::GpuContext;
use ui_state::UiState;

const NUM_INSTANCES_PER_ROW: u32 = 9;
const NUM_INSTANCES_PER_COL: u32 = 9;

pub struct VgonioApp {
    pub gpu: GpuContext,
    pub ui_state: UiState,
    pub gui: VgonioUi,

    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_vertices: u32,
    pub num_indices: u32,
    pub texture_bind_groups: [wgpu::BindGroup; 2],
    pub current_texture_index: usize,

    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    pub camera_uniform_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_controller: OrbitControls,
    pub projection: Projection,

    pub object_model_matrix: glam::Mat4,
    pub instancing_buffer: wgpu::Buffer,
    pub instancing_transforms: Vec<Mat4>,
    pub depth_texture: Texture,
    pub start_time: Instant,
    pub prev_frame_time: Option<f32>,

    pub input: InputState,

    pub ui: egui_demo_lib::WrapApp,
    // pub graphics_pipeline_single_vertex_attribute: wgpu::RenderPipeline,
    // pub grid_vertex_buffer: wgpu::Buffer,
}

/// User event handling.
pub enum UserEvent {
    RequestRedraw,
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
    pub async fn new(window: &Window) -> Result<Self, Error> {
        let mut gpu_ctx = GpuContext::new(window).await;
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
        let projection = Projection::new(
            0.1,
            100.0,
            45.0f32.to_radians(),
            gpu_ctx.surface_config.width,
            gpu_ctx.surface_config.height,
        );
        let object_model_matrix = glam::Mat4::IDENTITY;
        let camera = Camera::new(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
        let camera_uniform = CameraUniform::new(&camera, &projection);
        let camera_uniform_buffer =
            gpu_ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("camera-uniform-buffer"),
                    contents: bytemuck::cast_slice(&[
                        camera_uniform.view_matrix,
                        camera_uniform.proj_matrix,
                        object_model_matrix,
                    ]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let camera_bind_group_layout =
            gpu_ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("camera-uniform-bind-group-create_bind_group_layout"),
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
        let camera_bind_group = gpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("camera-uniform-bind-group"),
                layout: &camera_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buffer.as_entire_binding(),
                }],
            });
        let camera_controller =
            OrbitControls::new(0.3, f32::INFINITY, true, true, true, 200.0, 400.0, 100.0);

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
                    include_str!("../assets/shaders/shader.wgsl").into(),
                ),
            });

        // let grid_shader_module = gfx_state.device.create_shader_module(
        //     &wgpu::ShaderModuleDescriptor {
        //         label: Some("grid_vertex_shader"),
        //         source: ()
        //     }
        // );

        // Pipeline layout
        let render_pipeline_layout =
            gpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("default-pipeline-layout"),
                    bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                    push_constant_ranges: &[],
                });
        // let graphics_pipeline_layout = gfx_state.device.create_pipeline_layout(
        //     &wgpu::PipelineLayoutDescriptor {
        //         label: Some("single_vertex_attribute_graphics_pipeline_layout"),
        //         bind_group_layouts: &[&camera_bind_group_layout],
        //         push_constant_ranges: &[]
        //     }
        // );

        let instancing_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
            // Switch from using as step mode of Vertex to Instance.
            // It means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 5,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                },
            ],
        };

        let render_pipeline =
            gpu_ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::layout(), instancing_buffer_layout],
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
                        targets: &[gpu_ctx.surface_config.format.into()],
                    }),
                    multiview: None,
                });

        // let graphics_pipeline_single_vertex_attribute =
        // gfx_state.device.create_render_pipeline(     &wgpu::
        // RenderPipelineDescriptor {         label:
        // Some("single_vertex_attribute_graphics_pipeline"),         layout:
        // Some(&graphics_pipeline_layout),         vertex: wgpu::VertexState {
        //             module: &(),
        //             entry_point: "",
        //             buffers: &[]
        //         },
        //         primitive: Default::default(),
        //         depth_stencil: None,
        //         multisample: Default::default(),
        //         fragment: None,
        //         multiview: None
        //     }
        // )

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

        let ui_state = UiState::new(window, &gpu_ctx.device, gpu_ctx.surface_config.format, 1);
        let ui = egui_demo_lib::WrapApp::default();

        // let grid_vertex_buffer =
        // gfx_state.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("grid_vertex_buffer"),
        //     contents: &[],
        //     usage: wgpu::BufferUsages::VERTEX
        // });

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        Ok(Self {
            gpu: gpu_ctx,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_vertices,
            num_indices,
            texture_bind_groups,
            current_texture_index: 0,
            camera,
            camera_uniform,
            camera_uniform_buffer,
            camera_bind_group,
            camera_controller,
            projection,
            object_model_matrix,
            instancing_buffer,
            instancing_transforms,
            depth_texture,
            start_time: Instant::now(),
            prev_frame_time: None,
            ui_state,
            ui,
            gui: VgonioUi::new(),
            input,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.gpu.surface_config.width = new_size.width;
            self.gpu.surface_config.height = new_size.height;
            self.gpu
                .surface
                .configure(&self.gpu.device, &self.gpu.surface_config);
            self.depth_texture = Texture::create_depth_texture(
                &self.gpu.device,
                &self.gpu.surface_config,
                "depth_texture",
            );
            self.projection.resize(new_size.width, new_size.height);
        }
    }

    pub fn process_input(&mut self, event: &WindowEvent) -> bool {
        if !self.ui_state.handle_event(event) {
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
                // WindowEvent::KeyboardInput {
                //     input:
                //         KeyboardInput {
                //             state: ElementState::Pressed,
                //             virtual_keycode: Some(VirtualKeyCode::Space),
                //             ..
                //         },
                //     ..
                // } => {
                //     self.current_texture_index += 1;
                //     self.current_texture_index %= 2;
                //     true
                // }
                // WindowEvent::KeyboardInput {
                //     input:
                //         KeyboardInput {
                //             state: ElementState::Pressed,
                //             virtual_keycode: Some(VirtualKeyCode::R),
                //             ..
                //         },
                //     ..
                // } => {
                //     self.object_model_matrix =
                //         Mat4::from_rotation_y(3.0f32.to_radians()) * self.object_model_matrix;
                //     true
                // }
                // WindowEvent::KeyboardInput {
                //     input:
                //         KeyboardInput {
                //             state: ElementState::Pressed,
                //             virtual_keycode: Some(VirtualKeyCode::I),
                //             ..
                //         },
                //     ..
                // } => true,
                WindowEvent::MouseWheel { delta, .. } => {
                    self.input.update_scroll_delta(*delta);
                    true
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    self.input.update_mouse_map(*button, *state);
                    println!(
                        "mouse middle button {:?}",
                        self.input.is_mouse_button_pressed(MouseButton::Middle)
                    );
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
        self.camera_controller
            .update_camera(&self.input, &mut self.camera, dt);
        self.camera_uniform.update(&self.camera, &self.projection);
        if self.gui.selected_workspace == "Simulation" {
            self.gui.workspaces.simulation.update_gizmo_matrices(
                Mat4::IDENTITY,
                self.camera_uniform.view_matrix,
                Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
            );
        }
        self.gpu.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::cast_slice(&[
                self.camera_uniform.view_matrix,
                self.camera_uniform.proj_matrix,
                self.object_model_matrix,
            ]),
        );
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];
        for m in &mut self.instancing_transforms {
            *m *= Mat4::from_rotation_x(0.2f32.to_radians());
        }
        self.gpu.queue.write_buffer(
            &self.instancing_buffer,
            0,
            bytemuck::cast_slice(&self.instancing_transforms),
        );
    }

    pub fn render(
        &mut self,
        window: &winit::window::Window,
        repaint_signal: std::sync::Arc<RepaintSignal>,
    ) -> Result<(), wgpu::SurfaceError> {
        let output_frame = self.gpu.surface.get_current_texture()?;
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoders = [
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_render_encoder"),
                }),
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_ui_render_encoder"),
                }),
        ];

        {
            let mut render_pass = encoders[0].begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(
                0,
                &self.texture_bind_groups[self.current_texture_index],
                &[],
            );
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instancing_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(
                0..self.num_indices,
                0,
                0..self.instancing_transforms.len() as _,
            );
        }

        {
            // Record UI
            let ui_start_time = Instant::now();
            let input = self.ui_state.egui_state_mut().take_egui_input(window);

            self.ui_state.egui_context_mut().begin_frame(input);
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
            self.gui.update(self.ui_state.egui_context(), &ui_frame);
            let ui_output_frame = self.ui_state.egui_context().end_frame();

            let meshes = self
                .ui_state
                .egui_context()
                .tessellate(ui_output_frame.shapes);
            let frame_time = (Instant::now() - ui_start_time).as_secs_f64() as f32;
            self.prev_frame_time = Some(frame_time);

            let win_size = ui_state::WindowSize {
                physical_width: self.gpu.surface_config.width,
                physical_height: self.gpu.surface_config.height,
                scale_factor: window.scale_factor() as f32,
            };

            self.ui_state
                .update_textures(
                    &self.gpu.device,
                    &self.gpu.queue,
                    ui_output_frame.textures_delta,
                )
                .unwrap();
            self.ui_state
                .update_buffers(&self.gpu.device, &self.gpu.queue, &meshes, &win_size);
            self.ui_state
                .render(&mut encoders[1], &output_view, &meshes, &win_size, None)
                .unwrap();
        }

        self.gpu.queue.submit(encoders.map(|enc| enc.finish()));

        output_frame.present();

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    // color: [f32; 3],
    tex_coord: [f32; 2],
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

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

impl Vertex {
    const fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                },
            ],
        }
    }
}

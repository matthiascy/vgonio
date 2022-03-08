use crate::app::camera::{Camera, CameraController, CameraUniform};
use crate::error::Error;
use glam::{Mat4, Quat, Vec3};
use image::{GenericImageView, ImageFormat, RgbaImage};
use std::default::Default;
use std::mem;
use wgpu::util::DeviceExt;
use wgpu::VertexFormat;
use winit::event::KeyboardInput;
use winit::event::{ElementState, VirtualKeyCode, WindowEvent};
use winit::window::Window;

const NUM_INSTANCES_PER_ROW: u32 = 9;
const NUM_INSTANCES_PER_COL: u32 = 9;

pub struct UiState {
    pub pipeline: wgpu::RenderPipeline,
    pub index_buffers: Vec<wgpu::Buffer>,
    pub vertex_buffers: Vec<wgpu::Buffer>,
    pub uniform_buffers: Vec<wgpu::Buffer>,
}

pub struct VgonioState {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
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
    pub camera_controller: CameraController,
    pub object_model_matrix: glam::Mat4,
    pub instancing_buffer: wgpu::Buffer,
    pub instancing_transforms: Vec<Mat4>,
}

impl VgonioState {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(window: &Window) -> Result<Self, Error> {
        let size = window.inner_size();

        let num_vertices = VERTICES.len() as u32;

        // Create instance handle to GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        // An abstract type of surface to present rendered images to.
        let surface = unsafe { instance.create_surface(window) };
        // Physical device: handle to actual graphics card.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap_or_else(|| {
                panic!(
                    "Failed to request physical device! {}",
                    concat!(file!(), ":", line!())
                )
            });
        // Logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to request logical device! {}",
                    concat!(file!(), ":", line!())
                )
            });

        // Swapchain format
        let swapchain_format = surface.get_preferred_format(&adapter).unwrap();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &surface_config);

        // Create texture
        let texture_images = [
            image::load_from_memory_with_format(
                include_bytes!("assets/damascus001.jpg"),
                ImageFormat::Jpeg,
            )
            .unwrap()
            .flipv()
            .to_rgba8(),
            image::load_from_memory_with_format(
                include_bytes!("assets/damascus002.jpg"),
                ImageFormat::Jpeg,
            )
            .unwrap()
            .flipv()
            .to_rgba8(),
        ];
        let texture_dims = [
            texture_images[0].dimensions(),
            texture_images[1].dimensions(),
        ];
        let texture_extents = [
            wgpu::Extent3d {
                width: texture_dims[0].0,
                height: texture_dims[0].1,
                depth_or_array_layers: 1,
            },
            wgpu::Extent3d {
                width: texture_dims[1].0,
                height: texture_dims[1].1,
                depth_or_array_layers: 1,
            },
        ];
        let textures = [
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("damascus-texture-001"),
                size: texture_extents[0],
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            }),
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("damascus-texture-002"),
                size: texture_extents[1],
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            }),
        ];

        // Load texture in
        for i in 0..textures.len() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &textures[i],
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                // actual pixel data
                &texture_images[i],
                // layout of the texture
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(4 * texture_dims[i].0),
                    rows_per_image: std::num::NonZeroU32::new(texture_dims[i].1),
                },
                texture_extents[i],
            );
        }

        let texture_views = [
            textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Descriptor Sets
        // [`BindGroup`] describes a set of resources and how they can be accessed by a
        // shader.
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            // SamplerBindingType::Comparison is only for TextureSampleType::Depth
                            // SamplerBindingType::Filtering if the sample_type of the texture is:
                            //     TextureSampleType::Float { filterable: true }
                            // Otherwise you'll get an error.
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        let texture_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("diffuse-texture-bind-group-001"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("diffuse-texture-bind-group-002"),
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&&texture_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler),
                    },
                ],
            }),
        ];

        // Camera
        let object_model_matrix = glam::Mat4::IDENTITY;
        let camera = Camera {
            eye: (0.0, 1.0, 2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vec3::Y,
            aspect: surface_config.width as f32 / surface_config.height as f32,
            fov: 45.0f32.to_radians(),
            near: 0.1,
            far: 100.0,
        };
        let camera_uniform = camera.uniform();
        let camera_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform-buffer"),
            contents: bytemuck::cast_slice(&[
                camera_uniform.view_matrix,
                camera_uniform.proj_matrix,
                object_model_matrix,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-uniform-bind-group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });
        let camera_controller = CameraController::new(0.2);

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
        let instancing_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("object-instancing-buffer"),
            contents: bytemuck::cast_slice(&instancing_transforms),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("default-vertex-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("assets/shaders/shader.wgsl").into()),
        });

        // Pipeline layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("default-pipeline-layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

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
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                },
                wgpu::VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[swapchain_format.into()],
            }),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex-buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index-buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = INDICES.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            size,
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
            object_model_matrix,
            instancing_buffer,
            instancing_transforms,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        let status_camera = self.camera_controller.process_event(event);
        let status = match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Space),
                        ..
                    },
                ..
            } => {
                self.current_texture_index += 1;
                self.current_texture_index %= 2;
                true
            }
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::R),
                        ..
                    },
                ..
            } => {
                self.object_model_matrix =
                    Mat4::from_rotation_z(3.0f32.to_radians()) * self.object_model_matrix;
                true
            }
            WindowEvent::KeyboardInput {
                input:
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::I),
                    ..
                },
                ..
            } => {
                true
            }
            _ => false,
        };

        status && status_camera
    }

    pub fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform = self.camera.uniform();
        self.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::cast_slice(&[
                self.camera_uniform.view_matrix,
                self.camera_uniform.proj_matrix,
                self.object_model_matrix,
            ]),
        );

        for m in &mut self.instancing_transforms {
            *m =  (*m) * Mat4::from_rotation_x(0.2f32.to_radians());
        }
        self.queue.write_buffer(&self.instancing_buffer, 0, bytemuck::cast_slice(&self.instancing_transforms));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        // Image view
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &view,
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
                depth_stencil_attachment: None,
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
            // render_pass.draw(0..self.num_vertices, 0..1);
            render_pass.draw_indexed(
                0..self.num_indices,
                0,
                0..self.instancing_transforms.len() as _,
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

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

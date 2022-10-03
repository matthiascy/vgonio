use egui::epaint::Primitive;
use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    num::NonZeroU32,
};

use crate::{error::Error, gfx::SizedBuffer};
use wgpu::util::DeviceExt;
use crate::app::gui::VgonioEvent;

/// Enum for selecting the right buffer type.
#[derive(Debug)]
enum BufferType {
    Uniform,
    Index,
    Vertex,
}

/// Information about the window on which the UI is presented.
pub struct WindowSize {
    /// Width of the window in physical pixel.
    pub physical_width: u32,
    /// Height of the window in physical pixel.
    pub physical_height: u32,
    /// HiDPI scale factor.
    pub scale_factor: f32,
}

impl WindowSize {
    fn logical_size(&self) -> (u32, u32) {
        let logical_width = self.physical_width as f32 / self.scale_factor;
        let logical_height = self.physical_height as f32 / self.scale_factor;
        (logical_width as u32, logical_height as u32)
    }
}

/// Uniform buffer used when rendering.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct UiUniform {
    window_size: [f32; 2],
}

unsafe impl bytemuck::Pod for UiUniform {}

unsafe impl bytemuck::Zeroable for UiUniform {}

/// Render pass set up for rendering the UI (wgpu).
pub struct GuiContext {
    /// Context for UI generation
    egui_context: egui::Context,

    /// States managing the translation of input from winit to egui
    egui_state: egui_winit::State,

    /// Inputs translated from winit
    egui_input: egui::RawInput,

    /// Render pipline for UI
    render_pipeline: wgpu::RenderPipeline,

    /// Index buffers for meshes generated by egui
    index_buffers: Vec<SizedBuffer>,

    /// Vertex buffers for meshes generated by egui
    vertex_buffers: Vec<SizedBuffer>,

    /// Uniform buffer storing the size
    uniform_buffer: SizedBuffer,

    /// Bind group of uniform
    uniform_bind_group: wgpu::BindGroup,

    /// Bind group layout of textures (texture view + sampler)
    texture_bind_group_layout: wgpu::BindGroupLayout,

    /// The mapping from egui texture IDs to the associated bind group (texture
    /// view + sampler)
    textures: HashMap<egui::TextureId, (Option<wgpu::Texture>, wgpu::BindGroup)>,

    /// Id used to identify textures created by user
    next_user_texture_id: u64,
}

impl GuiContext {
    /// Creates related resources used for UI rendering.
    ///
    /// If the format passed is not a *Srgb format, the shader will
    /// automatically convert to sRGB colors in the shader.
    pub fn new(
        event_loop: &winit::event_loop::EventLoopWindowTarget<VgonioEvent>,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let egui_context = egui::Context::default();
        let egui_state = egui_winit::State::new(event_loop);
        let shader_module = {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ui_shader_module"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../assets/shaders/wgsl/ui.wgsl"
                ))),
            })
        };

        let uniform_buffer = {
            let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui_uniform_buffer"),
                contents: bytemuck::cast_slice(&[UiUniform {
                    window_size: [0.0, 0.0],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            SizedBuffer {
                raw,
                size: std::mem::size_of::<UiUniform>(),
            }
        };

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer.raw,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: if surface_format.describe().srgb {
                    "vs_main"
                } else {
                    "vs_conv_main"
                },
                module: &shader_module,
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 5 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    // 0: vec2 position
                    // 1: vec2 texture coordinates
                    // 2: uint color
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Uint32
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                unclipped_depth: false,
                conservative: false,
                cull_mode: None,
                front_face: wgpu::FrontFace::default(),
                polygon_mode: wgpu::PolygonMode::default(),
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: msaa_samples,
                mask: !0,
            },

            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            egui_context,
            egui_state,
            egui_input: Default::default(),
            render_pipeline,
            vertex_buffers: Vec::with_capacity(64),
            index_buffers: Vec::with_capacity(64),
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            textures: HashMap::new(),
            next_user_texture_id: 0,
        }
    }

    pub fn take_input(&mut self, window: &winit::window::Window) {
        self.egui_input = self.egui_state.take_egui_input(window);
    }

    pub fn begin_frame(&mut self) { self.egui_context.begin_frame(self.egui_input.take()); }

    pub fn handle_event(&mut self, event: &winit::event::WindowEvent) -> bool {
        self.egui_state.on_event(&self.egui_context, event)
    }

    /// Recording the UI rendering command. When `clear_color` is set, the
    /// output target will get cleared before writing to it.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        primitives: &[egui::epaint::ClippedPrimitive],
        screen_descriptor: &WindowSize,
        clear_color: Option<wgpu::Color>,
    ) -> Result<(), Error> {
        let load_op = if let Some(color) = clear_color {
            wgpu::LoadOp::Clear(color)
        } else {
            wgpu::LoadOp::Load
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui_main_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.push_debug_group("ui_pass");

        self.record_render_pass(&mut render_pass, primitives, screen_descriptor)?;

        render_pass.pop_debug_group();

        Ok(())
    }

    /// Executes the egui render pass onto an existing wgpu renderpass.
    fn record_render_pass<'rdrps>(
        &'rdrps self,
        render_pass: &mut wgpu::RenderPass<'rdrps>,
        primitives: &[egui::epaint::ClippedPrimitive],
        screen_descriptor: &WindowSize,
    ) -> Result<(), Error> {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

        let scale_factor = screen_descriptor.scale_factor;
        let physical_width = screen_descriptor.physical_width;
        let physical_height = screen_descriptor.physical_height;

        for (
            (
                egui::epaint::ClippedPrimitive {
                    clip_rect,
                    primitive,
                },
                vertex_buffer,
            ),
            index_buffer,
        ) in primitives
            .iter()
            .zip(self.vertex_buffers.iter())
            .zip(self.index_buffers.iter())
        {
            let (x, y, width, height) = {
                // Transform clip rect to physical pixels position.
                let (clip_min_x, clip_min_y, clip_max_x, clip_max_y) = {
                    let clip_min_x =
                        (scale_factor * clip_rect.min.x).clamp(0.0, physical_width as f32);
                    let clip_min_y =
                        (scale_factor * clip_rect.min.y).clamp(0.0, physical_height as f32);
                    let clip_max_x =
                        (scale_factor * clip_rect.max.x).clamp(clip_min_x, physical_width as f32);
                    let clip_max_y =
                        (scale_factor * clip_rect.max.y).clamp(clip_min_y, physical_height as f32);
                    (
                        clip_min_x.round() as u32,
                        clip_min_y.round() as u32,
                        clip_max_x.round() as u32,
                        clip_max_y.round() as u32,
                    )
                };
                let width = (clip_max_x - clip_min_x).max(1);
                let height = (clip_max_y - clip_min_y).max(1);

                // Clip scissor rectangle to target size.
                let x = clip_min_x.min(physical_width);
                let y = clip_min_y.min(physical_height);
                let width = width.min(physical_width - x);
                let height = height.min(physical_height - y);

                (x, y, width, height)
            };

            // Skip rendering with zero-sized clip areas.
            if width == 0 || height == 0 {
                continue;
            }

            match primitive {
                Primitive::Mesh(mesh) => {
                    render_pass.set_scissor_rect(x, y, width, height);
                    let bind_group = self.get_texture_bind_group(mesh.texture_id)?;
                    render_pass.set_bind_group(1, bind_group, &[]);
                    render_pass
                        .set_index_buffer(index_buffer.raw.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_vertex_buffer(0, vertex_buffer.raw.slice(..));
                    render_pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
                }
                Primitive::Callback(_) => {
                    todo!("Primitive::Callback not implemented");
                }
            }
        }

        Ok(())
    }

    fn get_texture_bind_group(
        &self,
        texture_id: egui::TextureId,
    ) -> Result<&wgpu::BindGroup, Error> {
        self.textures
            .get(&texture_id)
            .ok_or_else(|| Error::Any(format!("Texture {:?} used but not live", texture_id)))
            .map(|x| &x.1)
    }

    /// Updates the textures used by egui. Remove unused textures.
    ///
    /// WARNING: this function should be called before `render()`.
    pub fn update_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: egui::TexturesDelta,
    ) -> Result<(), Error> {
        for (texture_id, image_delta) in textures.set.iter() {
            let image_size = image_delta.image.size();

            let origin = match image_delta.pos {
                Some([x, y]) => wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                None => wgpu::Origin3d::ZERO,
            };

            let image_data = match &image_delta.image {
                egui::ImageData::Color(image) => {
                    assert_eq!(
                        image_size[0] * image_size[1],
                        image.pixels.len(),
                        "Mismatch between texture size and texel count"
                    );
                    Cow::Borrowed(&image.pixels)
                }
                egui::ImageData::Font(image) => {
                    assert_eq!(
                        image_size[0] * image_size[1],
                        image.pixels.len(),
                        "Mismatch between texture size and texel count"
                    );
                    Cow::Owned(image.srgba_pixels(1.0).collect::<Vec<_>>())
                }
            };

            let image_data_bytes = bytemuck::cast_slice(image_data.as_slice());

            let image_size = wgpu::Extent3d {
                width: image_size[0] as u32,
                height: image_size[1] as u32,
                depth_or_array_layers: 1,
            };

            let image_data_layout = wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * image_size.width),
                rows_per_image: None,
            };

            let label_base = match texture_id {
                egui::TextureId::Managed(m) => format!("egui_image_{}", m),
                egui::TextureId::User(u) => format!("egui_user_image_{}", u),
            };

            match self.textures.entry(*texture_id) {
                Entry::Occupied(mut o) => match image_delta.pos {
                    None => {
                        let texture = create_texture(
                            device,
                            queue,
                            origin,
                            image_data_bytes,
                            image_data_layout,
                            image_size,
                            &label_base,
                        );

                        let bind_group = create_texture_bind_group(
                            device,
                            &texture,
                            &self.texture_bind_group_layout,
                            &label_base,
                        );

                        let (texture, _) = o.insert((Some(texture), bind_group));

                        if let Some(texture) = texture {
                            texture.destroy();
                        }
                    }
                    Some(_) => {
                        if let Some(texture) = o.get().0.as_ref() {
                            queue.write_texture(
                                wgpu::ImageCopyTexture {
                                    texture,
                                    mip_level: 0,
                                    origin,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                image_data_bytes,
                                image_data_layout,
                                image_size,
                            );
                        } else {
                            return Err(Error::Any(format!(
                                "InvalidTextureId - Update of unmanaged texture {:?}",
                                texture_id
                            )));
                        }
                    }
                },
                Entry::Vacant(v) => {
                    let texture = create_texture(
                        device,
                        queue,
                        origin,
                        image_data_bytes,
                        image_data_layout,
                        image_size,
                        &label_base,
                    );

                    let bind_group = create_texture_bind_group(
                        device,
                        &texture,
                        &self.texture_bind_group_layout,
                        &label_base,
                    );

                    v.insert((Some(texture), bind_group));
                }
            }
        }

        self.remove_unused_textures(textures)
    }

    /// Remove the textures egui no longer needs.
    ///
    /// WARNING: This function should be called after `render()`.
    fn remove_unused_textures(&mut self, textures: egui::TexturesDelta) -> Result<(), Error> {
        for texture_id in textures.free {
            let (texture, _binding) = self.textures.remove(&texture_id).ok_or_else(|| {
                // This can happen due to a bug in egui, or if the user doesn't call
                // `add_textures` when required.
                Error::Any(format!(
                    "InvalidTextureId - Attempted to remove an unknown texture {:?}",
                    texture_id
                ))
            })?;

            if let Some(texture) = texture {
                texture.destroy();
            }
        }

        Ok(())
    }

    /// Registers a `wgpu::Texture` with a `egui::TextureId`.
    ///
    /// This enables the application to reference the texture inside an image ui
    /// element. This effectively enables off-screen rendering inside the
    /// egui UI. Texture must have the texture format
    /// `TextureFormat::Rgba8UnormSrgb` and Texture usage
    /// `TextureUsage::SAMPLED`.
    pub fn register_egui_texture_from_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
    ) -> egui::TextureId {
        self.create_egui_texture_from_wgpu_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(
                    format!(
                        "egui_user_image_{}_texture_sampler",
                        self.next_user_texture_id
                    )
                    .as_str(),
                ),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
        )
    }

    /// Registers a `wgpu::Texture` with an existing `egui::TextureId`.
    ///
    /// This enables applications to reuse `TextureId`s.
    pub fn update_egui_texture_from_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
        id: egui::TextureId,
    ) -> Result<(), Error> {
        self.update_egui_texture_from_wgpu_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(
                    format!(
                        "egui_user_image_{}_texture_sampler",
                        self.next_user_texture_id
                    )
                    .as_str(),
                ),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
            id,
        )
    }

    /// Registers a `wgpu::Texture` with a `egui::TextureId` while also
    /// accepting custom `wgpu::SamplerDescriptor` options.
    ///
    /// This allows applications to specify individual
    /// minification/magnification filters as well as custom mipmap and
    /// tiling options.
    ///
    /// The `Texture` must have the format `TextureFormat::Rgba8UnormSrgb` and
    /// usage `TextureUsage::SAMPLED`. Any compare function supplied in the
    /// `SamplerDescriptor` will be ignored.
    fn create_egui_texture_from_wgpu_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        desc: wgpu::SamplerDescriptor,
    ) -> egui::TextureId {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..desc
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(
                format!(
                    "egui_user_image_{}_texture_bind_group",
                    self.next_user_texture_id
                )
                .as_str(),
            ),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let id = egui::TextureId::User(self.next_user_texture_id);
        self.textures.insert(id, (None, bind_group));
        self.next_user_texture_id += 1;

        id
    }

    /// Registers a `wgpu::Texture` with an existing `egui::TextureId` while
    /// also accepting custom `wgpu::SamplerDescriptor` options.
    ///
    /// This allows applications to reuse `TextureId`s created with custom
    /// sampler options.
    fn update_egui_texture_from_wgpu_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler_descriptor: wgpu::SamplerDescriptor,
        id: egui::TextureId,
    ) -> Result<(), Error> {
        if let egui::TextureId::Managed(_) = id {
            return Err(Error::Any(
                "Invalid texture ID - ID was not of type `TextureId::User`".to_string(),
            ));
        }

        let (_user_texture, user_texture_binding) =
            self.textures.get_mut(&id).ok_or_else(|| {
                Error::Any(format!(
                    "InvalidTextureId - user texture for TextureId {:?} could not be found",
                    id
                ))
            })?;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler_descriptor
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(
                format!("egui_user_{}_texture_bind_group", self.next_user_texture_id).as_str(),
            ),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        *user_texture_binding = bind_group;

        Ok(())
    }

    /// Uploads the uniform, vertex and index data used by the render pass.
    /// Should be called before `execute()`.
    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[egui::ClippedPrimitive],
        win_size: &WindowSize,
    ) {
        let index_size = self.index_buffers.len();
        let vertex_size = self.vertex_buffers.len();

        let (logical_width, logical_height) = win_size.logical_size();

        self.update_buffer(
            device,
            queue,
            BufferType::Uniform,
            0,
            bytemuck::cast_slice(&[UiUniform {
                window_size: [logical_width as f32, logical_height as f32],
            }]),
        );

        for (
            i,
            egui::ClippedPrimitive {
                clip_rect: _,
                primitive,
            },
        ) in meshes.iter().enumerate()
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    let data: &[u8] = bytemuck::cast_slice(&mesh.indices);
                    if i < index_size {
                        self.update_buffer(device, queue, BufferType::Index, i, data)
                    } else {
                        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("ui_index_buffer"),
                            contents: data,
                            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                        });
                        self.index_buffers.push(SizedBuffer {
                            raw: buffer,
                            size: data.len(),
                        });
                    }

                    let data: &[u8] = bytemuck::cast_slice(&mesh.vertices);
                    if i < vertex_size {
                        self.update_buffer(device, queue, BufferType::Vertex, i, data)
                    } else {
                        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("ui_vertex_buffer"),
                            contents: data,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        });

                        self.vertex_buffers.push(SizedBuffer {
                            raw: buffer,
                            size: data.len(),
                        });
                    }
                }
                Primitive::Callback(_) => {
                    todo!("Primitive::Callback not implemented");
                }
            }
        }
    }

    /// Updates the buffers used by egui. Will properly re-size the buffers if
    /// needed.
    fn update_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_type: BufferType,
        index: usize,
        data: &[u8],
    ) {
        let (buffer, storage, name) = match buffer_type {
            BufferType::Index => (
                &mut self.index_buffers[index],
                wgpu::BufferUsages::INDEX,
                "index",
            ),
            BufferType::Vertex => (
                &mut self.vertex_buffers[index],
                wgpu::BufferUsages::VERTEX,
                "vertex",
            ),
            BufferType::Uniform => (
                &mut self.uniform_buffer,
                wgpu::BufferUsages::UNIFORM,
                "uniform",
            ),
        };

        if data.len() > buffer.size {
            buffer.size = data.len();
            buffer.raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("egui_{}_buffer", name).as_str()),
                contents: bytemuck::cast_slice(data),
                usage: storage | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&buffer.raw, 0, data);
        }
    }

    pub fn egui_context(&self) -> &egui::Context { &self.egui_context }

    pub fn egui_context_mut(&mut self) -> &mut egui::Context { &mut self.egui_context }

    pub fn egui_state(&self) -> &egui_winit::State { &self.egui_state }

    pub fn egui_state_mut(&mut self) -> &mut egui_winit::State { &mut self.egui_state }
}

/// Create a texture and bind group from existing data
fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    origin: wgpu::Origin3d,
    image_data: &[u8],
    image_data_layout: wgpu::ImageDataLayout,
    image_size: wgpu::Extent3d,
    label_base: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(format!("{}_texture", label_base).as_str()),
        size: image_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        image_data,
        image_data_layout,
        image_size,
    );

    texture
}

fn create_texture_bind_group(
    device: &wgpu::Device,
    texture: &wgpu::Texture,
    layout: &wgpu::BindGroupLayout,
    label_base: &str,
) -> wgpu::BindGroup {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(format!("{}_sampler", label_base).as_str()),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(format!("{}_bind_group", label_base).as_str()),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    })
}

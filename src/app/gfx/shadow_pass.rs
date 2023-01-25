use crate::{
    app::gfx::{remap_depth, GpuContext, RenderPass, Texture},
    Error,
};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::{num::NonZeroU32, sync::Arc};

/// Render pass generating depth map (from light P.O.V.) used later for shadow
/// mapping.
pub struct ShadowPass {
    /// Pipeline and corresponding shader inputs.
    inner: RenderPass,

    /// Depth attachment of the render pass.
    depth_attachment: Texture,

    /// Buffer aiming to be used as the storage of the `depth_map`, accessed
    /// through address mapping.
    depth_attachment_storage: Option<wgpu::Buffer>,

    /// Depth map image width
    width: u32,

    /// Depth map image height
    height: u32,

    /// Size of depth map in bytes
    size: u64,
}

/// Required uniforms to generate depth map.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DepthPassUniforms {
    /// Object model matrix. TODO: maybe better to use push constant
    pub model_matrix: [f32; 16],

    // Light space viewing and projection matrix.
    pub light_space_matrix: [f32; 16],
}

impl ShadowPass {
    pub fn new(
        ctx: &GpuContext,
        width: u32,
        height: u32,
        alloc_buf: bool,
        shader_accessible: bool,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let uniform_buffer_size = std::mem::size_of::<DepthPassUniforms>() as wgpu::BufferAddress;
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shadow_pass_uniform_buffer"),
                contents: bytemuck::cast_slice(&[0.0f32; 48]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(uniform_buffer_size),
                        },
                        count: None,
                    }],
                });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_pass_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_pass_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow_pass_shader_module"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../gui/assets/shaders/wgsl/shadow_pass.wgsl").into(),
                ),
            });
        let sampler = shader_accessible.then(|| {
            Arc::new(ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: -100.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }))
        });
        let depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            None,
            sampler,
            None,
            Some("shadow_pass_depth_attachment"),
        );
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_pass_pipeline"),
                layout: Some(&pipeline_layout),
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
                fragment: None,
                multiview: None,
            });
        let pass = RenderPass {
            pipeline,
            bind_groups: vec![bind_group],
            uniform_buffer: Some(uniform_buffer),
        };

        let size = (std::mem::size_of::<f32>() * (width * height) as usize) as wgpu::BufferAddress;

        let depth_attachment_storage = alloc_buf.then(|| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_pass_depth_attachment_storage"),
                size: size as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });

        Self {
            inner: pass,
            depth_attachment,
            depth_attachment_storage,
            width,
            height,
            size,
        }
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline { &self.inner.pipeline }

    pub fn bind_groups(&self) -> &[wgpu::BindGroup] { &self.inner.bind_groups }

    pub fn uniform_buffer(&self) -> Option<&wgpu::Buffer> { self.inner.uniform_buffer.as_ref() }

    pub fn depth_attachment(&self) -> (&wgpu::Texture, &wgpu::TextureView, &wgpu::Sampler) {
        (
            &self.depth_attachment.raw,
            &self.depth_attachment.view,
            &self.depth_attachment.sampler,
        )
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.size = (std::mem::size_of::<f32>() * (width * height) as usize) as u64;
        self.depth_attachment =
            Texture::create_depth_texture(device, width, height, None, None, None, None);
        self.depth_attachment_storage = if self.depth_attachment_storage.is_some() {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow_pass_depth_attachment_storage"),
                size: self.size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
        } else {
            None
        };
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, model: Mat4, view: Mat4, proj: Mat4) {
        queue.write_buffer(
            self.inner.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[model, proj * view]),
        );
    }

    pub fn bake(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        v_buf: &wgpu::Buffer,
        i_buf: &wgpu::Buffer,
        index_ranges: &[(u32, u32)],
        i_format: wgpu::IndexFormat,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth_pass_bake"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_attachment.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.inner.pipeline);
            render_pass.set_bind_group(0, &self.inner.bind_groups[0], &[]);
            render_pass.set_vertex_buffer(0, v_buf.slice(..));
            render_pass.set_index_buffer(i_buf.slice(..), i_format);
            for (start, end) in index_ranges {
                render_pass.draw_indexed(*start..*end, 0, 0..1);
            }
        }

        // Copy depth buffer values to depth attachment storage
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.depth_attachment.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: self.depth_attachment_storage.as_ref().unwrap(),
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(std::mem::size_of::<f32>() as u32 * self.width),
                    rows_per_image: NonZeroU32::new(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
    }

    pub fn save_to_image(
        &self,
        device: &wgpu::Device,
        near: f32,
        far: f32,
        path: &std::path::Path,
    ) -> Result<(), Error> {
        if let Some(buffer) = &self.depth_attachment_storage {
            {
                let slice = buffer.slice(..);

                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap();
                });
                device.poll(wgpu::Maintain::Wait);
                pollster::block_on(async {
                    receiver.receive().await.unwrap().unwrap();
                });

                let buffer_view = slice.get_mapped_range();
                let data = unsafe {
                    let (_, data, _) = buffer_view.align_to::<f32>();
                    data.iter()
                        .map(|val| (remap_depth(*val, near, far) * 255.0) as u8)
                        .collect::<Vec<u8>>()
                };

                use image::{ImageBuffer, Luma};
                ImageBuffer::<Luma<u8>, _>::from_raw(self.width, self.height, data)
                    .ok_or_else(|| {
                        Error::Any(
                            "Failed to create image from depth map buffer, please check if the \
                             data have been transferred to the buffer!"
                                .into(),
                        )
                    })
                    .and_then(|img| img.save(path).map_err(Error::from))?;
            }
            buffer.unmap();

            Ok(())
        } else {
            Err(Error::Any("Depth map buffer is not created!".into()))
        }
    }

    pub fn compute_pixels_count(&self, device: &wgpu::Device) -> u32 {
        if let Some(buffer) = &self.depth_attachment_storage {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
            let count = {
                let slice = buffer.slice(..);

                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap();
                });
                device.poll(wgpu::Maintain::Wait);
                pollster::block_on(async {
                    receiver.receive().await.unwrap().unwrap();
                });

                let buffer_view = slice.get_mapped_range();
                unsafe {
                    let (_, data, _) = buffer_view.align_to::<f32>();
                    data.par_iter()
                        .fold(
                            || 0,
                            |count, val| {
                                if (*val - 1.0).abs() < f32::EPSILON {
                                    count
                                } else {
                                    count + 1
                                }
                            },
                        )
                        .sum()
                }
            };
            buffer.unmap();
            count
        } else {
            0
        }
    }
}

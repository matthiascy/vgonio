use crate::{
    app::state::remap_depth,
    gfx::{GpuContext, RdrPass, Texture},
    Error,
};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::num::NonZeroU32;
use wgpu::{include_spirv, util::DeviceExt, ColorTargetState};

/// Render pass computing the shadowing/masking (caused by occlusion of
/// micro-facets) function of a micro-surface. For a certain viewing direction,
/// this is done in two steps:
///
/// 1. Depth pass
///
/// Generate a depth map from light POV just for visible facets, then
/// count the number of pixels as the projected area of all visible facets while
/// accounting for occlusion.
///
/// 2. Render pass
///
/// Render all visible facets with depth test disabled. Each fragment outputs
/// 0.015625, then at the blending stage, sum up.
pub struct OcclusionEstimationPass {
    /// Pipeline and corresponding shader inputs.
    inner: RdrPass,

    /// Depth attachment of the render pass.
    depth_attachment: Texture,

    /// Buffer aiming to be used as the storage of the `depth_map`, accessed
    /// through address mapping. It is used to compute the projected area of all
    /// visible facets.
    depth_attachment_storage: wgpu::Buffer,

    /// Color attachment of the render pass.
    color_attachment: Texture,

    /// Color attachment used to compute the projected whole area of all visible
    /// facets.
    color_attachment_storage: wgpu::Buffer,

    /// Width of color and depth attachment.
    attachment_width: u32,

    /// Height of color and depth attachment.
    attachment_height: u32,

    /// Size of depth storage in bytes.
    depth_attachment_storage_size: u64,

    // Size of color storage in bytes.
    color_attachment_storage_size: u64,
}

/// Uniforms used by the `OcclusionPass`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    /// Object model matrix. TODO: maybe better to use push constant
    model_matrix: [f32; 16],

    // Light space viewing and projection matrix.
    light_space_matrix: [f32; 16],
}

impl OcclusionEstimationPass {
    pub const COLOR_ATTACHMENT_TEXTURE_FORMAT: wgpu::TextureFormat =
        wgpu::TextureFormat::Rgba8Unorm;

    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let uniform_buffer_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("occlusion_pass_uniform_buffer"),
                contents: bytemuck::cast_slice(&[0.0f32; 32]),
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
            label: Some("occlusion_pass_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("occlusion_pass_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let vert_shader = ctx.device.create_shader_module(include_spirv!(
            "../app/assets/shaders/spirv/geom_term.vert.spv"
        ));
        let frag_shader = ctx.device.create_shader_module(include_spirv!(
            "../app/assets/shaders/spirv/geom_term.frag.spv"
        ));
        let depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            None,
            Some("shadow_pass_depth_attachment"),
        );
        let color_attachment = Texture::new(
            &ctx.device,
            &wgpu::TextureDescriptor {
                label: Some("occlusion_pass_color_attachment"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::COLOR_ATTACHMENT_TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            },
            None,
        );

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("occlusion_pass_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vert_shader,
                    entry_point: "main",
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
                    depth_compare: wgpu::CompareFunction::Always,
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
                    targets: &[Some(ColorTargetState {
                        format: Self::COLOR_ATTACHMENT_TEXTURE_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });
        let pass = RdrPass {
            pipeline,
            bind_groups: vec![bind_group],
            uniform_buffer: Some(uniform_buffer),
        };

        let depth_attachment_storage_size =
            (std::mem::size_of::<f32>() * (width * height) as usize) as wgpu::BufferAddress;

        let color_attachment_storage_size =
            (std::mem::size_of::<f32>() * (width * height) as usize) as wgpu::BufferAddress;

        let depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_pass_depth_attachment_storage"),
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let color_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_pass_color_attachment_storage"),
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            inner: pass,
            depth_attachment,
            depth_attachment_storage,
            color_attachment,
            color_attachment_storage,
            attachment_width: width,
            attachment_height: height,
            depth_attachment_storage_size,
            color_attachment_storage_size,
        }
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline { &self.inner.pipeline }

    pub fn bind_groups(&self) -> &[wgpu::BindGroup] { &self.inner.bind_groups }

    pub fn uniform_buffer(&self) -> Option<&wgpu::Buffer> { self.inner.uniform_buffer.as_ref() }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, model: Mat4, view_proj: Mat4) {
        queue.write_buffer(
            self.inner.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[model, view_proj]),
        );
    }

    pub fn run_once(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        v_buf: &wgpu::Buffer,
        i_buf: &wgpu::Buffer,
        i_count: u32,
        i_format: wgpu::IndexFormat,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("occlusion_pass_compute_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_attachment.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: true,
                    },
                })],
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
            render_pass.draw_indexed(0..i_count, 0, 0..1);
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
                buffer: &self.depth_attachment_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(
                        std::mem::size_of::<f32>() as u32 * self.attachment_width,
                    ),
                    rows_per_image: NonZeroU32::new(self.attachment_height),
                },
            },
            wgpu::Extent3d {
                width: self.attachment_width,
                height: self.attachment_height,
                depth_or_array_layers: 1,
            },
        );

        // Copy color attachment values to its storage.
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color_attachment.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.color_attachment_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(
                        std::mem::size_of::<u32>() as u32 * self.attachment_width,
                    ),
                    rows_per_image: NonZeroU32::new(self.attachment_height),
                },
            },
            wgpu::Extent3d {
                width: self.attachment_width,
                height: self.attachment_height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
    }

    pub fn save_depth_attachment(
        &self,
        device: &wgpu::Device,
        near: f32,
        far: f32,
        path: &std::path::Path,
    ) -> Result<(), Error> {
        {
            let slice = self.depth_attachment_storage.slice(..);
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
            ImageBuffer::<Luma<u8>, _>::from_raw(
                self.attachment_width,
                self.attachment_height,
                data,
            )
            .ok_or_else(|| {
                Error::Any(
                    "Failed to create image from depth map buffer, please check if the data have \
                     been transferred to the buffer!"
                        .into(),
                )
            })
            .and_then(|img| img.save(path).map_err(Error::from))?;
        }
        self.depth_attachment_storage.unmap();

        Ok(())
    }

    pub fn save_color_attachment(
        &self,
        device: &wgpu::Device,
        path: &std::path::Path,
    ) -> Result<(), Error> {
        {
            let slice = self.color_attachment_storage.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });

            let buffer_view = slice.get_mapped_range();
            use image::{ImageBuffer, Rgba};
            ImageBuffer::<Rgba<u8>, _>::from_raw(
                self.attachment_width,
                self.attachment_height,
                buffer_view,
            )
            .ok_or_else(|| {
                Error::Any(
                    "Failed to create image from depth map buffer, please check if the data have \
                     been transferred to the buffer!"
                        .into(),
                )
            })
            .and_then(|img| img.save(path).map_err(Error::from))?;
        }
        self.color_attachment_storage.unmap();

        Ok(())
    }

    fn calculate_area_with_occlusion(&self, device: &wgpu::Device) -> f32 {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        let count: u32 = {
            let slice = self.depth_attachment_storage.slice(..);
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
        self.depth_attachment_storage.unmap();
        count as f32
    }

    fn calculate_area_without_occlusion(&self, device: &wgpu::Device) -> f32 {
        let count = {
            let slice = self.color_attachment_storage.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });

            let data = slice.get_mapped_range();
            let mut count = 0.0f32;
            for i in (0..data.len()).step_by(4) {
                if data[i] != 0 {
                    // 64 possible levels
                    // (256 * 0.015625) = 4
                    count += data[i] as f32 * 0.25;
                }
            }
            count
        };
        self.color_attachment_storage.unmap();
        count
    }

    pub fn calculate_ratio(&self, device: &wgpu::Device) -> f32 {
        let area_with_occlusion = self.calculate_area_with_occlusion(device);
        let area_without_occlusion = self.calculate_area_without_occlusion(device);
        log::info!("Area with occlusion: {}", area_with_occlusion);
        log::info!("Area without occlusion: {}", area_without_occlusion);
        if area_without_occlusion == 0.0 {
            f32::NAN
        } else {
            area_with_occlusion / area_without_occlusion
        }
    }
}

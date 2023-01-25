use crate::{
    acq,
    acq::{measurement::MicrofacetShadowingMaskingMeasurement, Handedness},
    app::{
        cache::{Cache, Handle},
        gfx::{
            bytes_per_pixel,
            camera::{Camera, Projection},
            GpuContext, RenderPass, Texture, WgpuConfig,
        },
    },
    msurf::MicroSurface,
    units::Radians,
    Error,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rayon::prelude::IndexedParallelIterator;
use std::{fs::File, num::NonZeroU32, path::Path};
use wgpu::{util::DeviceExt, ColorTargetState};

/// Render pass computing the shadowing/masking (caused by occlusion of
/// micro-facets) function of a micro-surface. For a certain viewing direction,
/// this is done in two steps:
///
/// 1. Depth pass
///
/// Generates a depth buffer for the entire micro-surface. The depth buffer will
/// later be used to determine if the micro-facet is occluded by other
/// micro-facets that are not in visible in the direction of the viewing
/// direction.
///
/// 2. Render pass
///
/// Render view-dependent micro-facet shadowing/masking function using the
/// depth buffer from the depth pass.
///
/// Render all visible facets. Each fragment outputs value of 1/1024, then at
/// the blending stage, sum up; later stores inside of a texture.
pub struct OcclusionEstimator {
    /// Depth pass to generate depth buffer for the entire micro-surface.
    depth_pass: RenderPass,

    /// Uniform buffer shared by both passes.
    uniform_buffer: wgpu::Buffer,

    /// Render pass to compute the view-dependent micro-facet shadowing/masking
    /// function.
    render_pass: RenderPass,

    /// Width of color and depth attachment.
    attachment_width: u32,

    /// Height of color and depth attachment.
    attachment_height: u32,

    /// Color attachment of the render pass. Used to compute the projected area
    /// of all visible facets.
    color_attachment: Texture,

    /// Color attachment used to compute the projected whole area of all visible
    /// facets.
    color_attachment_storage: wgpu::Buffer,

    /// Depth attachment.
    depth_attachment: Texture,

    /// Depth attachment used to copy the depth buffer from the depth pass.
    depth_attachment_storage: wgpu::Buffer,
}

/// Uniforms used by the `OcclusionPass`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    proj_view_matrix: [f32; 16],
    viewport_resolution: [f32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            proj_view_matrix: Mat4::IDENTITY.to_cols_array(),
            viewport_resolution: [1.0, 1.0, 0.0, 0.0],
        }
    }
}

/// Occlusion estimation result.
#[derive(Debug, Clone, Copy)]
pub struct OcclusionEstimationResult {
    area_without_occlusion: u32,
    area_with_occlusion: u32,
}

impl OcclusionEstimationResult {
    pub fn visibility(&self) -> f32 {
        if self.area_without_occlusion == 0 {
            log::warn!("Area without occlusion is zero.");
            f32::NAN
        } else {
            self.area_with_occlusion as f32 / self.area_without_occlusion as f32
        }
    }
}

impl OcclusionEstimator {
    pub const COLOR_ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const DEPTH_ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Creates a new `OcclusionEstimator` with the given attachment size.
    ///
    /// The attachment size is the size of the color and depth attachment.
    /// The color attachment is used to compute the projected area of all
    /// visible facets.
    ///
    /// The depth attachment is used to determine if a facet is
    /// occluded by other facets in the direction of the
    /// viewing direction.
    ///
    /// # Arguments
    /// * `gpu_context` - GPU context.
    /// * `width` - Width of the color and depth attachment.
    /// * `height` - Height of the color and depth attachment.
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        log::info!(
            "Initialising occlusion estimator with attachment size {}x{} and format {:?}",
            width,
            height,
            Self::COLOR_ATTACHMENT_FORMAT
        );
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("./occlusion.wgsl"));
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
                format: Self::COLOR_ATTACHMENT_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            },
            None,
        );
        let depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            Some(Self::DEPTH_ATTACHMENT_FORMAT),
            None,
            Some("oc_depth_attachment"),
        );
        let uniform_buffer_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("oc_uniform_buffer"),
                contents: bytemuck::cast_slice(&[Uniforms::default()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let depth_pass = {
            let bind_group_layout =
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("oc_depth_pass_bind_group_layout"),
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
                label: Some("oc_depth_pass_bind_group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("oc_depth_pass_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("oc_depth_pass_pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &&shader_module,
                        entry_point: "vs_depth_pass",
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
                        depth_compare: wgpu::CompareFunction::LessEqual,
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

            RenderPass {
                pipeline,
                bind_groups: vec![bind_group],
                uniform_buffer: None,
            }
        };

        let render_pass = {
            let bind_group_layout =
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("oc_render_pass_bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX
                                    | wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(uniform_buffer_size),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Depth,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::Comparison,
                                ),
                                count: None,
                            },
                        ],
                    });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oc_render_pass_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_attachment.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&depth_attachment.sampler),
                    },
                ],
            });
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("oc_render_pass_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });
            let (blend, write_mask) =
                if Self::COLOR_ATTACHMENT_FORMAT == wgpu::TextureFormat::Rgb10a2Unorm {
                    // For capturing.
                    (
                        Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        Some(wgpu::ColorWrite::RED),
                    )
                } else {
                    // For debug purposes.
                    (Some(wgpu::BlendState::REPLACE), Some(wgpu::ColorWrite::ALL))
                };
            let pipeline = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("oc_render_pass_pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &&shader_module,
                        entry_point: "vs_render_pass",
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
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &&shader_module,
                        entry_point: "fs_render_pass",
                        targets: &[Some(ColorTargetState {
                            format: Self::COLOR_ATTACHMENT_FORMAT,
                            blend,
                            write_mask,
                        })],
                    }),
                    multiview: None,
                });
            RenderPass {
                pipeline,
                bind_groups: vec![bind_group],
                uniform_buffer: None,
            }
        };

        let color_attachment_storage_size =
            bytes_per_pixel(Self::COLOR_ATTACHMENT_FORMAT) as u64 * (width * height) as u64;

        let color_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_pass_color_attachment_storage"),
            size: color_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let depth_attachment_storage_size =
            bytes_per_pixel(Texture::DEPTH_FORMAT) as u64 * (width * height) as u64;
        let depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_pass_depth_attachment_storage"),
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            depth_pass,
            render_pass,
            uniform_buffer,
            color_attachment,
            color_attachment_storage,
            attachment_width: width,
            attachment_height: height,
            depth_attachment,
            depth_attachment_storage,
        }
    }

    pub fn uniform_buffer(&self) -> Option<&wgpu::Buffer> {
        self.depth_pass.uniform_buffer.as_ref()
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, proj_view_mat: Mat4) {
        let uniform = Uniforms {
            proj_view_matrix: proj_view_mat.to_cols_array(),
            viewport_resolution: [
                self.attachment_width as f32,
                self.attachment_height as f32,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    pub fn estimate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facets_verts_buf: &wgpu::Buffer,
        facets_indices_buf: &wgpu::Buffer,
        facets_indices_num: u32,
        visible_facets_index_buf: &wgpu::Buffer,
        visible_facets_index_num: u32,
        index_format: wgpu::IndexFormat,
    ) {
        // TODO: cache the depth buffer for all possible view directions
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.push_debug_group("oc_depth_pass");
        {
            // compute depth buffer
            encoder.insert_debug_marker("render entire micro-surface to depth buffer");
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("oc_depth_pass"),
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

            render_pass.set_pipeline(&self.depth_pass.pipeline);
            render_pass.set_bind_group(0, &self.depth_pass.bind_groups[0], &[]);
            render_pass.set_vertex_buffer(0, facets_verts_buf.slice(..));
            render_pass.set_index_buffer(facets_indices_buf.slice(..), index_format);
            render_pass.draw_indexed(0..facets_indices_num, 0, 0..1);
        }
        encoder.pop_debug_group();

        // Copy depth attachment values to its storage.
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
                        bytes_per_pixel(Self::DEPTH_ATTACHMENT_FORMAT) as u32
                            * self.attachment_width,
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

        encoder.push_debug_group("oc_render_pass");
        {
            // compute occulusions
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
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pass.pipeline);
            render_pass.set_bind_group(0, &self.render_pass.bind_groups[0], &[]);
            render_pass.set_vertex_buffer(0, facets_verts_buf.slice(..));
            // render_pass.set_index_buffer(visible_facets_index_buf.slice(..),
            // index_format); render_pass.draw_indexed(0..
            // visible_facets_index_num, 0, 0..1);
            render_pass.set_index_buffer(facets_indices_buf.slice(..), index_format);
            render_pass.draw_indexed(0..facets_indices_num, 0, 0..1);
        }
        encoder.pop_debug_group();

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
                        bytes_per_pixel(Self::COLOR_ATTACHMENT_FORMAT) as u32
                            * self.attachment_width,
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

    pub fn save_color_attachment(
        &self,
        device: &wgpu::Device,
        filename: &str,
        as_image: bool,
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

            if as_image {
                use image::{ImageBuffer, Rgba};
                let values = if Self::COLOR_ATTACHMENT_FORMAT == wgpu::TextureFormat::Rgb10a2Unorm {
                    let ratio = 255.0 / 1023.0;
                    buffer_view
                        .chunks(4)
                        .flat_map(|chunk| {
                            let v = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            [
                                ((v & 0x03FF) as f32 * ratio) as u8,             // r
                                (((v >> 10 & 0x03FF) as f32) * ratio) as u8,     // g
                                (((v >> 20 & 0x03FF) as f32) * ratio) as u8,     // b
                                (((v >> 30 & 0x03) as f32) * 255.0 / 3.0) as u8, // a
                            ]
                        })
                        .collect::<Vec<_>>()
                } else if Self::COLOR_ATTACHMENT_FORMAT == wgpu::TextureFormat::Rgba8Unorm {
                    buffer_view.to_vec()
                } else {
                    vec![]
                };

                ImageBuffer::<Rgba<u8>, _>::from_raw(
                    self.attachment_width,
                    self.attachment_height,
                    values,
                )
                .ok_or_else(|| {
                    Error::Any(
                        "Failed to create image from color attachment buffer, please check if the \
                         data have been transferred to the buffer!"
                            .into(),
                    )
                })
                .and_then(|img| img.save(format!("{filename}.png")).map_err(Error::from))?;
            } else {
                let values = if Self::COLOR::ATTACHMENT_FORMAT == wgpu::TextureFormat::Rbg10a2Unorm
                {
                    let values = buffer_view
                        .chunks(4)
                        .map(|chunk| {
                            let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            // Only take the last 10 bits which is the R channel.
                            val & 0x03FF
                        })
                        .collect::<Vec<_>>();
                } else if Self::COLOR_ATTACHMENT_FORMAT == wgpu::TextureFormat::Rgba8Unorm {
                    // Debug only, not used in the actual program.
                    buffer_view
                        .chunks(4)
                        .flat_map(|chunk| {
                            [
                                chunk[0], // r
                                chunk[1], // g
                                chunk[2], // b
                                chunk[3], // a
                            ]
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                };
                use std::io::Write;
                let mut file = File::create(format!("{filename}.txt"))?;
                for (i, val) in values.iter().enumerate() {
                    if i as u32 % self.attachment_width == 0 {
                        writeln!(file)?;
                    } else {
                        write!(file, "{} ", val)?;
                    }
                }
            }
        }
        self.color_attachment_storage.unmap();

        Ok(())
    }

    pub fn save_depth_attachment(
        &self,
        device: &wgpu::Device,
        filename: &str,
        as_image: bool,
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
            let data = {
                let (_, data, _) = unsafe { buffer_view.align_to::<f32>() };
                data
            };

            if as_image {
                use image::{ImageBuffer, Luma};
                ImageBuffer::<Luma<u8>, _>::from_raw(
                    self.attachment_width,
                    self.attachment_height,
                    data.iter().map(|&v| (v * 255.0) as u8).collect::<Vec<_>>(),
                )
                .ok_or_else(|| {
                    Error::Any(
                        "Failed to create image from depth buffer, please check if the data have \
                         been transferred to the buffer!"
                            .into(),
                    )
                })
                .and_then(|img| img.save(format!("{filename}.png")).map_err(Error::from))?;
            } else {
                use std::io::Write;
                let mut file = File::create(format!("{filename}.txt"))?;
                for (i, val) in data.iter().enumerate() {
                    if i as u32 % self.attachment_width == 0 {
                        writeln!(file)?;
                    } else {
                        write!(file, "{} ", val)?;
                    }
                }
            }
        }
        self.depth_attachment_storage.unmap();

        Ok(())
    }

    fn calculate_areas(&self, device: &wgpu::Device) -> OcclusionEstimationResult {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let (non_occluded_area, occluded_area) = {
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
            unsafe {
                buffer_view
                    .par_iter()
                    .chunks(4)
                    .fold(
                        || (0u32, 0u32),
                        |(non_blocked_area, blocked_area), bytes| {
                            let value =
                                u32::from_le_bytes([*bytes[0], *bytes[1], *bytes[2], *bytes[3]]);
                            if value != 0 {
                                (non_blocked_area + value, blocked_area + 1)
                            } else {
                                (non_blocked_area, blocked_area)
                            }
                        },
                    )
                    .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
            }
        };
        self.color_attachment_storage.unmap();
        OcclusionEstimationResult {
            area_without_occlusion: non_occluded_area,
            area_with_occlusion: occluded_area,
        }
    }
}

/// Structure holding the data for microfacet shadowing and masking measurement.
///
/// G(i, o, m) is the micro facet shadowing-masking function, which describes
/// the fraction of micro-facets with normal m that are visible from both the
/// incident direction i and the outgoing direction o.
///
/// The Smith microfacet shadowing-masking function is defined as:
/// G(i, o, m) = G1(i, m) * G1(o, m)
pub struct MicrofacetShadowingMaskingFunction {
    /// The bin size of azimuthal angle when sampling the microfacet
    /// distribution.
    pub azimuth_bin_size: Radians,
    /// The bin size of zenith angle when sampling the microfacet
    /// distribution.
    pub zenith_bin_size: Radians,
    /// The number of bins in the azimuthal angle.
    pub azimuth_bins_count: usize,
    /// The number of bins in the zenith angle.
    pub zenith_bins_count: usize,
    /// The distribution data. The first index is the azimuthal angle, and the
    /// second index is the zenith angle.
    pub samples: Vec<f32>,
}

impl MicrofacetShadowingMaskingFunction {
    /// Saves the microfacet shadowing and masking function in ascii format.
    pub fn save_ascii(&self, filepath: &Path) -> Result<(), std::io::Error> {
        use std::io::{BufWriter, Write};
        log::info!(
            "Saving microfacet shadowing and masking distribution in ascii format to {}",
            filepath.display()
        );
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(filepath)?;
        let mut writter = BufWriter::new(file);
        let header = format!(
            "microfacet shadowing and masking function\nazimuth - bin size: {}, bins count: \
             {}\nzenith - bin size: {}, bins count: {}\n",
            self.azimuth_bin_size.in_degrees().prettified(),
            self.azimuth_bins_count,
            self.zenith_bin_size.in_degrees().prettified(),
            self.zenith_bins_count
        );
        let _ = writter.write(header.as_bytes())?;
        self.samples.iter().for_each(|s| {
            let value = format!("{s} ");
            let _ = writter.write(value.as_bytes()).unwrap();
        });
        Ok(())
    }
}

pub fn measure_microfacet_shadowing_masking(
    desc: MicrofacetShadowingMaskingMeasurement,
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
    handedness: Handedness,
) -> Vec<MicrofacetShadowingMaskingFunction> {
    log::info!("Measuring microfacet shadowing and masking function...");
    let wgpu_config = WgpuConfig {
        device_descriptor: wgpu::DeviceDescriptor {
            label: Some("occlusion-measurement-device"),
            features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM | wgpu::Features::POLYGON_MODE_LINE,
            limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
        },
        target_format: Some(OcclusionEstimator::COLOR_ATTACHMENT_FORMAT),
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    };
    let gpu = pollster::block_on(GpuContext::offscreen(&wgpu_config));
    let estimator = OcclusionEstimator::new(&gpu, desc.resolution, desc.resolution);
    let surfaces = cache.get_micro_surface_meshes(surfaces);
    let mut final_results = Vec::with_capacity(surfaces.len());
    surfaces.iter().for_each(|surface| {
        let mut results =
            Vec::with_capacity((desc.azimuth.step_count() * (desc.zenith.step_count() + 1)).pow(2));
        let facets_verts_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&surface.verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let facets_indices_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&surface.facets),
                usage: wgpu::BufferUsages::INDEX,
            });
        let diagnal =
            (surface.extent.max - surface.extent.min).max_element() * std::f32::consts::SQRT_2;
        // let half_zenith_bin_size_cos = (desc.zenith.step_size / 2.0).cos();
        let projection = Projection::orthographic_matrix(0.1, diagnal * 1.5, diagnal, diagnal);
        for azimuth_idx in 0..desc.azimuth.step_count() {
            for zenith_idx in 0..desc.zenith.step_count() + 1 {
                let azimuth = azimuth_idx as f32 * desc.azimuth.step_size;
                let zenith = zenith_idx as f32 * desc.zenith.step_size;
                let index = azimuth_idx * (desc.zenith.step_count() + 1) + zenith_idx;
                let view_matrix = Camera::new(
                    acq::spherical_to_cartesian(diagnal, zenith, azimuth, handedness),
                    Vec3::ZERO,
                    Vec3::Y,
                )
                .matrix();
                estimator.update_uniforms(&gpu.queue, projection * view_matrix);
                let view_dir =
                    acq::spherical_to_cartesian(1.0, zenith, azimuth, Handedness::RightHandedYUp)
                        .normalize();
                let visible_facets = surface
                    .facet_normals
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, normal)| {
                        //if normal.dot(view_dir) >= half_zenith_bin_size_cos {
                        if normal.dot(view_dir) >= 0.0 {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                let visible_facets_indices = visible_facets
                    .iter()
                    .flat_map(|idx| {
                        [
                            surface.facets[idx * 3],
                            surface.facets[idx * 3 + 1],
                            surface.facets[idx * 3 + 2],
                        ]
                    })
                    .collect::<Vec<_>>();

                let visible_facets_index_buffer =
                    gpu.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("shadowing-masking-index-buffer"),
                            contents: bytemuck::cast_slice(&visible_facets_indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                estimator.estimate(
                    &gpu.device,
                    &gpu.queue,
                    &facets_verts_buf,
                    &facets_indices_buf,
                    surface.facets.len() as u32,
                    &visible_facets_index_buffer,
                    visible_facets_indices.len() as u32,
                    wgpu::IndexFormat::Uint32,
                );
                let filename = format!("{index:06}_roi_color");
                estimator
                    .save_color_attachment(&gpu.device, &filename, true)
                    .unwrap();
                let filename = format!("{index:06}_roi_depth");
                estimator
                    .save_depth_attachment(&gpu.device, &filename, true)
                    .unwrap();
                let result = estimator.calculate_areas(&gpu.device);
                log::trace!(
                    "azimuth: {}, zenith: {}, visible facets count: {} ==> {}",
                    azimuth.prettified(),
                    zenith.prettified(),
                    visible_facets.len(),
                    result.visibility()
                );

                let mut visibilities =
                    vec![0.0; desc.azimuth.step_count() * (desc.zenith.step_count() + 1)];
                // for inner_azimuth_idx in 0..desc.azimuth.step_count() {
                //     for inner_zenith_idx in 0..desc.zenith.step_count() + 1 {
                //         let inner_azimuth = inner_azimuth_idx as f32 *
                // desc.azimuth.step_size;         let inner_zenith =
                // inner_zenith_idx as f32 * desc.zenith.step_size;         let
                // inner_view_matrix = Camera::new(
                // acq::spherical_to_cartesian(                 radius,
                //                 inner_zenith,
                //                 inner_azimuth,
                //                 handedness,
                //             ),
                //             Vec3::ZERO,
                //             Vec3::Y,
                //         )
                //         .matrix();
                //         estimation.update_uniforms(&gpu.queue, projection *
                // inner_view_matrix);         let inner_view_dir =
                // acq::spherical_to_cartesian(             1.0,
                //             inner_zenith,
                //             inner_azimuth,
                //             Handedness::RightHandedYUp,
                //         );
                //         let inner_visible_facets = visible_facets
                //             .iter()
                //             .filter_map(|idx| {
                //                 if surface.facet_normals[*idx].dot(inner_view_dir) >= 0.0 {
                //                     Some(*idx)
                //                 } else {
                //                     None
                //                 }
                //             })
                //             .collect::<Vec<_>>();
                //         let inner_visible_facets_indices = inner_visible_facets
                //             .iter()
                //             .flat_map(|idx| {
                //                 [
                //                     surface.facets[*idx * 3],
                //                     surface.facets[*idx * 3 + 1],
                //                     surface.facets[*idx * 3 + 2],
                //                 ]
                //             })
                //             .collect::<Vec<_>>();
                //         let inner_visible_facets_index_buffer =
                //             gpu.device
                //                 .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                //                     label: Some("shadowing-masking-index-buffer"),
                //                     contents:
                // bytemuck::cast_slice(&inner_visible_facets_indices),
                //                     usage: wgpu::BufferUsages::INDEX,
                //                 });
                //         estimation.run_once(
                //             &gpu.device,
                //             &gpu.queue,
                //             &vertex_buffer,
                //             &inner_visible_facets_index_buffer,
                //             inner_visible_facets_indices.len() as u32,
                //             wgpu::IndexFormat::Uint32,
                //         );
                //         visibilities[inner_azimuth_idx * (desc.zenith.step_count() + 1)
                //             + inner_zenith_idx] =
                //             estimation.calculate_areas(&gpu.device).visibility();
                //     }
                // }
                results.append(&mut visibilities)
            }
        }
        final_results.push(MicrofacetShadowingMaskingFunction {
            azimuth_bin_size: desc.azimuth.step_size,
            zenith_bin_size: desc.zenith.step_size,
            azimuth_bins_count: desc.azimuth.step_count(),
            zenith_bins_count: desc.zenith.step_count() + 1,
            samples: results,
        });
    });
    final_results
}

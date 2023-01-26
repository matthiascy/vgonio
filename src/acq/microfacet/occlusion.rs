use crate::{
    acq,
    acq::{measurement::MicrofacetShadowingMaskingMeasurement, Handedness},
    app::{
        cache::{Cache, Handle},
        gfx::{
            self,
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
use std::{num::NonZeroU32, ops::Deref, path::Path};
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

    /// Measurement points. Stores an array of [`MeasurementPoint`]s.
    measurement_points_buffer: wgpu::Buffer,

    /// Number of measurement points.
    num_measurement_points: u32,

    /// Width of color and depth attachment.
    attachment_width: u32,

    /// Height of color and depth attachment.
    attachment_height: u32,

    /// Color attachment used to compute the projected area of all visible
    /// facets.
    color_attachment: ColorAttachment,

    /// Depth buffers of all micro-facets at all possible measurement points.
    depth_attachment: DepthAttachment,
}

/// Uniforms used by the `OcclusionPass`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    proj_view_matrix: [f32; 16],
    meas_point_index: [u32; 4],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            proj_view_matrix: Mat4::IDENTITY.to_cols_array(),
            meas_point_index: [0, 0, 0, 0],
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
    /// Texture format used for the color attachment.
    pub const COLOR_ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    /// Bytes per pixel of the color attachment.
    pub const COLOR_ATTACHMENT_FORMAT_BPP: u32 =
        gfx::tex_fmt_bpp(Self::COLOR_ATTACHMENT_FORMAT) as u32;

    /// Texture format used for the depth attachment.
    pub const DEPTH_ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    /// Bytes per pixel of the depth attachment.
    pub const DEPTH_ATTACHMENT_FORMAT_BPP: u32 =
        gfx::tex_fmt_bpp(Self::DEPTH_ATTACHMENT_FORMAT) as u32;
    /// Index buffer format.
    pub const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;

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
    /// * `meas_count` - Number of measurement points.
    pub fn new(ctx: &GpuContext, width: u32, height: u32, meas_count: u32) -> Self {
        log::info!(
            "Initialising occlusion estimator with attachment size {}x{} and format {:?}",
            width,
            height,
            Self::COLOR_ATTACHMENT_FORMAT
        );
        let measurement_points_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion-estimator-measurement-points"),
            size: MeasurementPointGpuData::SIZE_IN_BYTES * meas_count as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("./occlusion.wgsl"));
        let color_attachment = ColorAttachment::new(
            &ctx.device,
            width,
            height,
            meas_count,
            Self::COLOR_ATTACHMENT_FORMAT,
        );
        let depth_attachment = DepthAttachment::new(
            &ctx.device,
            width,
            height,
            meas_count,
            Self::DEPTH_ATTACHMENT_FORMAT,
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
                                    // For `texture_2d`
                                    // wgpu::TextureSampleType::Float {
                                    //     filterable: false,
                                    // },
                                    view_dimension: wgpu::TextureViewDimension::D2Array,
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
                                // wgpu::BindingType::Sampler(
                                //     wgpu::SamplerBindingType::NonFiltering,
                                // ),
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
                        resource: wgpu::BindingResource::TextureView(
                            &depth_attachment.whole_view(),
                        ),
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
                        wgpu::ColorWrites::RED,
                    )
                } else {
                    // For debug purposes.
                    (Some(wgpu::BlendState::REPLACE), wgpu::ColorWrites::ALL)
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

        Self {
            depth_pass,
            render_pass,
            uniform_buffer,
            color_attachment,
            attachment_width: width,
            attachment_height: height,
            depth_attachment,
            measurement_points_buffer,
            num_measurement_points: meas_count,
        }
    }

    /// Updates the measurement points buffer.
    ///
    /// Must be called before `bake_depth_maps` and `estimate`.
    pub fn update_measurement_points(
        &mut self,
        queue: &wgpu::Queue,
        meas_points: &[MeasurementPoint],
    ) {
        assert_eq!(
            meas_points.len() as u32,
            self.num_measurement_points,
            "The number of measurement points must be the same as the number of measurement \
             points used to create the `OcclusionEstimator`."
        );
        let data = meas_points
            .iter()
            .map(|mp| (*mp).into())
            .collect::<Vec<MeasurementPointGpuData>>();
        queue.write_buffer(
            &self.measurement_points_buffer,
            0,
            bytemuck::cast_slice(data.as_slice()),
        );
    }

    /// Bakes the depth buffer into the depth attachment texture.
    ///
    /// Must be called after `update_measurement_points`.
    pub fn bake(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facets_vtx_buf: &wgpu::Buffer,
        facets_idx_buf: &wgpu::Buffer,
        facets_idx_num: u32,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("occlusion_pass_bake_depth_maps_encoder"),
        });

        encoder.push_debug_group("oc_depth_maps_bake");

        for i in 0..self.num_measurement_points {
            encoder.push_debug_group(&format!("oc_depth_maps_bake_{}", i));
            encoder.copy_buffer_to_buffer(
                &self.measurement_points_buffer,
                i as u64 * MeasurementPointGpuData::SIZE_IN_BYTES,
                &self.uniform_buffer,
                0,
                MeasurementPointGpuData::SIZE_IN_BYTES,
            );
            encoder.insert_debug_marker("bake all facets");
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("occlusion_pass_bake_depth_maps_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_attachment.layer_view(i as u32),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });
                pass.set_pipeline(&self.depth_pass.pipeline);
                pass.set_bind_group(0, &self.depth_pass.bind_groups[0], &[]);
                pass.set_vertex_buffer(0, facets_vtx_buf.slice(..));
                pass.set_index_buffer(facets_idx_buf.slice(..), Self::INDEX_FORMAT);
                pass.draw_indexed(0..facets_idx_num, 0, 0..1);
            }
            encoder.pop_debug_group();
        }
        encoder.pop_debug_group();

        self.depth_attachment.copy_to_storage(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    /// Estimate the occlusion for visible facets.
    ///
    /// The `bake` method must be called before this method.
    pub fn estimate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        facets_vtx_buf: &wgpu::Buffer,
        visible_facets_idx_buf: &wgpu::Buffer,
        visible_facets_idx_num: u32,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("occlusion_estimation_render_encoder"),
        });

        encoder.push_debug_group("oc_render_pass");
        for i in 0..self.num_measurement_points {
            encoder.push_debug_group(&format!("oc_render_pass_{}", i));
            encoder.copy_buffer_to_buffer(
                &self.measurement_points_buffer,
                i as u64 * MeasurementPointGpuData::SIZE_IN_BYTES,
                &self.uniform_buffer,
                0,
                MeasurementPointGpuData::SIZE_IN_BYTES,
            );
            encoder.insert_debug_marker("render all visible facets");
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("occlusion_pass_compute_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.color_attachment.layer_view(i as u32),
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
                render_pass.set_vertex_buffer(0, facets_vtx_buf.slice(..));
                render_pass.set_index_buffer(visible_facets_idx_buf.slice(..), Self::INDEX_FORMAT);
                render_pass.draw_indexed(0..visible_facets_idx_num, 0, 0..1);
            }
            encoder.pop_debug_group();
        }
        encoder.pop_debug_group();

        self.color_attachment.copy_to_storage(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    pub fn save_color_attachment(
        &self,
        device: &wgpu::Device,
        dir: &Path,
        as_image: bool,
    ) -> Result<(), Error> {
        self.color_attachment.save(device, dir, as_image)?;
        Ok(())
    }

    pub fn save_depth_attachment(
        &self,
        device: &wgpu::Device,
        dir: &Path,
        as_image: bool,
    ) -> Result<(), Error> {
        self.depth_attachment.save(device, dir, as_image)?;
        Ok(())
    }

    // fn calculate_areas(&self, device: &wgpu::Device) -> OcclusionEstimationResult
    // {     use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    //     let (non_occluded_area, occluded_area) = {
    //         let slice = self.color_attachment_storage.slice(..);
    //         let (sender, receiver) =
    // futures_intrusive::channel::shared::oneshot_channel();         slice.
    // map_async(wgpu::MapMode::Read, move |result| {
    // sender.send(result).unwrap();         });
    //         device.poll(wgpu::Maintain::Wait);
    //         pollster::block_on(async {
    //             receiver.receive().await.unwrap().unwrap();
    //         });

    //         let buffer_view = slice.get_mapped_range();
    //         unsafe {
    //             buffer_view
    //                 .par_iter()
    //                 .chunks(4)
    //                 .fold(
    //                     || (0u32, 0u32),
    //                     |(non_blocked_area, blocked_area), bytes| {
    //                         let value =
    //                             u32::from_le_bytes([*bytes[0], *bytes[1],
    // *bytes[2], *bytes[3]]);                         if value != 0 {
    //                             (non_blocked_area + value, blocked_area + 1)
    //                         } else {
    //                             (non_blocked_area, blocked_area)
    //                         }
    //                     },
    //                 )
    //                 .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    //         }
    //     };
    //     self.color_attachment_storage.unmap();
    //     OcclusionEstimationResult {
    //         area_without_occlusion: non_occluded_area,
    //         area_with_occlusion: occluded_area,
    //     }
    // }
}

/// An array of 2D textures.
pub struct LayeredAttachment {
    /// The type of texture used for the depth attachment.
    pub format: wgpu::TextureFormat,
    /// The texture.
    pub texture: wgpu::Texture,
    /// The texture views of each layer of the texture.
    /// The first view is the whole texture. The other views are the layers.
    pub views: Vec<wgpu::TextureView>,
    /// The extent of the texture.
    pub extent: wgpu::Extent3d,
    /// The number of bytes of each layer of the texture.
    pub layer_size_in_bytes: u64,
}

impl LayeredAttachment {
    /// Creates a new layered texture.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - The width of the texture.
    /// * `height` - The height of the texture.
    /// * `layers` - The number of layers of the texture.
    /// * `format` - The format of the texture.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        layers: u32,
        format: wgpu::TextureFormat,
        label: Option<&str>,
    ) -> Self {
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        });
        let mut views = Vec::with_capacity(layers as usize + 1);
        views.push(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        for i in 0..layers {
            views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("oc_depth_texture_layer_view"),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i,
                array_layer_count: NonZeroU32::new(1),
            }));
        }
        let layer_size_in_bytes = (width * height * format.describe().block_size as u32) as u64;
        let size = layers as u64 * layer_size_in_bytes;
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oc_depth_texture_storage"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            format,
            texture,
            views,
            extent,
            layer_size_in_bytes,
        }
    }

    /// Copies the texture data to the storage.
    pub fn copy_to_buffer(&self, encoder: &mut wgpu::CommandEncoder, buffer: &wgpu::Buffer) {
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(
                        self.extent.width * gfx::tex_fmt_bpp(self.format),
                    ),
                    rows_per_image: NonZeroU32::new(self.extent.height),
                },
            },
            self.extent,
        );
    }

    /// Returns the texture view of the whole texture.
    pub fn whole_view(&self) -> &wgpu::TextureView { &self.views[0] }

    /// Returns the texture view of the specified layer.
    pub fn layer_view(&self, layer: u32) -> &wgpu::TextureView { &self.views[layer as usize + 1] }

    /// Returns the number of layers of the texture.
    pub fn layers(&self) -> u32 { self.extent.depth_or_array_layers }
}

/// Depth attachment used for occlusion estimation, which is a layered texture
/// with the same size as the color attachment; each layer is a depth
/// map of the micro-surface at a specific angle.
struct DepthAttachment {
    /// The texture used for the depth attachment.
    inner: LayeredAttachment,
    /// The sampler used to sample the depth map in the shader.
    sampler: wgpu::Sampler,
    /// The storage of the texture data in case the data need to be saved.
    storage: wgpu::Buffer,
}

impl Deref for DepthAttachment {
    type Target = LayeredAttachment;

    fn deref(&self) -> &Self::Target { &self.inner }
}

/// Color attachment used for occlusion estimation, which is a texture with the
/// same size as the depth attachment. The number of non-zero pixels is the area
/// of micro-facets together with the occlusion. The sum of values of all pixels
/// is the area of micro-facets in total without occlusion.
struct ColorAttachment {
    /// The texture used for the color attachment.
    inner: LayeredAttachment,
    /// The storage of the texture data in case the data need to be saved.
    storage: wgpu::Buffer,
}

impl Deref for ColorAttachment {
    type Target = LayeredAttachment;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl DepthAttachment {
    /// Creates a new depth attachment.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - The width of the texture.
    /// * `height` - The height of the texture.
    /// * `layers` - The number of layers of the texture.
    /// * `format` - The format of the depth texture.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        layers: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            // Depth compare function if using textureSampleCompare otherwise None.
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let inner = LayeredAttachment::new(
            device,
            width,
            height,
            layers,
            format,
            Some("oc_depth_texture"),
        );
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oc_depth_texture_storage"),
            size: inner.layer_size_in_bytes * layers as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            inner,
            sampler,
            storage,
        }
    }

    /// Copies the texture data to the storage.
    pub fn copy_to_storage(&self, encoder: &mut wgpu::CommandEncoder) {
        self.inner.copy_to_buffer(encoder, &self.storage);
    }

    /// Save the attachment data to files.
    pub fn save(&self, device: &wgpu::Device, dir: &Path, as_image: bool) -> Result<(), Error> {
        {
            let buffer = self.storage.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });
            let buffer_view = buffer.get_mapped_range();
            let (_, data, _) = unsafe { buffer_view.align_to::<f32>() };

            if !dir.exists() {
                std::fs::create_dir_all(dir).unwrap();
            }

            let num_pixels = (self.extent.width * self.extent.height) as usize;

            if as_image {
                use image::{ImageBuffer, Luma};
                for (i, layer) in data.chunks_exact(num_pixels).enumerate() {
                    let mut img = ImageBuffer::new(self.extent.width, self.extent.height);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let index = (y * self.extent.width + x) as usize;
                        *pixel = Luma([(layer[index] * 255.0) as u8]);
                    }
                    let path = dir.join(format!("depth_layer_{i:04}.png"));
                    img.save(path)?;
                }
            } else {
                use std::io::Write;
                for (i, layer) in data.chunks_exact(num_pixels).enumerate() {
                    let mut file =
                        std::fs::File::create(dir.join(format!("depth_layer_{i:04}.txt"))).unwrap();
                    for (j, val) in layer.iter().enumerate() {
                        if j % self.extent.width as usize == 0 {
                            writeln!(file, "{val}")?;
                        } else {
                            write!(file, "{val} ")?;
                        }
                    }
                }
            }
        }
        self.storage.unmap();
        Ok(())
    }
}

impl ColorAttachment {
    /// Creates a new color attachment.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - The width of the texture.
    /// * `height` - The height of the texture.
    /// * `layers` - The number of layers of the texture.
    /// * `format` - The format of the depth texture.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        layers: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        assert!(
            format == wgpu::TextureFormat::Rgba8Unorm
                || format == wgpu::TextureFormat::Rgb10a2Unorm,
            "Unsupported color texture format. Only Rgba8Unorm and Rgb10a2Unorm are supported."
        );
        let inner = LayeredAttachment::new(
            device,
            width,
            height,
            layers,
            format,
            Some("oc_color_texture"),
        );
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oc_color_texture_storage"),
            size: inner.layer_size_in_bytes * layers as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { inner, storage }
    }

    /// Copies the texture data to the storage.
    pub fn copy_to_storage(&self, encoder: &mut wgpu::CommandEncoder) {
        self.inner.copy_to_buffer(encoder, &self.storage);
    }

    /// Save the color attachment data to files.
    pub fn save(&self, device: &wgpu::Device, dir: &Path, as_image: bool) -> Result<(), Error> {
        {
            let is_8bit = self.inner.format == wgpu::TextureFormat::Rgba8Unorm;

            let buffer = self.storage.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });
            let buffer_view = buffer.get_mapped_range();

            if !dir.exists() {
                std::fs::create_dir_all(dir).unwrap();
            }

            if as_image {
                use image::{ImageBuffer, Rgba};
                for (i, layer) in buffer_view
                    .chunks_exact(self.inner.layer_size_in_bytes as usize)
                    .enumerate()
                {
                    if is_8bit {
                        let img = ImageBuffer::<Rgba<u8>, _>::from_raw(
                            self.extent.width,
                            self.extent.height,
                            layer,
                        )
                        .ok_or(Error::Any("Failed to create image buffer".to_string()))?;
                        img.save(dir.join(format!("color_layer_{i:04}.png")))?;
                    } else {
                        let ratio = 255.0 / 1023.0;
                        let img = ImageBuffer::<Rgba<u8>, _>::from_raw(
                            self.extent.width,
                            self.extent.height,
                            layer
                                .chunks(4)
                                .flat_map(|chunk| {
                                    let rgba = u32::from_le_bytes([
                                        chunk[0], chunk[1], chunk[2], chunk[3],
                                    ]);
                                    [
                                        (((rgba >> 0) & 0x3ff) as f32 * ratio) as u8,      // r
                                        (((rgba >> 10) & 0x3ff) as f32 * ratio) as u8,     // g
                                        (((rgba >> 20) & 0x3ff) as f32 * ratio) as u8,     // b
                                        (((rgba >> 30) & 0x3) as f32 * 255.0 / 3.0) as u8, // a
                                    ]
                                })
                                .collect::<Vec<_>>(),
                        )
                        .ok_or(Error::Any("Failed to create image buffer".to_string()))?;
                        img.save(dir.join(format!("color_layer_{i:04}.png")))?;
                    };
                }
            } else {
                // Only write R channel.
                use std::io::Write;
                for (i, layer) in buffer_view
                    .chunks_exact(self.inner.layer_size_in_bytes as usize)
                    .enumerate()
                {
                    let mut file =
                        std::fs::File::create(dir.join(format!("color_layer_{i:04}.txt"))).unwrap();
                    if is_8bit {
                        layer
                            .chunks(4)
                            .map(|chunk| chunk[0])
                            .enumerate()
                            .for_each(|(j, val)| {
                                if j % self.extent.width as usize == 0 {
                                    writeln!(file, "{val}")
                                } else {
                                    write!(file, "{val} ")
                                }
                                .expect("Failed to write to file");
                            });
                    } else {
                        layer
                            .chunks(4)
                            .map(|chunk| {
                                let rgba =
                                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                                rgba & 0x3ff
                            })
                            .enumerate()
                            .for_each(|(j, val)| {
                                if j % self.extent.width as usize == 0 {
                                    writeln!(file, "{val}")
                                } else {
                                    write!(file, "{val} ")
                                }
                                .expect("Failed to write to file");
                            });
                    }
                }
            }
        }
        self.storage.unmap();
        Ok(())
    }
}

/// Information about measurement location.
#[derive(Debug, Clone, Copy)]
pub struct MeasurementPoint {
    /// The direction from the center to the measurement location.
    pub view_dir: Vec3,
    /// The azimuth of the measurement location.
    pub azimuth: Radians,
    /// The inclination of the measurement location.
    pub zenith: Radians,
    /// The view and projection matrix for the measurement location.
    pub proj_view_mat: Mat4,
    /// The index of the measurement location.
    pub index: u32, // y, z, w are unused, just for alignment
}

/// Information about measurement location.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MeasurementPointGpuData {
    proj_view_mat: [f32; 16],
    index: [u32; 4],
}

impl From<MeasurementPoint> for MeasurementPointGpuData {
    fn from(point: MeasurementPoint) -> Self {
        Self {
            proj_view_mat: point.proj_view_mat.to_cols_array(),
            index: [point.index, 0, 0, 0],
        }
    }
}

impl MeasurementPointGpuData {
    const SIZE_IN_BYTES: wgpu::BufferAddress = std::mem::size_of::<Self>() as u64;
}

impl MeasurementPoint {
    /// Creates a new measurement point.
    pub fn new(
        azimuth: Radians,
        zenith: Radians,
        view_dir: Vec3,
        proj_view_mat: Mat4,
        index: u32,
    ) -> Self {
        Self {
            view_dir,
            azimuth,
            zenith,
            proj_view_mat,
            index,
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
    let mut estimator = OcclusionEstimator::new(
        &gpu,
        desc.resolution,
        desc.resolution,
        desc.measurement_location_count() as u32,
    );
    let surfaces = cache.get_micro_surface_meshes(surfaces);
    let mut final_results = Vec::with_capacity(surfaces.len());
    surfaces.iter().for_each(|surface| {
        // let mut results =
        //     Vec::with_capacity((desc.azimuth.step_count() * (desc.zenith.step_count()
        // + 1)).pow(2));
        let facets_vtx_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&surface.verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let facets_idx_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                contents: bytemuck::cast_slice(&surface.facets),
                usage: wgpu::BufferUsages::INDEX,
            });
        let facets_idx_num = surface.facets.len() as u32;
        let diagonal =
            (surface.extent.max - surface.extent.min).max_element() * std::f32::consts::SQRT_2;
        // let half_zenith_bin_size_cos = (desc.zenith.step_size / 2.0).cos();
        let proj_mat =
            Projection::orthographic_matrix(diagonal * 0.5, diagonal * 1.5, diagonal, diagonal);

        let meas_points = (0..desc.measurement_location_count())
            .into_iter()
            .map(|i| {
                let azimuth = desc.azimuth.step_size * (i / desc.zenith_step_count()) as f32;
                let zenith = desc.zenith.step_size * (i % desc.zenith_step_count()) as f32;
                let view_dir = acq::spherical_to_cartesian(1.0, zenith, azimuth, handedness);
                let pos = view_dir * diagonal;
                let proj_view_mat = {
                    let view_mat = Camera::new(pos, Vec3::ZERO, handedness.up()).view_matrix();
                    proj_mat * view_mat
                };
                MeasurementPoint::new(azimuth, zenith, view_dir, proj_view_mat, i as u32)
            })
            .collect::<Vec<_>>();

        estimator.update_measurement_points(&gpu.queue, &meas_points);
        estimator.bake(
            &gpu.device,
            &gpu.queue,
            &facets_vtx_buf,
            &facets_idx_buf,
            facets_idx_num,
        );

        // Save depth attachment for debugging
        estimator
            .save_depth_attachment(&gpu.device, Path::new("debug_depth_maps"), true)
            .expect("Failed to save depth attachment");

        #[cfg(not(debug_assertions))]
        {
            for meas_point in meas_points {
                let view_dir = Vec3::new(
                    meas_point.view_dir[0],
                    meas_point.view_dir[1],
                    meas_point.view_dir[2],
                );

                let visible_facets_indices = surface
                    .facet_normals
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, normal)| {
                        //if normal.dot(view_dir) >= half_zenith_bin_size_cos
                        if normal.dot(view_dir) >= 0.0 {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .map(|idx| {
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
                    &facets_vtx_buf,
                    &visible_facets_index_buffer,
                    visible_facets_indices.len() as u32,
                );
                let dir = format!(
                    "occlusion_maps_{}_{}_{}",
                    meas_point.index,
                    meas_point.azimuth.prettified(),
                    meas_point.zenith.prettified()
                );
                estimator
                    .save_color_attachment(&gpu.device, Path::new(&dir), true)
                    .expect("Failed to save color attachment");
            }
        }

        #[cfg(debug_assertions)]
        {
            for meas_point in meas_points {
                let visible_facets_indices = surface
                    .facet_normals
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, normal)| {
                        //if normal.dot(view_dir) >= half_zenith_bin_size_cos
                        if normal.dot(meas_point.view_dir) >= 0.0 {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .map(|idx| {
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
                    &facets_vtx_buf,
                    &facets_idx_buf,
                    facets_idx_num,
                    // &visible_facets_index_buffer,
                    // visible_facets_indices.len() as u32,
                );
                let dir = format!(
                    "debug_facets_{}_{}_{}",
                    meas_point.index,
                    meas_point.azimuth.prettified(),
                    meas_point.zenith.prettified()
                );
                estimator
                    .save_color_attachment(&gpu.device, Path::new(&dir), true)
                    .expect("Failed to save color attachment");
            }

            estimator.estimate(
                &gpu.device,
                &gpu.queue,
                &facets_vtx_buf,
                &facets_idx_buf,
                facets_idx_num,
            );
            estimator
                .save_color_attachment(&gpu.device, Path::new("debug_occlusion_maps"), true)
                .expect("Failed to save color attachment");
        }

        // for azimuth_idx in 0..desc.azimuth.step_count() {
        //     for zenith_idx in 0..desc.zenith.step_count() + 1 {
        //         let azimuth = azimuth_idx as f32 * desc.azimuth.step_size;
        //         let zenith = zenith_idx as f32 * desc.zenith.step_size;
        //         let index = azimuth_idx * (desc.zenith.step_count() + 1) +
        // zenith_idx;         let view_matrix = Camera::new(
        //             acq::spherical_to_cartesian(diagnal, zenith, azimuth,
        // handedness),             Vec3::ZERO,
        //             handedness.up(),
        //         )
        //         .matrix();
        //         estimator.update_uniforms(&gpu.queue, proj_mat *
        // view_matrix);         let view_dir =
        //             acq::spherical_to_cartesian(1.0, zenith, azimuth,
        // Handedness::RightHandedYUp)                 .normalize();
        //         let visible_facets = surface
        //             .facet_normals
        //             .iter()
        //             .enumerate()
        //             .filter_map(|(idx, normal)| {
        //                 //if normal.dot(view_dir) >= half_zenith_bin_size_cos
        // {                 if normal.dot(view_dir) >= 0.0 {
        //                     Some(idx)
        //                 } else {
        //                     None
        //                 }
        //             })
        //             .collect::<Vec<_>>();

        //         let visible_facets_indices = visible_facets
        //             .iter()
        //             .flat_map(|idx| {
        //                 [
        //                     surface.facets[idx * 3],
        //                     surface.facets[idx * 3 + 1],
        //                     surface.facets[idx * 3 + 2],
        //                 ]
        //             })
        //             .collect::<Vec<_>>();

        //         let visible_facets_index_buffer =
        //             gpu.device
        //                 .create_buffer_init(&wgpu::util::BufferInitDescriptor
        // {                     label:
        // Some("shadowing-masking-index-buffer"),
        // contents: bytemuck::cast_slice(&visible_facets_indices),
        //                     usage: wgpu::BufferUsages::INDEX,
        //                 });
        //         estimator.estimate(
        //             &gpu.device,
        //             &gpu.queue,
        //             &facets_verts_buf,
        //             &facets_indices_buf,
        //             surface.facets.len() as u32,
        //             &visible_facets_index_buffer,
        //             visible_facets_indices.len() as u32,
        //             wgpu::IndexFormat::Uint32,
        //         );
        //         let filename = format!("{index:06}_roi_color");
        //         estimator
        //             .save_color_attachment(&gpu.device, &filename, true)
        //             .unwrap();
        //         let dir = format!("{index:06}_roi_depth");
        //         estimator
        //             .save_depth_attachment(&gpu.device, Path::new(&dir),
        // true)             .unwrap();
        //         let result = estimator.calculate_areas(&gpu.device);
        //         log::trace!(
        //             "azimuth: {}, zenith: {}, visible facets count: {} ==>
        // {}",             azimuth.prettified(),
        //             zenith.prettified(),
        //             visible_facets.len(),
        //             result.visibility()
        //         );

        //         let mut visibilities =
        //             vec![0.0; desc.azimuth.step_count() *
        // (desc.zenith.step_count() + 1)];         // for
        // inner_azimuth_idx in 0..desc.azimuth.step_count() {         //
        // for inner_zenith_idx in 0..desc.zenith.step_count() + 1 {
        //         //         let inner_azimuth = inner_azimuth_idx as f32 *
        //         // desc.azimuth.step_size;         let inner_zenith =
        //         // inner_zenith_idx as f32 * desc.zenith.step_size;
        // let         // inner_view_matrix = Camera::new(
        //         // acq::spherical_to_cartesian(                 radius,
        //         //                 inner_zenith,
        //         //                 inner_azimuth,
        //         //                 handedness,
        //         //             ),
        //         //             Vec3::ZERO,
        //         //             Vec3::Y,
        //         //         )
        //         //         .matrix();
        //         //         estimation.update_uniforms(&gpu.queue, projection
        // * // inner_view_matrix);         let inner_view_dir = //
        //   acq::spherical_to_cartesian(             1.0, // inner_zenith, //
        //   inner_azimuth, // Handedness::RightHandedYUp, //         ); // let
        //   inner_visible_facets = visible_facets //             .iter() //
        //   .filter_map(|idx| { //                 if
        // surface.facet_normals[*idx].dot(inner_view_dir) >= 0.0 {
        //         //                     Some(*idx)
        //         //                 } else {
        //         //                     None
        //         //                 }
        //         //             })
        //         //             .collect::<Vec<_>>();
        //         //         let inner_visible_facets_indices =
        // inner_visible_facets         //             .iter()
        //         //             .flat_map(|idx| {
        //         //                 [
        //         //                     surface.facets[*idx * 3],
        //         //                     surface.facets[*idx * 3 + 1],
        //         //                     surface.facets[*idx * 3 + 2],
        //         //                 ]
        //         //             })
        //         //             .collect::<Vec<_>>();
        //         //         let inner_visible_facets_index_buffer =
        //         //             gpu.device
        //         //
        // .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //         //                     label:
        // Some("shadowing-masking-index-buffer"),         //
        // contents:         //
        // bytemuck::cast_slice(&inner_visible_facets_indices),
        //         //                     usage: wgpu::BufferUsages::INDEX,
        //         //                 });
        //         //         estimation.run_once(
        //         //             &gpu.device,
        //         //             &gpu.queue,
        //         //             &vertex_buffer,
        //         //             &inner_visible_facets_index_buffer,
        //         //             inner_visible_facets_indices.len() as u32,
        //         //             wgpu::IndexFormat::Uint32,
        //         //         );
        //         //         visibilities[inner_azimuth_idx *
        // (desc.zenith.step_count() + 1)         //             +
        // inner_zenith_idx] =         //
        // estimation.calculate_areas(&gpu.device).visibility();
        //         //     }
        //         // }
        //         results.append(&mut visibilities)
        //     }
        // }
        // final_results.push(MicrofacetShadowingMaskingFunction {
        //     azimuth_bin_size: desc.azimuth.step_size,
        //     zenith_bin_size: desc.zenith.step_size,
        //     azimuth_bins_count: desc.azimuth.step_count(),
        //     zenith_bins_count: desc.zenith.step_count() + 1,
        //     samples: results,
        // });
    });
    final_results
}

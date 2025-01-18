use crate::{
    app::cache::RawCache,
    measure::{params::GafMeasurementParams, AnyMeasured, Measurement, MeasurementSource},
};
use base::{
    error::VgonioError,
    impl_any_measured_trait, math,
    math::{Mat4, Vec3},
    units::Radians,
    utils::{handle::Handle, range::StepRangeIncl},
    MeasurementKind,
};
use bytemuck::{Pod, Zeroable};
use gxtk::{
    camera::{Camera, Projection},
    context::{GpuContext, WgpuConfig},
    render_pass::{tex_fmt_bpp, RenderPass},
    texture::Texture,
};
use std::path::Path;
use surf::MicroSurface;
use wgpu::{util::DeviceExt, ColorTargetState};

/// Render pass computing the shadowing/masking (caused by microfacets
/// occlusion) function of a micro-surface.
///
/// For a certain viewing direction, this is done in two steps:
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
/// Render all visible facets. Each fragment outputs value of 1/ 256 or 1/1024
/// depending on the format of the color attachment used. At the blending
/// stage, values will be summed up; later stores inside a texture.
pub struct VisibilityEstimator {
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

    /// Colour attachments used to compute the ratio of visible projected area
    /// over the whole area of all visible facets at each measurement point.
    /// Each colour attachment is a 2D texture array with one layer per
    /// measurement point. The first colour attachment is used to store the
    /// visible projected area (area of visible facets respecting each
    /// other), the second colour attachment is used to store the whole area of
    /// all visible facets.
    color_attachments: [ColorAttachment; 2],

    /// Depth buffers of all micro-facets at all possible measurement points.
    depth_attachment: DepthAttachment,
}

/// Uniforms used by the `VisibilityEstimator`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    proj_view_matrix: [f32; 16],
    meas_point_index: [u32; 2],
    /// Number of measurements stored inside one layered texture.
    /// Same as number of [`DepthAttachment::layers_per_texture`].
    /// Each layer is a depth buffer of the micro-facets at one measurement
    /// point.
    meas_point_per_depth_map: [u32; 2],
}

impl Uniforms {
    const SIZE_IN_BYTES: wgpu::BufferAddress = std::mem::size_of::<Self>() as u64;
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            proj_view_matrix: Mat4::IDENTITY.to_cols_array(),
            meas_point_index: [0, 0],
            meas_point_per_depth_map: [0, 0],
        }
    }
}

/// Visibility estimation result.
#[derive(Debug, Clone, Copy, Default)]
pub struct VisibilityEstimation {
    /// Micro-facets area without occlusion. This is the total area of all
    /// visible micro-facets; this is the denominator of the visibility
    /// function. This is the sum of all values in the color attachment (R
    /// channel).
    total_area: u32,
    /// Micro-facets area with occlusion. This is the total area of all visible
    /// microfacets minus the occluded area; calculated by summing up the
    /// number of fragments that are covered by other micro-facets.
    visible_area: u32,
}

impl VisibilityEstimation {
    /// Returns the visibility value.
    pub fn visibility(&self) -> f32 {
        if self.total_area == 0 {
            0.0
        } else {
            self.visible_area as f32 / self.total_area as f32
        }
    }
}

/// Structure that holds the necessary gpu resources for the shadowing/masking
/// function estimation.
impl VisibilityEstimator {
    /// Texture format used for the color attachment.
    pub const COLOR_ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;
    // wgpu::TextureFormat::Rgba8Unorm

    /// Bytes per pixel of the color attachment.
    pub const COLOR_ATTACHMENT_FORMAT_BPP: u32 = tex_fmt_bpp(Self::COLOR_ATTACHMENT_FORMAT);

    /// Texture format used for the depth attachment.
    pub const DEPTH_ATTACHMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    /// Bytes per pixel of the depth attachment.
    pub const DEPTH_ATTACHMENT_FORMAT_BPP: u32 = tex_fmt_bpp(Self::DEPTH_ATTACHMENT_FORMAT);
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
            size: Uniforms::SIZE_IN_BYTES * meas_count as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("./gaf.wgsl"));
        let color_attachments = [
            ColorAttachment::new(
                &ctx.device,
                width,
                height,
                meas_count,
                Self::COLOR_ATTACHMENT_FORMAT,
            ),
            ColorAttachment::new(
                &ctx.device,
                width,
                height,
                meas_count,
                Self::COLOR_ATTACHMENT_FORMAT,
            ),
        ];
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
                        module: &shader_module,
                        entry_point: "vs_depth_pass",
                        compilation_options: Default::default(),
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
                    cache: None,
                });

            RenderPass {
                pipeline,
                bind_groups: vec![bind_group],
                uniform_buffers: None,
            }
        };

        let render_pass = {
            let uniform_bind_group_layout =
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("oc_render_pass_uniform_bind_group_layout"),
                        entries: &[wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(uniform_buffer_size),
                            },
                            count: None,
                        }],
                    });
            let depth_map_bind_group_layout =
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("oc_render_pass_depth_map_bind_group_layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Depth,
                                    view_dimension: wgpu::TextureViewDimension::D2Array,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::Comparison,
                                ),
                                count: None,
                            },
                        ],
                    });
            let mut bind_groups = Vec::with_capacity(1 + depth_attachment.textures.len());
            log::debug!(
                "Creating {} bind groups for render pass",
                bind_groups.capacity()
            );
            bind_groups
                .push_within_capacity(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("oc_render_pass_uniform_bind_group"),
                    layout: &uniform_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    }],
                }))
                .unwrap();
            depth_attachment
                .textures
                .iter()
                .enumerate()
                .for_each(|(i, texture)| {
                    bind_groups
                        .push_within_capacity(ctx.device.create_bind_group(
                            &wgpu::BindGroupDescriptor {
                                label: Some(&format!("oc_render_pass_depth_map_bind_group_{i}")),
                                layout: &depth_map_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            texture.whole_view(),
                                        ),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: wgpu::BindingResource::Sampler(
                                            &depth_attachment.sampler,
                                        ),
                                    },
                                ],
                            },
                        ))
                        .unwrap();
                });
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("oc_render_pass_pipeline_layout"),
                        bind_group_layouts: &[
                            &uniform_bind_group_layout,
                            &depth_map_bind_group_layout,
                        ],
                        push_constant_ranges: &[],
                    });
            let pipeline = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("oc_render_pass_pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader_module,
                        entry_point: "vs_render_pass",
                        compilation_options: Default::default(),
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
                        module: &shader_module,
                        entry_point: "fs_render_pass",
                        compilation_options: Default::default(),
                        targets: &[
                            Some(ColorTargetState {
                                format: Self::COLOR_ATTACHMENT_FORMAT,
                                blend: Some(wgpu::BlendState {
                                    color: wgpu::BlendComponent {
                                        src_factor: wgpu::BlendFactor::One,
                                        dst_factor: wgpu::BlendFactor::One,
                                        operation: wgpu::BlendOperation::Add,
                                    },
                                    alpha: wgpu::BlendComponent::REPLACE,
                                }),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(ColorTargetState {
                                format: Self::COLOR_ATTACHMENT_FORMAT,
                                blend: Some(wgpu::BlendState {
                                    color: wgpu::BlendComponent {
                                        src_factor: wgpu::BlendFactor::One,
                                        dst_factor: wgpu::BlendFactor::One,
                                        operation: wgpu::BlendOperation::Add,
                                    },
                                    alpha: wgpu::BlendComponent::REPLACE,
                                }),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                        ],
                    }),
                    multiview: None,
                    cache: None,
                });
            RenderPass {
                pipeline,
                bind_groups,
                uniform_buffers: None,
            }
        };

        Self {
            depth_pass,
            render_pass,
            uniform_buffer,
            color_attachments,
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
             points used to create the `VisibilityEstimator`."
        );
        let data = meas_points
            .iter()
            .map(|mp| mp.shader_uniforms(self.depth_attachment.layers_per_texture))
            .collect::<Vec<Uniforms>>();
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
            label: Some("ve_bake_depth_maps_encoder"),
        });

        log::debug!(
            "Baking depth maps... {} measurement points",
            self.num_measurement_points
        );

        // encoder.push_debug_group("oc_depth_maps_bake");

        for i in 0..self.num_measurement_points {
            //encoder.push_debug_group(&format!("oc_depth_maps_bake_{i}"));
            encoder.copy_buffer_to_buffer(
                &self.measurement_points_buffer,
                i as u64 * Uniforms::SIZE_IN_BYTES,
                &self.uniform_buffer,
                0,
                Uniforms::SIZE_IN_BYTES,
            );
            encoder.insert_debug_marker("bake all facets");
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ve_bake_depth_maps_pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.depth_attachment.layer_view(i),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.depth_pass.pipeline);
                pass.set_bind_group(0, &self.depth_pass.bind_groups[0], &[]);
                pass.set_vertex_buffer(0, facets_vtx_buf.slice(..));
                pass.set_index_buffer(facets_idx_buf.slice(..), Self::INDEX_FORMAT);
                pass.draw_indexed(0..facets_idx_num, 0, 0..1);
            }
            //encoder.pop_debug_group();
        }
        //encoder.pop_debug_group();

        self.depth_attachment.copy_to_storage(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    /// Estimate the visibility percentage for visible facets.
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
            label: Some("visibility_estimation_render_encoder"),
        });

        //encoder.push_debug_group("oc_render_pass");
        for i in 0..self.num_measurement_points {
            //encoder.push_debug_group(&format!("oc_render_pass_{i}"));
            encoder.copy_buffer_to_buffer(
                &self.measurement_points_buffer,
                i as u64 * Uniforms::SIZE_IN_BYTES,
                &self.uniform_buffer,
                0,
                Uniforms::SIZE_IN_BYTES,
            );
            encoder.insert_debug_marker("render all visible facets");
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("visibility_pass_compute_render_pass"),
                    color_attachments: &[
                        // Visible projected area
                        Some(wgpu::RenderPassColorAttachment {
                            view: self.color_attachments[0].layer_view(i),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        // Whole projected area
                        Some(wgpu::RenderPassColorAttachment {
                            view: self.color_attachments[1].layer_view(i),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let depth_map_idx = self.depth_attachment.texture_index_of_layer(i);
                log::trace!("layer {} in texture {}", i, depth_map_idx);

                render_pass.set_pipeline(&self.render_pass.pipeline);
                render_pass.set_bind_group(0, &self.render_pass.bind_groups[0], &[]);
                render_pass.set_bind_group(
                    1,
                    &self.render_pass.bind_groups[depth_map_idx as usize + 1],
                    &[],
                );
                render_pass.set_vertex_buffer(0, facets_vtx_buf.slice(..));
                render_pass.set_index_buffer(visible_facets_idx_buf.slice(..), Self::INDEX_FORMAT);
                render_pass.draw_indexed(0..visible_facets_idx_num, 0, 0..1);
            }
            //encoder.pop_debug_group();
        }
        //encoder.pop_debug_group();

        self.color_attachments[0].copy_to_storage(&mut encoder);
        self.color_attachments[1].copy_to_storage(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }

    /// Saves the visibility estimation outputs to the given directory.
    pub fn save_color_attachment(
        &self,
        device: &wgpu::Device,
        dir: &Path,
        names: &[String],
        remap: bool,
        as_image: bool,
    ) -> Result<(), VgonioError> {
        log::info!("Saving color attachment to {:?}", dir);
        self.color_attachments[0].save(
            device,
            &dir.join("visible_projected_area"),
            names,
            remap,
            as_image,
        )?;
        self.color_attachments[1].save(
            device,
            &dir.join("total_projected_area"),
            names,
            remap,
            as_image,
        )
    }

    /// Saves the depth estimation outputs to the given directory.
    pub fn save_depth_attachment(
        &self,
        device: &wgpu::Device,
        dir: &Path,
        names: &[String],
        as_image: bool,
    ) -> Result<(), VgonioError> {
        log::info!("Saving depth attachment to {:?}", dir);
        self.depth_attachment.save(device, dir, names, as_image)
    }

    fn compute_area(&self, device: &wgpu::Device, color_attachment: &ColorAttachment) -> Vec<u32> {
        use rayon::{prelude::ParallelIterator, slice::ParallelSlice};
        let layer_size_in_bytes = color_attachment.layer_size_in_bytes();
        let is_8bit = color_attachment.is_8bit();
        color_attachment
            .storage_buffers
            .iter()
            .flat_map(|storage| {
                let area = {
                    let buf_view = {
                        let buf_slice = storage.slice(..);
                        let (sender, receiver) =
                            futures_intrusive::channel::shared::oneshot_channel();
                        buf_slice.map_async(wgpu::MapMode::Read, move |result| {
                            sender.send(result).unwrap();
                        });
                        device.poll(wgpu::Maintain::Wait);
                        pollster::block_on(async {
                            receiver.receive().await.unwrap().unwrap();
                        });
                        buf_slice.get_mapped_range()
                    };
                    if is_8bit {
                        buf_view
                            .par_chunks_exact(layer_size_in_bytes as usize)
                            .map(|layer| {
                                layer.chunks(4).fold(0, |acc, chunk| {
                                    let value = chunk[0] as u32;
                                    acc + value
                                })
                            })
                            .collect::<Vec<_>>()
                    } else {
                        buf_view
                            .par_chunks_exact(layer_size_in_bytes as usize)
                            .map(|layer| {
                                layer.chunks(4).fold(0, |acc, chunk| {
                                    let rgba = u32::from_le_bytes([
                                        chunk[0], chunk[1], chunk[2], chunk[3],
                                    ]);
                                    let value = rgba & 0x03FF;
                                    acc + value
                                })
                            })
                            .collect::<Vec<_>>()
                    }
                };
                storage.unmap();
                area
            })
            .collect()
    }

    /// Computes the visibility.
    ///
    /// The `bake` and `estimate` methods must be called before this method.
    pub fn compute_visibility(&self, device: &wgpu::Device) -> Vec<VisibilityEstimation> {
        log::info!("Computing visibility...");
        let visible_area = self.compute_area(device, &self.color_attachments[0]);
        let total_area = self.compute_area(device, &self.color_attachments[1]);
        visible_area
            .iter()
            .zip(total_area.iter())
            .enumerate()
            .map(|(i, (&visible_area, &total_area))| {
                #[cfg(debug_assertions)]
                {
                    if total_area == 0 {
                        eprintln!(
                            "WARNING: total area is zero! (visible_area: {}) -- index: {}",
                            visible_area, i
                        );
                    }
                }
                VisibilityEstimation {
                    total_area,
                    visible_area,
                }
            })
            .collect()
    }
}

/// An array of 2D textures.
pub struct LayeredTexture {
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
    /// The number of bytes of the whole texture.
    pub size_in_bytes: u64,
}

impl LayeredTexture {
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
            view_formats: &[],
        });
        let mut views = Vec::with_capacity(layers as usize + 1);
        views.push(texture.create_view(&wgpu::TextureViewDescriptor {
            label,
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_array_layer: 0,
            array_layer_count: Some(layers),
            ..Default::default()
        }));
        for i in 0..layers {
            views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("oc_depth_texture_layer_view"),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i,
                array_layer_count: Some(1),
            }));
        }
        let layer_size_in_bytes = (width * height * format.block_copy_size(None).unwrap()) as u64;
        Self {
            format,
            texture,
            views,
            extent,
            layer_size_in_bytes,
            size_in_bytes: layer_size_in_bytes * layers as u64,
        }
    }

    /// Copies the texture data to the storage.
    pub fn copy_to_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        offset: u64,
    ) {
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
                    offset,
                    bytes_per_row: Some(self.extent.width * tex_fmt_bpp(self.format)),
                    rows_per_image: Some(self.extent.height),
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
///
/// Because of hardware limitations, the depth attachment is an array of layered
/// textures.
struct DepthAttachment {
    /// The texture used for the depth attachment.
    textures: Vec<LayeredTexture>,
    /// Number of layers of all the textures.
    layers: u32,
    /// Maximum number of layers per texture.
    layers_per_texture: u32,
    /// The sampler used to sample the depth map in the shader.
    sampler: wgpu::Sampler,
    /// The storage of the texture data in case the data need to be saved.
    /// Each texture is stored in a separate buffer. It has the same length
    /// as the textures' array.
    storage_buffers: Vec<wgpu::Buffer>,
}

impl DepthAttachment {
    /// Create a new depth attachment.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `width` - The width of the texture.
    /// * `height` - The height of the texture.
    /// * `layers` - The number of layers in total.
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
        let max_layers_per_texture = device.limits().max_texture_array_layers;
        let max_buffer_size = device.limits().max_buffer_size;

        let layers_per_texture = max_layers_per_texture.min(
            (max_buffer_size / ((width * height * format.block_copy_size(None).unwrap()) as u64))
                as u32,
        );

        let textures_count = (layers as f32 / layers_per_texture as f32).ceil() as u32;
        log::debug!(
            "Creating depth attachment with {} layers, {} textures, {} layers per texture",
            layers,
            textures_count,
            layers_per_texture
        );
        let mut textures = Vec::with_capacity(textures_count as usize);
        let mut layers_left = layers;
        while layers_left > 0 {
            let layers_in_this_texture = layers_left.min(layers_per_texture);
            textures.push(LayeredTexture::new(
                device,
                width,
                height,
                layers_in_this_texture,
                format,
                Some(&format!("oc_depth_texture_part_{}", textures.len())),
            ));
            layers_left -= layers_in_this_texture;
        }

        let storage_buffers = textures
            .iter()
            .enumerate()
            .map(|(i, texture)| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("oc_depth_texture_storage_buffer_{i}")),
                    size: texture.size_in_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Self {
            textures,
            sampler,
            storage_buffers,
            layers,
            layers_per_texture,
        }
    }

    /// Number of layers in total.
    pub fn layers(&self) -> u32 { self.layers }

    /// Returns the texture view of the specified layer of the whole attachment.
    pub fn layer_view(&self, layer: u32) -> &wgpu::TextureView {
        assert!(layer < self.layers, "Layer index out of range");
        let texture_index = layer / self.layers_per_texture;
        let layer_in_texture = layer % self.layers_per_texture;
        self.textures[texture_index as usize].layer_view(layer_in_texture)
    }

    /// Returns the texture index of the specified layer.
    pub fn texture_index_of_layer(&self, layer: u32) -> u32 {
        assert!(layer < self.layers, "Layer index out of range");
        layer / self.layers_per_texture
    }

    /// Copies the texture data to the storage.
    pub fn copy_to_storage(&self, encoder: &mut wgpu::CommandEncoder) {
        self.textures
            .iter()
            .zip(self.storage_buffers.iter())
            .for_each(|(texture, storage)| {
                texture.copy_to_buffer(encoder, storage, 0);
            });
    }

    /// Save the attachment data to files.
    pub fn save(
        &self,
        device: &wgpu::Device,
        dir: &Path,
        names: &[String],
        as_image: bool,
    ) -> Result<(), VgonioError> {
        self.textures
            .iter()
            .zip(self.storage_buffers.iter())
            .enumerate()
            .for_each(|(storage_texture_index, (texture, storage))| {
                {
                    let buffer = storage.slice(..);
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
                    let width = texture.extent.width;
                    let height = texture.extent.height;

                    let num_pixels = (width * height) as usize;

                    if as_image {
                        use image::{ImageBuffer, Luma};
                        for (i, layer) in data.chunks_exact(num_pixels).enumerate() {
                            let mut img = ImageBuffer::new(width, height);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let index = (y * width + x) as usize;
                                *pixel = Luma([(layer[index] * 255.0) as u8]);
                            }
                            let path = dir.join(format!(
                                "depth_layer_{}.png",
                                names[storage_texture_index * self.layers_per_texture as usize + i]
                            ));
                            img.save(path).expect("Failed to save image");
                        }
                    } else {
                        use std::io::Write;
                        for (i, layer) in data.chunks_exact(num_pixels).enumerate() {
                            let mut file = std::fs::File::create(dir.join(format!(
                                "depth_layer_{}.txt",
                                names[storage_texture_index * self.layers_per_texture as usize + i]
                            )))
                            .unwrap();
                            for (j, val) in layer.iter().enumerate() {
                                if j % width as usize == 0 {
                                    writeln!(file, "{val}").unwrap();
                                } else {
                                    write!(file, "{val} ").unwrap();
                                }
                            }
                        }
                    }
                }
                storage.unmap();
            });
        Ok(())
    }
}

/// Color attachment used for occlusion estimation, which is a texture with the
/// same size as the depth attachment. The number of non-zero pixels is the area
/// of micro-facets together with the occlusion. The sum of all pixel values
/// is the area of micro-facets in total without the occlusion.
///
/// Because of underlying hardware's limitation of maximum image array layers of
/// a single texture, the color attachment may contain multiple layered
/// textures. Each layer of the texture is a color map of the micro-surface at a
/// specific angle.
struct ColorAttachment {
    /// The texture used for the color attachment.
    textures: Vec<LayeredTexture>,
    /// Number of layers of all the textures.
    layers: u32,
    /// Maximum number of layers per texture.
    layers_per_texture: u32,
    /// The storage of the texture data in case the data need to be saved.
    /// Each texture has a corresponding storage buffer.
    storage_buffers: Vec<wgpu::Buffer>,
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
        let max_layers_per_texture = device.limits().max_texture_array_layers;

        let max_buffer_size = device.limits().max_buffer_size;

        let layers_per_texture = max_layers_per_texture.min(
            (max_buffer_size / ((width * height * format.block_copy_size(None).unwrap()) as u64))
                as u32,
        );

        let textures_count = (layers as f32 / layers_per_texture as f32).ceil() as u32;
        log::debug!(
            "Creating color attachment with {} layers, {} textures, {} layers per texture",
            layers,
            textures_count,
            layers_per_texture
        );
        let mut textures = Vec::with_capacity(textures_count as usize);
        let mut layers_left = layers;
        while layers_left > 0 {
            let layers_in_this_texture = layers_left.min(layers_per_texture);
            textures.push(LayeredTexture::new(
                device,
                width,
                height,
                layers_in_this_texture,
                format,
                Some(&format!("oc_color_texture_part_{}", textures.len())),
            ));
            layers_left -= layers_in_this_texture;
        }
        let storage_buffers = textures
            .iter()
            .enumerate()
            .map(|(i, texture)| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("oc_color_texture_storage_part_{i}")),
                    size: texture.layer_size_in_bytes * texture.layers() as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            textures,
            storage_buffers,
            layers,
            layers_per_texture,
        }
    }

    /// Returns the texture view of the specified layer of the whole attachment.
    pub fn layer_view(&self, layer: u32) -> &wgpu::TextureView {
        assert!(layer < self.layers, "Layer index out of range");
        let texture_index = layer / self.layers_per_texture;
        let layer_in_texture = layer % self.layers_per_texture;
        log::debug!(
            "Getting color attach. view of {} which is located in texture {}, layer {}",
            layer,
            texture_index,
            layer_in_texture,
        );
        self.textures[texture_index as usize].layer_view(layer_in_texture)
    }

    /// Returns the size of one layer in bytes.
    pub fn layer_size_in_bytes(&self) -> u64 { self.textures[0].layer_size_in_bytes }

    /// Returns the texture index of the specified layer.
    pub fn texture_index(&self, layer: u32) -> u32 {
        assert!(layer < self.layers, "Layer index out of range");
        layer / self.layers_per_texture
    }

    /// Copies the texture data to the storage.
    ///
    /// # Note
    ///
    /// Only records the command to the encoder. The command won't be executed
    /// until the encoder is submitted.
    pub fn copy_to_storage(&self, encoder: &mut wgpu::CommandEncoder) {
        for (i, texture) in self.textures.iter().enumerate() {
            texture.copy_to_buffer(encoder, &self.storage_buffers[i], 0);
        }
    }

    /// Save the color attachment data to files.
    pub fn save(
        &self,
        device: &wgpu::Device,
        dir: &Path,
        names: &[String],
        remap: bool,
        as_image: bool,
    ) -> Result<(), VgonioError> {
        self.textures
            .iter()
            .zip(self.storage_buffers.iter())
            .enumerate()
            .for_each(|(storage_texture_index, (texture, storage))| {
                {
                    let is_8bit = texture.format == wgpu::TextureFormat::Rgba8Unorm;

                    let buffer = storage.slice(..);
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

                    let width = texture.extent.width;
                    let height = texture.extent.height;

                    if as_image {
                        use image::{ImageBuffer, Rgba};
                        for (i, layer) in buffer_view
                            .chunks_exact(texture.layer_size_in_bytes as usize)
                            .enumerate()
                        {
                            if !is_8bit {
                                panic!("Saving 10-bit color textures as images is not supported.");
                            } else {
                                let mut px_data = vec![0u8; layer.len()];
                                px_data.copy_from_slice(layer);
                                let mut img =
                                    ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, px_data)
                                        .ok_or(VgonioError::new(
                                            "Failed to create image buffer",
                                            None,
                                        ))
                                        .unwrap();
                                // Remap the values from min-max to 0-255.
                                if remap {
                                    let min = *layer.iter().step_by(4).min().unwrap() as f32;
                                    let max = *layer.iter().step_by(4).max().unwrap() as f32;
                                    img.pixels_mut().for_each(|px| {
                                        let remapped = ((px[0] as f32 - min) * 255.0 / max) as u8;
                                        px[0] = remapped;
                                        px[1] = remapped;
                                        px[2] = remapped;
                                        px[3] = 255;
                                    });
                                }
                                img.save(dir.join(format!(
                                    "color_layer_{}.png",
                                    names[storage_texture_index * self.layers_per_texture as usize
                                        + i],
                                )))
                                .unwrap();
                            }
                        }
                    } else {
                        // Only write the R channel.
                        use std::io::Write;
                        for (i, layer) in buffer_view
                            .chunks_exact(texture.layer_size_in_bytes as usize)
                            .enumerate()
                        {
                            let mut file = std::fs::File::create(dir.join(format!(
                                "color_layer_{:04}.txt",
                                storage_texture_index * self.layers_per_texture as usize + i
                            )))
                            .unwrap();
                            if is_8bit {
                                layer.chunks(4).map(|chunk| chunk[0]).enumerate().for_each(
                                    |(j, val)| {
                                        if j % width as usize == 0 {
                                            writeln!(file, "{val}")
                                        } else {
                                            write!(file, "{val} ")
                                        }
                                        .expect("Failed to write to file");
                                    },
                                );
                            } else {
                                layer
                                    .chunks(4)
                                    .map(|chunk| {
                                        let rgba = u32::from_le_bytes([
                                            chunk[0], chunk[1], chunk[2], chunk[3],
                                        ]);
                                        rgba & 0x3ff
                                    })
                                    .enumerate()
                                    .for_each(|(j, val)| {
                                        if j % width as usize == 0 {
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
                storage.unmap();
            });
        Ok(())
    }

    /// Returns true if the color attachment is 8-bit.
    /// Otherwise, it is 10-bit. This is used to determine how to save the data.
    /// Also, 8-bit color attachment is used for debugging.
    pub fn is_8bit(&self) -> bool { self.textures[0].format == wgpu::TextureFormat::Rgba8Unorm }
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

impl MeasurementPoint {
    /// Returns the uniforms for the measurement point used in the shader.
    fn shader_uniforms(&self, layers_per_texture: u32) -> Uniforms {
        Uniforms {
            proj_view_matrix: self.proj_view_mat.to_cols_array(),
            meas_point_index: [self.index, 0],
            meas_point_per_depth_map: [layers_per_texture, 0],
        }
    }
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
/// G(i, o, m) is the micro facet masking-shadowing function, which describes
/// the fraction of micro-facets with normal m that are visible from both the
/// incident direction i and the outgoing direction o.
///
/// The Smith microfacet masking-shadowing function is defined as:
///
/// G(i, o, m) = G1(i, m) * G1(o, m)
///
/// This structure holds the data for G1(i, m).
#[derive(Debug, Clone)]
pub struct MeasuredGafData {
    /// The measurement parameters.
    pub params: GafMeasurementParams,
    /// The distribution data. The outermost dimension is the view direction
    /// (microfacet normal) generated by the azimuthal and zenith angle
    /// (azimuth first then zenith). The innermost dimension is the
    /// visibility data for each incident direction.
    pub samples: Box<[f32]>,
}

impl_any_measured_trait!(MeasuredGafData, Gaf);

impl MeasuredGafData {
    // TODO: review the necessity of this method.
    /// Returns the measurement range of the azimuthal and zenith angles.
    /// The azimuthal angle is in the range 0 ~ 2π, and the zenith angle is in
    /// the range 0 ~ π/2.
    pub fn measurement_range(&self) -> (StepRangeIncl<Radians>, StepRangeIncl<Radians>) {
        (self.params.azimuth, self.params.zenith)
    }

    /// Returns the Masking Shadowing Function data slice for the given
    /// microfacet normal and azimuthal angle of the incident direction.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if it exists.
    pub fn slice_at(
        &self,
        azim_m: Radians,
        zeni_m: Radians,
        azim_i: Radians,
    ) -> (&[f32], Option<&[f32]>) {
        let azimuth_m = azim_m.wrap_to_tau();
        let azimuth_i = azim_i.wrap_to_tau();
        let zenith_m = zeni_m.clamp(self.params.zenith.start, self.params.zenith.stop);
        let azimuth_m_idx = self.params.azimuth.index_of(azimuth_m);
        let zenith_m_idx = self.params.zenith.index_of(zenith_m);
        let azimuth_i_idx = self.params.azimuth.index_of(azimuth_i);
        let opposite_azimuth_i = azimuth_i.opposite();
        let opposite_azimuth_i_idx = if self.params.azimuth.start <= opposite_azimuth_i
            && opposite_azimuth_i <= self.params.azimuth.stop
        {
            Some(self.params.azimuth.index_of(opposite_azimuth_i))
        } else {
            None
        };
        (
            self.slice_at_indices(azimuth_m_idx, zenith_m_idx, azimuth_i_idx),
            opposite_azimuth_i_idx
                .map(|index| self.slice_at_indices(azimuth_m_idx, zenith_m_idx, index)),
        )
    }

    /// Returns a data slice of the Masking Shadowing Function for the given
    /// indices.
    fn slice_at_indices(
        &self,
        azimuth_m_idx: usize,
        zenith_m_idx: usize,
        azimuth_i_idx: usize,
    ) -> &[f32] {
        debug_assert!(self.kind() == MeasurementKind::Gaf);
        debug_assert!(
            azimuth_m_idx < self.params.azimuth.step_count_wrapped(),
            "index out of range"
        );
        debug_assert!(
            azimuth_i_idx < self.params.azimuth.step_count_wrapped(),
            "index out of range"
        );
        debug_assert!(
            zenith_m_idx < self.params.zenith.step_count_wrapped(),
            "index out of range"
        );
        let zenith_bin_count = self.params.zenith.step_count_wrapped();
        let azimuth_bin_count = self.params.azimuth.step_count_wrapped();
        let offset = azimuth_m_idx * zenith_bin_count * azimuth_bin_count * zenith_bin_count
            + zenith_m_idx * azimuth_bin_count * zenith_bin_count
            + azimuth_i_idx * zenith_bin_count;
        &self.samples[offset..offset + zenith_bin_count]
    }
}

// pub type MeasuredMsf = MeasuredData2<MsfMeasurementParams, 4>;

/// Measurement of microfacet shadowing and masking function.
pub fn measure_masking_shadowing_function(
    params: GafMeasurementParams,
    handles: &[Handle<MicroSurface>],
    cache: &RawCache,
) -> Box<[Measurement]> {
    log::info!("Measuring microfacet masking/shadowing function...");
    let wgpu_config = WgpuConfig {
        device_descriptor: wgpu::DeviceDescriptor {
            label: Some("occlusion-measurement-device"),
            required_features: wgpu::Features::POLYGON_MODE_LINE,
            required_limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
            memory_hints: Default::default(),
        },
        target_format: Some(VisibilityEstimator::COLOR_ATTACHMENT_FORMAT),
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    };
    let gpu = pollster::block_on(GpuContext::offscreen(&wgpu_config));
    let mut estimator = VisibilityEstimator::new(
        &gpu,
        params.resolution,
        params.resolution,
        params.measurement_points_count() as u32,
    );
    let surfaces = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let mut results = Vec::with_capacity(meshes.len());

    for ((hdl, surface), mesh) in handles.iter().zip(surfaces.iter()).zip(meshes.iter()) {
        if mesh.is_none() {
            continue;
        }

        let mesh = mesh.unwrap();
        let facets_vtx_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface_vertex_buffer"),
                contents: bytemuck::cast_slice(&mesh.verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        log::debug!(
            "Creating index buffer of {} indices, expected size {} | maximum buffer size {}",
            mesh.facets.len(),
            mesh.facets.len() * 4,
            gpu.device.limits().max_buffer_size
        );
        let facets_idx_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface_index_buffer"),
                contents: bytemuck::cast_slice(&mesh.facets),
                usage: wgpu::BufferUsages::INDEX,
            });
        let facets_idx_num = mesh.facets.len() as u32;
        let diagonal = mesh.bounds.max_extent() * 1.5;
        let half_zenith_bin_size_cos = (params.zenith.step_size / 2.0).cos();
        let proj_mat =
            Projection::orthographic_matrix(diagonal * 0.5, diagonal * 1.5, diagonal, diagonal);
        #[rustfmt::skip]
        // Generate measurement points.
        // NOTE(yang): Measurement points represent the view direction (microfacet normal), not
        //             the incident direction. The measurement points are generated by the
        //             azimuthal and zenith angle (azimuth first then zenith).
        let meas_points = (0..params.measurement_points_count())
            .map(|i| {
                let azimuth =
                    params.azimuth.step_size * (i / params.zenith.step_count_wrapped()) as f32;
                let zenith =
                    params.zenith.step_size * (i % params.zenith.step_count_wrapped()) as f32;
                let view_dir =
                    math::sph_to_cart(zenith, azimuth);
                let pos = view_dir * diagonal;
                let proj_view_mat = {
                    let view_mat = Camera::new(pos, Vec3::ZERO).view_matrix();
                    proj_mat * view_mat
                };
                MeasurementPoint::new(azimuth, zenith, view_dir, proj_view_mat, i as u32)
            })
            .collect::<Vec<_>>();
        let names = meas_points
            .iter()
            .map(|p| format!("ph{}_th{}", p.azimuth.prettified(), p.zenith.prettified()))
            .collect::<Vec<_>>();

        // Upload measurement points to GPU.
        estimator.update_measurement_points(&gpu.queue, &meas_points);
        // Render the depth buffer.
        estimator.bake(
            &gpu.device,
            &gpu.queue,
            &facets_vtx_buf,
            &facets_idx_buf,
            facets_idx_num,
        );

        #[cfg(debug_assertions)]
        {
            estimator
                .save_depth_attachment(&gpu.device, Path::new("debug_depth_maps"), &names, true)
                .expect("Failed to save depth attachment");
        }

        // Measure the visibility.
        // for each m_phi
        //    for each m_theta
        //        for each i_phi
        //            for each i_theta
        let measurement = meas_points
            .iter()
            .flat_map(|mp| {
                let visible_facets_indices = if params.strict {
                    mesh.facet_normals
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, normal)| {
                            (normal.dot(mp.view_dir) >= half_zenith_bin_size_cos).then_some(idx)
                        })
                        .flat_map(|idx| {
                            [
                                mesh.facets[idx * 3],
                                mesh.facets[idx * 3 + 1],
                                mesh.facets[idx * 3 + 2],
                            ]
                        })
                        .collect::<Vec<_>>()
                } else {
                    mesh.facet_normals
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, normal)| (normal.dot(mp.view_dir) >= 0.0).then_some(idx))
                        .flat_map(|idx| {
                            [
                                mesh.facets[idx * 3],
                                mesh.facets[idx * 3 + 1],
                                mesh.facets[idx * 3 + 2],
                            ]
                        })
                        .collect::<Vec<_>>()
                };

                let visible_facets_index_buffer =
                    gpu.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("masking-shadowing-index-buffer"),
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

                #[cfg(debug_assertions)]
                {
                    let dir = format!(
                        "debug_facets_{}_{}_{}",
                        mp.index,
                        mp.azimuth.prettified(),
                        mp.zenith.prettified()
                    );
                    estimator
                        .save_color_attachment(&gpu.device, Path::new(&dir), &names, false, true)
                        .expect("Failed to save color attachment");
                }

                estimator
                    .compute_visibility(&gpu.device)
                    .into_iter()
                    .map(|est| est.visibility())
            })
            .collect::<Box<_>>();

        results.push(Measurement {
            name: surface.unwrap().file_stem().unwrap().to_owned(),
            source: MeasurementSource::Measured(*hdl),
            timestamp: chrono::Local::now(),
            measured: Box::new(MeasuredGafData {
                params,
                samples: measurement,
            }),
        });
    }
    results.into_boxed_slice()
}

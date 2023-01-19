use crate::{
    acq,
    acq::{measurement::MicrofacetShadowingMaskingMeasurement, Handedness},
    app::{
        cache::{Cache, Handle},
        gfx::{
            bytes_per_pixel,
            camera::{Camera, Projection},
            GpuContext, RdrPass, Texture, WgpuConfig,
        },
    },
    msurf::MicroSurface,
    units::Radians,
    Error,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rayon::{iter::IntoParallelRefIterator, prelude::IndexedParallelIterator};
use std::{
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroU32,
    path::Path,
};
use wgpu::{util::DeviceExt, ColorTargetState};

/// Render pass computing the shadowing/masking (caused by occlusion of
/// micro-facets) function of a micro-surface. For a certain viewing direction,
/// this is done in two steps:
///
/// Render all visible facets. Each fragment outputs 1, then at the blending
/// stage, sum up; then stores inside of 8-bit depth image.
pub struct OcclusionEstimationPass {
    /// Pipeline and corresponding shader inputs.
    inner: RdrPass,

    /// Color attachment of the render pass. Used to compute the projected area
    /// of all visible facets.
    target: Texture,

    /// Color attachment used to compute the projected whole area of all visible
    /// facets.
    target_storage: wgpu::Buffer,

    /// Width of color and depth attachment.
    target_width: u32,

    /// Height of color and depth attachment.
    target_height: u32,
}

/// Uniforms used by the `OcclusionPass`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    projection_matrix: [f32; 16],
}

/// Occlusion estimation result.
#[derive(Debug, Clone, Copy)]
pub struct OcclusionEstimation {
    area_without_occlusion: u32,
    area_with_occlusion: u32,
}

impl OcclusionEstimation {
    pub fn visibility(&self) -> f32 {
        if self.area_without_occlusion == 0 {
            log::warn!("Area without occlusion is zero.");
            f32::NAN
        } else {
            self.area_with_occlusion as f32 / self.area_without_occlusion as f32
        }
    }
}

impl OcclusionEstimationPass {
    pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;

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
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("./occlusion.wgsl"));
        let target = Texture::new(
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
                format: Self::TARGET_FORMAT,
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
                    module: &&shader_module,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &&shader_module,
                    entry_point: "fs_main",
                    targets: &[Some(ColorTargetState {
                        format: Self::TARGET_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::RED,
                    })],
                }),
                multiview: None,
            });
        let pass = RdrPass {
            pipeline,
            bind_groups: vec![bind_group],
            uniform_buffer: Some(uniform_buffer),
        };

        let target_storage_size =
            bytes_per_pixel(Self::TARGET_FORMAT) as u64 * (width * height) as u64;

        let target_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion_pass_color_attachment_storage"),
            size: target_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            inner: pass,
            target,
            target_storage,
            target_width: width,
            target_height: height,
        }
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline { &self.inner.pipeline }

    pub fn bind_groups(&self) -> &[wgpu::BindGroup] { &self.inner.bind_groups }

    pub fn uniform_buffer(&self) -> Option<&wgpu::Buffer> { self.inner.uniform_buffer.as_ref() }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, proj_mat: Mat4) {
        queue.write_buffer(
            self.inner.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[proj_mat]),
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
                    view: &self.target.view,
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

            render_pass.set_pipeline(&self.inner.pipeline);
            render_pass.set_bind_group(0, &self.inner.bind_groups[0], &[]);
            render_pass.set_vertex_buffer(0, v_buf.slice(..));
            render_pass.set_index_buffer(i_buf.slice(..), i_format);
            render_pass.draw_indexed(0..i_count, 0, 0..1);
        }

        // Copy color attachment values to its storage.
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.target.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.target_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(
                        bytes_per_pixel(Self::TARGET_FORMAT) as u32 * self.target_width,
                    ),
                    rows_per_image: NonZeroU32::new(self.target_height),
                },
            },
            wgpu::Extent3d {
                width: self.target_width,
                height: self.target_height,
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
            let slice = self.target_storage.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });

            let buffer_view = slice.get_mapped_range();
            let values = buffer_view
                .chunks(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();

            if as_image {
                use image::{ImageBuffer, Luma};
                ImageBuffer::<Luma<u8>, _>::from_raw(
                    self.target_width,
                    self.target_height,
                    values
                        .iter()
                        .map(|&v| if v == 0 { 0 } else { 255 })
                        .collect::<Vec<_>>(),
                )
                .ok_or_else(|| {
                    Error::Any(
                        "Failed to create image from depth map buffer, please
            check if the data have been transferred to
            the buffer!"
                            .into(),
                    )
                })
                .and_then(|img| img.save(format!("{filename}.png")).map_err(Error::from))?;
            } else {
                let mut file = File::create(format!("{filename}.txt"))?;
                for (i, val) in values.iter().enumerate() {
                    if i as u32 % self.target_width == 0 {
                        writeln!(file)?;
                    } else {
                        write!(file, "{} ", val)?;
                    }
                }
            }
        }
        self.target_storage.unmap();

        Ok(())
    }

    fn calculate_areas(&self, device: &wgpu::Device) -> OcclusionEstimation {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let (non_occluded_area, occluded_area) = {
            let slice = self.target_storage.slice(..);
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
        self.target_storage.unmap();
        OcclusionEstimation {
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
            features: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
        },
        target_format: Some(OcclusionEstimationPass::TARGET_FORMAT),
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    };
    let gpu = pollster::block_on(GpuContext::offscreen(&wgpu_config));
    let estimation = OcclusionEstimationPass::new(&gpu, desc.resolution, desc.resolution);
    let surfaces = cache.get_micro_surface_meshes(surfaces);
    let mut final_results = Vec::with_capacity(surfaces.len());
    surfaces.iter().for_each(|surface| {
        let mut results =
            Vec::with_capacity((desc.azimuth.step_count() * (desc.zenith.step_count() + 1)).pow(2));
        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&surface.verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let radius =
            (surface.extent.max - surface.extent.min).max_element() * std::f32::consts::SQRT_2;
        let projection = Projection::orthographic_matrix(0.1, radius * 2.0, radius, radius);
        for azimuth_idx in 0..desc.azimuth.step_count() {
            for zenith_idx in 0..desc.zenith.step_count() + 1 {
                let azimuth = azimuth_idx as f32 * desc.azimuth.step_size;
                let zenith = zenith_idx as f32 * desc.zenith.step_size;
                let index = azimuth_idx * (desc.zenith.step_count() + 1) + zenith_idx;
                let view_matrix = Camera::new(
                    acq::spherical_to_cartesian(radius, zenith, azimuth, handedness),
                    Vec3::ZERO,
                    Vec3::Y,
                )
                .matrix();
                estimation.update_uniforms(&gpu.queue, projection * view_matrix);
                let view_dir =
                    acq::spherical_to_cartesian(1.0, zenith, azimuth, Handedness::RightHandedYUp)
                        .normalize();
                let visible_facets = surface
                    .facet_normals
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, normal)| {
                        // if normal.dot(view_dir) >= half_zenith_bin_size_cos {
                        if normal.dot(view_dir) >= 0.0 {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                // let visible_facets_indices = visible_facets
                //     .iter()
                //     .flat_map(|idx| {
                //         [
                //             surface.facets[idx * 3],
                //             surface.facets[idx * 3 + 1],
                //             surface.facets[idx * 3 + 2],
                //         ]
                //     })
                //     .collect::<Vec<_>>();
                //
                // let visible_facets_index_buffer =
                //     gpu.device
                //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                //             label: Some("shadowing-masking-index-buffer"),
                //             contents: bytemuck::cast_slice(&visible_facets),
                //             usage: wgpu::BufferUsages::INDEX,
                //         });
                // estimation.run_once(
                //     &gpu.device,
                //     &gpu.queue,
                //     &vertex_buffer,
                //     &visible_facets_index_buffer,
                //     visible_facets.len() as u32,
                //     wgpu::IndexFormat::Uint32,
                // );
                // // let filename = format!("{index:06}_roi");
                // // estimation
                // //     .save_color_attachment(&gpu.device, &filename, true)
                // //     .unwrap();
                // let result = estimation.calculate_areas(&gpu.device);
                // log::trace!(
                //     "azimuth: {}, zenith: {}, visible facets count: {} ==> {}",
                //     azimuth.prettified(),
                //     zenith.prettified(),
                //     visible_facets.len(),
                //     result.visibility()
                // );

                let mut visibilities =
                    vec![0.0; desc.azimuth.step_count() * (desc.zenith.step_count() + 1)];
                for inner_azimuth_idx in 0..desc.azimuth.step_count() {
                    for inner_zenith_idx in 0..desc.zenith.step_count() + 1 {
                        let inner_azimuth = inner_azimuth_idx as f32 * desc.azimuth.step_size;
                        let inner_zenith = inner_zenith_idx as f32 * desc.zenith.step_size;
                        let inner_view_matrix = Camera::new(
                            acq::spherical_to_cartesian(
                                radius,
                                inner_zenith,
                                inner_azimuth,
                                handedness,
                            ),
                            Vec3::ZERO,
                            Vec3::Y,
                        )
                        .matrix();
                        estimation.update_uniforms(&gpu.queue, projection * inner_view_matrix);
                        let inner_view_dir = acq::spherical_to_cartesian(
                            1.0,
                            inner_zenith,
                            inner_azimuth,
                            Handedness::RightHandedYUp,
                        );
                        let inner_visible_facets = visible_facets
                            .iter()
                            .filter_map(|idx| {
                                if surface.facet_normals[*idx].dot(inner_view_dir) >= 0.0 {
                                    Some(*idx)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();
                        let inner_visible_facets_indices = inner_visible_facets
                            .iter()
                            .flat_map(|idx| {
                                [
                                    surface.facets[*idx * 3],
                                    surface.facets[*idx * 3 + 1],
                                    surface.facets[*idx * 3 + 2],
                                ]
                            })
                            .collect::<Vec<_>>();
                        let inner_visible_facets_index_buffer =
                            gpu.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("shadowing-masking-index-buffer"),
                                    contents: bytemuck::cast_slice(&inner_visible_facets_indices),
                                    usage: wgpu::BufferUsages::INDEX,
                                });
                        estimation.run_once(
                            &gpu.device,
                            &gpu.queue,
                            &vertex_buffer,
                            &inner_visible_facets_index_buffer,
                            inner_visible_facets_indices.len() as u32,
                            wgpu::IndexFormat::Uint32,
                        );
                        visibilities[inner_azimuth_idx * (desc.zenith.step_count() + 1)
                            + inner_zenith_idx] =
                            estimation.calculate_areas(&gpu.device).visibility();
                    }
                }
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

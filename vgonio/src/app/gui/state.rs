pub(crate) mod camera;
mod context;
mod input;
mod renderer;

// TODO: create default config folder the first time the app is launched (gui
// and cli)

pub use context::RawGuiContext;
use egui_winit::EventResponse;
pub use input::InputState;
pub use renderer::GuiRenderer;

use crate::{
    app::{
        cache::{Handle, InnerCache},
        gfx::{remap_depth, GpuContext, RenderPass, Texture, VertexLayout, WindowSurface},
        gui::event::{EventLoopProxy, VgonioEvent},
    },
    measure::{bsdf::rtc::Ray, params::BsdfMeasurementParams},
};

#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::embr;

use crate::{
    app::{cache::Cache, gui::notify::NotifyKind},
    measure::bsdf::{
        detector::DetectorPatches,
        emitter::{EmitterParams, EmitterSamples, MeasurementPoints},
        rtc::RtcMethod,
    },
};
use std::{
    default::Default,
    ops::Deref,
    path::Path,
    sync::{Arc, RwLock},
};
use vgcore::math::{Mat4, Sph2, Vec3};
use vgsurf::{MicroSurface, MicroSurfaceMesh};
use wgpu::util::DeviceExt;
use winit::{event::WindowEvent, event_loop::EventLoopWindowTarget, window::Window};

pub const AZIMUTH_BIN_SIZE_DEG: usize = 5;
pub const ZENITH_BIN_SIZE_DEG: usize = 2;
pub const AZIMUTH_BIN_SIZE_RAD: f32 = (AZIMUTH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
pub const ZENITH_BIN_SIZE_RAD: f32 = (ZENITH_BIN_SIZE_DEG as f32 * std::f32::consts::PI) / 180.0;
pub const NUM_AZIMUTH_BINS: usize = ((2.0 * std::f32::consts::PI) / AZIMUTH_BIN_SIZE_RAD) as _;
pub const NUM_ZENITH_BINS: usize = ((0.5 * std::f32::consts::PI) / ZENITH_BIN_SIZE_RAD) as _;

pub const DEBUG_DRAWING_SHADER: &str = r#"
struct Uniforms {
    proj_view: mat4x4<f32>,
    lowest: f32,
    highest: f32,
    span: f32,
    scale: f32,
}

struct VOut {
    @builtin(position) position: vec4<f32>,
}

struct PushConstants {
    model: mat4x4<f32>,
    color: vec4<f32>,
}
 
var<push_constant> pcs: PushConstants;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) ->  VOut {
    var vout: VOut;
    let scale = mat4x4<f32>(
        vec4<f32>(uniforms.scale, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, uniforms.scale, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, uniforms.scale, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    vout.position = uniforms.proj_view * scale * pcs.model * vec4<f32>(position, 1.0);

    return vout;
}

@fragment
fn fs_main(vin: VOut) -> @location(0) vec4<f32> {
    return pcs.color;
}
"#;

/// Uniform buffer used when rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct UiUniforms {
    /// Size of the window in logical pixels (points).
    logical_size: [f32; 2],
    /// Padding to align the struct to 16 bytes for the reason of the alignment
    /// requirement of the uniform buffer in WebGL.
    _padding: [f32; 2],
}

impl Default for UiUniforms {
    fn default() -> Self {
        Self {
            logical_size: [0.0, 0.0],
            _padding: [0.0, 0.0],
        }
    }
}

impl From<&ScreenDescriptor> for UiUniforms {
    fn from(desc: &ScreenDescriptor) -> Self {
        Self {
            logical_size: [desc.width as f32, desc.height as f32],
            _padding: [0.0, 0.0],
        }
    }
}

/// Information about the screen on which the UI is presented.
#[derive(Debug)]
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub width: u32,
    /// Height of the window in physical pixel.
    pub height: u32,
    /// HiDPI scale factor; pixels per point.
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    /// Returns the screen width in pixels.
    fn physical_size(&self) -> [u32; 2] { [self.width, self.height] }

    /// Returns the screen width in points.
    fn logical_size(&self) -> [f32; 2] {
        [
            self.width as f32 / self.scale_factor,
            self.height as f32 / self.scale_factor,
        ]
    }
}

// TODO: fix blending.

/// Stores the content of depth buffer.
/// In general the size of the depth map is equal to the size of the window.
/// Width will be recalculated when the window's bytes per row is not a multiple
/// of 256.
pub struct DepthMap {
    pub(crate) depth_attachment: Texture,
    /// Buffer storing depth attachment data.
    pub(crate) depth_attachment_storage: wgpu::Buffer,
    /// Manually padded width to make sure the bytes per row is a multiple of
    /// 256.
    pub(crate) width: u32,
}

impl Deref for DepthMap {
    type Target = Texture;

    fn deref(&self) -> &Self::Target { &self.depth_attachment }
}

impl DepthMap {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            None,
            None,
            None,
            Some("depth-texture"),
        );
        // Manually align the width to 256 bytes.
        let width = (width as f32 * 4.0 / 256.0).ceil() as u32 * 64;
        let depth_attachment_storage_size =
            (std::mem::size_of::<f32>() * (width * height) as usize) as wgpu::BufferAddress;
        let depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            depth_attachment,
            depth_attachment_storage,
            width,
        }
    }

    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        self.depth_attachment = Texture::create_depth_texture(
            &ctx.device,
            width,
            height,
            None,
            None,
            None,
            Some("depth-texture"),
        );
        self.width = (width as f32 * 4.0 / 256.0).ceil() as u32 * 64;
        let depth_map_storage_size =
            (std::mem::size_of::<f32>() * (self.width * height) as usize) as wgpu::BufferAddress;
        self.depth_attachment_storage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_map_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    }

    pub fn copy_to_buffer(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                // texture: &self.depth_attachment.raw,
                texture: &self.depth_attachment.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                // buffer: &self.depth_attachment_storage,
                buffer: &self.depth_attachment_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::mem::size_of::<f32>() as u32 * self.width),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Save current depth buffer content to a PNG file.
    pub fn save_to_image(&mut self, ctx: &GpuContext, path: &Path, surface: &WindowSurface) {
        let mut image = image::GrayImage::new(self.width, surface.height());
        {
            let buffer_slice = self.depth_attachment_storage.slice(..);

            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                receiver.receive().await.unwrap().unwrap();
            });

            let buffer_view_f32 = buffer_slice.get_mapped_range();
            let data_u8 = unsafe {
                let (_, data, _) = buffer_view_f32.align_to::<f32>();
                data.iter()
                    .map(|d| (remap_depth(*d, 0.1, 100.0) * 255.0) as u8)
                    .collect::<Vec<u8>>()
            };

            image.copy_from_slice(&data_u8);
            image.save(path).unwrap();
        }
        self.depth_attachment_storage.unmap();
    }
}

pub struct DebugDrawingState {
    /// If true, the debug drawing is enabled.
    pub(crate) enabled: bool,
    pub(crate) detector_dome_drawing: bool,
    /// Basic render pipeline1 for debug drawing. Topology = TriangleList,
    /// PolygonMode = Line
    pub triangles_pipeline: wgpu::RenderPipeline,
    /// Pipeline for drawing points.
    pub points_pipeline: wgpu::RenderPipeline,
    /// Pipeline for drawing lines.
    pub lines_pipeline: wgpu::RenderPipeline,
    /// Bind group for basic render pipeline.
    pub bind_group: wgpu::BindGroup,
    /// Uniform buffer for basic render pipeline.
    pub uniform_buffer: wgpu::Buffer,
    /// If true, the offline rendering of [`SamplingDebugger`] is enabled.
    pub sampling_debug_enabled: bool,

    /// Whether to show emitter measurement points.
    pub emitter_points_drawing: bool,
    /// Whether to show emitter rays.
    pub emitter_rays_drawing: bool,
    /// Whether to show emitter samples.
    pub emitter_samples_drawing: bool,
    /// Emitter current position.
    pub emitter_position: Sph2,
    /// Emitter samples.
    pub emitter_samples: Option<EmitterSamples>,
    /// Emitter samples buffer.
    pub emitter_samples_buffer: Option<wgpu::Buffer>,
    /// Emitter samples uniform buffer.
    pub emitter_samples_uniform_buffer: Option<wgpu::Buffer>,
    /// Emitter measurement points buffer.
    pub emitter_points_buffer: Option<wgpu::Buffer>,
    /// Rays emitted from the emitter at the current position.
    pub emitter_rays: Option<Vec<Ray>>,
    /// GPU buffer storing all rays emitted from the emitter at the current
    /// position.
    pub emitter_rays_buffer: Option<wgpu::Buffer>,
    /// Ray segment parameter.
    pub emitter_rays_t: f32,
    /// Vertex buffer for the collector dome.
    pub detector_dome_vertex_buffer: Option<wgpu::Buffer>,
    /// Collector's patches.
    pub detector_patches: Option<DetectorPatches>,
    /// Points on the collector's surface. (Rays hitting the collector's
    /// surface)
    pub detector_ray_hit_points_buffer: Option<wgpu::Buffer>,
    /// Whether to show ray hit points on the collector's surface.
    pub detector_ray_hit_points_drawing: bool,
    /// Ray trajectories not hitting the surface.
    pub ray_trajectories_missed_buffer: Option<wgpu::Buffer>,
    /// Ray trajectories hitting the surface.
    pub ray_trajectories_reflected_buffer: Option<wgpu::Buffer>,
    /// Whether to show traced ray trajectories.
    pub ray_trajectories_drawing_reflected: bool,
    /// Whether to show missed ray trajectories.
    pub ray_trajectories_drawing_missed: bool,

    /// Surface of interest.
    surface_of_interest: Option<Handle<MicroSurface>>,
    /// The surface primitive to draw.
    surface_primitive_id: u32,
    /// Whether to show surface primitive.
    surface_primitive_drawing: bool,
    /// Surface normals buffer.
    surface_normals_buffer: Option<wgpu::Buffer>,
    /// Surface normals drawing.
    surface_normals_drawing: bool,
    event_loop: EventLoopProxy,
    cache: Cache,

    pub microsurface: Option<(Handle<MicroSurface>, Handle<MicroSurfaceMesh>)>,
    pub msurf_prim_rp: RenderPass,
    pub msurf_prim_index_buf: wgpu::Buffer,
    pub msurf_prim_index_count: u32,
    pub multiple_prims: bool,
    pub drawing_msurf_prims: bool,
}

impl DebugDrawingState {
    pub const EMITTER_SAMPLES_COLOR: [f32; 4] = [0.27, 1.0, 0.27, 1.0];
    pub const EMITTER_POINTS_COLOR: [f32; 4] = [1.0, 0.27, 0.27, 1.0];
    pub const EMITTER_RAYS_COLOR: [f32; 4] = [0.27, 0.4, 0.8, 0.6];
    pub const COLLECTOR_COLOR: [f32; 4] = [0.2, 0.5, 0.7, 0.5];
    pub const RAY_HIT_POINTS_COLOR: [f32; 4] = [1.0, 0.1, 0.1, 1.0];
    pub const RAY_TRAJECTORIES_MISSED_COLOR: [f32; 4] = [0.6, 0.3, 0.4, 0.2];
    pub const RAY_TRAJECTORIES_REFLECTED_COLOR: [f32; 4] = [0.45, 0.5, 0.4, 0.3];
    pub const SURFACE_PRIMITIVE_COLOR: [f32; 4] = [0.12, 0.23, 0.9, 0.8];
    pub const SURFACE_NORMAL_COLOR: [f32; 4] = [0.7, 0.23, 0.92, 0.8];

    pub fn new(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
        event_loop: EventLoopProxy,
        cache: Arc<RwLock<InnerCache>>,
    ) -> Self {
        let vert_layout = VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None);
        let vert_buffer_layout = vert_layout.buffer_layout(wgpu::VertexStepMode::Vertex);
        let prim_shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("debug-drawing-prim-shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../gui/assets/shaders/wgsl/prim.wgsl").into(),
                ),
            });
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("debug-drawing-rays-bind-group-layout"),
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
        let msurf_prim_rp_uniform_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug-drawing-msurf-uniform-buffer"),
                    contents: bytemuck::cast_slice(&[0.0f32; 16 * 3 + 4]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let msurf_prim_index_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-msurf-prim-index-buffer"),
            size: std::mem::size_of::<u32>() as u64 * 1024, // initial capacity of 1024 rays
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let collector_shape_vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-drawing-collector-shape-vertices-buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let collector_shape_index_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-drawing-collector-shape-indices-buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let (triangles_pipeline, points_pipeline, lines_pipeline, bind_group, uniform_buffer) = {
            let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("debug-drawing-basic-uniform-buffer"),
                size: std::mem::size_of::<[f32; 16 + 4]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group_layout =
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("debug-drawing-basic-bind-group-layout"),
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
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("debug-drawing-basic-bind-group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            let shader_module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("debug-drawing-shader"),
                    source: wgpu::ShaderSource::Wgsl(DEBUG_DRAWING_SHADER.into()),
                });
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("debug-drawing-shared-pipeline-layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[wgpu::PushConstantRange {
                            stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            range: 0..80,
                        }],
                    });
            let vert_state = wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[vert_buffer_layout.clone()],
            };
            let frag_state = wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            };
            let depth_state = wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            };
            let triangles_pipeline =
                ctx.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("debug-drawing-basic-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: vert_state.clone(),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            front_face: wgpu::FrontFace::Ccw,
                            polygon_mode: wgpu::PolygonMode::Line,
                            ..Default::default()
                        },
                        depth_stencil: Some(depth_state.clone()),
                        multisample: Default::default(),
                        fragment: Some(frag_state.clone()),
                        multiview: None,
                    });
            let points_pipeline =
                ctx.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("debug-drawing-points-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: vert_state.clone(),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::PointList,
                            front_face: wgpu::FrontFace::Ccw,
                            ..Default::default()
                        },
                        depth_stencil: Some(depth_state.clone()),
                        multisample: Default::default(),
                        fragment: Some(frag_state.clone()),
                        multiview: None,
                    });
            let lines_pipeline =
                ctx.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("debug-drawing-lines-pipeline"),
                        layout: Some(&pipeline_layout),
                        vertex: vert_state.clone(),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::LineList,
                            front_face: wgpu::FrontFace::Ccw,
                            ..Default::default()
                        },
                        depth_stencil: Some(depth_state.clone()),
                        multisample: Default::default(),
                        fragment: Some(frag_state.clone()),
                        multiview: None,
                    });
            (
                triangles_pipeline,
                points_pipeline,
                lines_pipeline,
                bind_group,
                uniform_buffer,
            )
        };

        Self {
            enabled: false,
            detector_dome_vertex_buffer: None,
            triangles_pipeline,
            points_pipeline,
            lines_pipeline,
            bind_group,
            uniform_buffer,
            sampling_debug_enabled: false,
            emitter_points_drawing: false,
            emitter_rays_drawing: false,
            emitter_samples_drawing: false,
            emitter_position: Sph2::zero(),
            emitter_samples: None,
            emitter_samples_buffer: None,
            emitter_samples_uniform_buffer: None,
            emitter_points_buffer: None,
            emitter_rays: None,
            emitter_rays_buffer: None,
            emitter_rays_t: 1.0,
            msurf_prim_rp: RenderPass {
                pipeline: ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("debug-msurf-prim-pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: Some("debug-prim-pipeline-layout"),
                                bind_group_layouts: &[&bind_group_layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        vertex: wgpu::VertexState {
                            module: &prim_shader_module,
                            entry_point: "vs_main",
                            buffers: &[VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None)
                                .buffer_layout(wgpu::VertexStepMode::Vertex)],
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
                        multisample: Default::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &prim_shader_module,
                            entry_point: "fs_main",
                            targets: &[Some(wgpu::ColorTargetState {
                                format: target_format,
                                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview: None,
                    }),
                bind_groups: vec![ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("debug-prim-bind-group"),
                    layout: &ctx.device.create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: Some("debug-drawing-prim-bind-group-layout"),
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
                        },
                    ),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: msurf_prim_rp_uniform_buffer.as_entire_binding(),
                    }],
                })],
                uniform_buffers: Some(vec![msurf_prim_rp_uniform_buffer]),
            },
            msurf_prim_index_buf,
            msurf_prim_index_count: 0,
            multiple_prims: false,
            drawing_msurf_prims: false,
            detector_dome_drawing: false,
            event_loop,
            detector_patches: None,
            detector_ray_hit_points_buffer: None,
            detector_ray_hit_points_drawing: false,
            ray_trajectories_missed_buffer: None,
            cache: Cache(cache),
            ray_trajectories_drawing_reflected: false,
            ray_trajectories_reflected_buffer: None,
            ray_trajectories_drawing_missed: false,
            surface_of_interest: None,
            surface_primitive_id: 0,
            surface_primitive_drawing: false,
            surface_normals_buffer: None,
            microsurface: None,
            surface_normals_drawing: false,
        }
    }

    /// Update the uniform buffer for the BSDF measurement debug view.
    pub fn update_uniform_buffer(
        &mut self,
        ctx: &GpuContext,
        proj_view_mat: &Mat4,
        lowest: f32,
        highest: f32,
        scale: f32,
    ) {
        let mut buf = [0f32; 20];
        buf[0..16].copy_from_slice(&proj_view_mat.to_cols_array());
        buf[16] = lowest;
        buf[17] = highest;
        buf[18] = highest - lowest;
        buf[19] = scale;
        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&buf));
    }

    pub fn update_emitter_samples(&mut self, ctx: &GpuContext, samples: EmitterSamples) {
        self.update_emitter_position_and_samples(ctx, Some(samples), None);
    }

    pub fn update_measurement_points(&mut self, ctx: &GpuContext, points: MeasurementPoints) {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-emitter-points"),
            size: points.len() as u64 * std::mem::size_of::<Vec3>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (orbit_radius, _) = self.estimate_radii().unwrap();
        let vertices = points
            .iter()
            .map(|s| s.to_cartesian() * orbit_radius)
            .collect::<Vec<_>>();
        ctx.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&vertices));
        self.emitter_points_buffer = Some(buffer);
    }

    pub fn update_emitter_position(&mut self, ctx: &Arc<GpuContext>, position: Sph2) {
        log::trace!("Updating emitter position to {}", position);
        self.update_emitter_position_and_samples(ctx, None, Some(position));
    }

    pub fn update_ray_params(&mut self, ctx: &Arc<GpuContext>, t: f32) {
        self.emitter_rays_t = t;
        self.update_emitter_position_and_samples(ctx, None, None);
    }

    fn update_emitter_position_and_samples(
        &mut self,
        ctx: &GpuContext,
        samples: Option<EmitterSamples>,
        position: Option<Sph2>,
    ) {
        if self.microsurface.is_none() {
            return;
        }

        if let Some(new_samples) = samples {
            if self
                .emitter_samples_buffer
                .as_ref()
                .is_some_and(|buf| buf.size() < new_samples.len() as u64 * 12)
                || self.emitter_samples_buffer.is_none()
            {
                self.emitter_samples_buffer =
                    Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("debug-emitter-samples"),
                        size: new_samples.len() as u64 * std::mem::size_of::<Vec3>() as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
            ctx.queue.write_buffer(
                self.emitter_samples_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&new_samples),
            );
            self.emitter_samples = Some(new_samples);
        }

        if let Some(pos) = position {
            self.emitter_position = pos;
        }

        if let Some(samples) = &self.emitter_samples {
            let (orbit_radius, disc_radius) = self.estimate_radii().unwrap();
            let vertices = EmitterParams::transform_samples(
                samples,
                self.emitter_position,
                orbit_radius,
                disc_radius,
            );
            ctx.queue.write_buffer(
                self.emitter_samples_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&vertices),
            );
        };

        if self.emitter_rays.is_some() {
            self.emit_rays(ctx);
        }
    }

    pub fn estimate_radii(&self) -> Option<(f32, f32)> {
        self.microsurface.map(|(_, mesh)| {
            self.cache.read(|cache| {
                let mesh = cache.get_micro_surface_mesh(mesh).unwrap();
                (
                    crate::measure::estimate_orbit_radius(mesh),
                    crate::measure::estimate_disc_radius(mesh),
                )
            })
        })
    }

    pub fn emit_rays(&mut self, ctx: &GpuContext) {
        if self.emitter_samples.is_none() {
            log::trace!("No emitter samples to emit rays from");
            self.event_loop
                .send_event(VgonioEvent::Notify {
                    kind: NotifyKind::Warning,
                    text: "Generate samples before generate rays".to_string(),
                    time: 4.0,
                })
                .unwrap();
            return;
        }
        log::trace!("Emitting rays at {}", self.emitter_position);
        let (orbit_radius, disc_radius) = self.estimate_radii().unwrap();
        self.emitter_rays = Some(EmitterParams::emit_rays(
            self.emitter_samples.as_ref().unwrap(),
            self.emitter_position,
            orbit_radius,
            disc_radius,
        ));
        let ray_segments = self
            .emitter_rays
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|r| [r.org, r.org + r.dir * self.emitter_rays_t])
            .collect::<Vec<_>>();
        self.emitter_rays_buffer = Some(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("debug-emitter-rays"),
                contents: bytemuck::cast_slice(&ray_segments),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            },
        ));
    }

    pub fn update_detector_drawing(
        &mut self,
        ctx: &GpuContext,
        status: bool,
        patches: DetectorPatches,
    ) {
        log::trace!("[DebugDrawingState] Updating collector drawing");
        self.detector_dome_drawing = status;
        if self.microsurface.is_none() {
            log::debug!("No microsurface to draw collector");
            return;
        }
        match &patches {
            DetectorPatches::Beckers { rings, patches } => {
                let mut vertices = vec![];
                // Generate from rings.
                let (disc_xs, disc_ys): (Vec<_>, Vec<_>) = (0..360)
                    .map(|i| {
                        let a = i as f32 * std::f32::consts::PI / 180.0;
                        (a.cos(), a.sin())
                    })
                    .unzip();
                for ring in rings.iter() {
                    let inner_radius = ring.theta_inner.sin();
                    let outer_radius = ring.theta_outer.sin();
                    let inner_height = ring.theta_inner.cos();
                    let outer_height = ring.theta_outer.cos();
                    // Generate the outer border of the ring
                    for (j, (x, y)) in disc_xs.iter().zip(disc_ys.iter()).enumerate() {
                        vertices.push(Vec3::new(
                            *x * outer_radius,
                            *y * outer_radius,
                            outer_height,
                        ));
                        vertices.push(Vec3::new(
                            disc_xs[(j + 1) % 360] * outer_radius,
                            disc_ys[(j + 1) % 360] * outer_radius,
                            outer_height,
                        ));
                    }
                    let step = std::f32::consts::TAU / ring.patch_count as f32;
                    // Generate the cells
                    if ring.patch_count == 1 {
                        continue;
                    } else {
                        for k in 0..ring.patch_count {
                            let x = (step * k as f32).cos();
                            let y = (step * k as f32).sin();
                            vertices.push(Vec3::new(
                                x * inner_radius,
                                y * inner_radius,
                                inner_height,
                            ));
                            vertices.push(Vec3::new(
                                x * outer_radius,
                                y * outer_radius,
                                outer_height,
                            ));
                        }
                    }
                }
                self.detector_dome_vertex_buffer = Some(ctx.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("debug-collector-dome"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
            DetectorPatches::Tregenza => {}
        }
        self.detector_patches = Some(patches);
    }

    pub fn prepare_surface_normals_buffer(&mut self, ctx: &GpuContext) {
        match self.microsurface {
            None => {}
            Some((_, mesh_hdl)) => {
                let normals = self.cache.read(|cache| {
                    let mesh = cache.get_micro_surface_mesh(mesh_hdl).unwrap();
                    mesh.facet_normals
                        .iter()
                        .zip(mesh.facets.chunks(3))
                        .flat_map(|(n, f)| {
                            let center = f
                                .iter()
                                .fold(Vec3::ZERO, |acc, v| acc + mesh.verts[*v as usize] / 3.0);
                            [center, center + *n * 0.5]
                        })
                        .collect::<Vec<_>>()
                });
                self.surface_normals_buffer = Some(ctx.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("debug-surface-normals"),
                        contents: bytemuck::cast_slice(&normals),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
            }
        };
    }

    pub fn toggle_surface_normals(&mut self) {
        self.surface_normals_drawing = !self.surface_normals_drawing;
    }

    pub fn update_surface_primitive_id(
        &mut self,
        surf: Option<Handle<MicroSurface>>,
        id: u32,
        status: bool,
    ) {
        log::trace!(
            "[DebugDrawingState] Updating surface {:?} primitive id to {}",
            surf,
            id
        );
        self.surface_of_interest = surf;
        self.surface_primitive_id = id;
        self.surface_primitive_drawing = status;
    }

    /// Updates the ray trajectories for the debugging draw calls.
    pub fn update_ray_trajectories(
        &mut self,
        ctx: &GpuContext,
        method: RtcMethod,
        params: BsdfMeasurementParams,
        mesh: Handle<MicroSurfaceMesh>,
    ) {
        if self.emitter_samples.is_none() {
            self.event_loop
                .send_event(VgonioEvent::Notify {
                    kind: NotifyKind::Warning,
                    text: "Please generate samples before tracing".to_string(),
                    time: 4.0,
                })
                .unwrap();
            return;
        }

        if self.detector_patches.is_none() {
            self.event_loop
                .send_event(VgonioEvent::Notify {
                    kind: NotifyKind::Warning,
                    text: "Please generate patches before tracing".to_string(),
                    time: 4.0,
                })
                .unwrap();
            return;
        }

        match method {
            #[cfg(feature = "embree")]
            RtcMethod::Embree => {
                todo!()
                //     #[cfg(debug_assertions)]
                //     {
                //         let measured = self.cache.read(|cache| {
                //             let mesh =
                // cache.get_micro_surface_mesh(mesh).unwrap();
                //             embr::measure_bsdf_once(
                //                 &params,
                //                 mesh,
                //                 self.emitter_samples.as_ref().unwrap(),
                //                 self.collector_patches.as_ref().unwrap(),
                //                 self.emitter_position,
                //                 &self.cache.0,
                //             )
                //         });
                //
                //         let mut reflected = vec![];
                //         let mut missed = vec![];
                //
                //         for trajectory in measured.trajectories.iter() {
                //             let mut iter = trajectory.0.iter().peekable();
                //             let mut i = 0;
                //             while let Some(node) = iter.next() {
                //                 let org: Vec3 = node.org.into();
                //                 let dir: Vec3 = node.dir.into();
                //                 match iter.peek() {
                //                     None => {
                //                         // Last node
                //                         if i == 0 {
                //                             missed.push(org);
                //                             missed.push(org + dir *
                // self.collector_orbit_radius * 1.2);
                //                         } else {
                //                             reflected.push(org);
                //                             reflected
                //                                 .push(org + dir *
                // self.collector_orbit_radius * 1.2);
                //                         }
                //                         i += 1;
                //                     }
                //                     Some(next) => {
                //                         reflected.push(org);
                //                         reflected.push(next.org.into());
                //                         i += 1;
                //                     }
                //                 }
                //             }
                //         }
                //         self.ray_trajectories_missed_buffer =
                // Some(ctx.device.create_buffer_init(
                //             &wgpu::util::BufferInitDescriptor {
                //                 label:
                // Some("debug-collector-ray-trajectories-missed"),
                //                 contents: bytemuck::cast_slice(&missed),
                //                 usage: wgpu::BufferUsages::VERTEX |
                // wgpu::BufferUsages::COPY_DST,             },
                //         ));
                //         self.ray_trajectories_reflected_buffer =
                // Some(ctx.device.create_buffer_init(
                //             &wgpu::util::BufferInitDescriptor {
                //                 label:
                // Some("debug-collector-ray-trajectories-reflected"),
                //                 contents: bytemuck::cast_slice(&reflected),
                //                 usage: wgpu::BufferUsages::VERTEX |
                // wgpu::BufferUsages::COPY_DST,             },
                //         ));
                //         self.collector_ray_hit_points_buffer =
                // Some(ctx.device.create_buffer_init(
                //             &wgpu::util::BufferInitDescriptor {
                //                 label:
                // Some("debug-collector-ray-hit-points"),
                //                 contents:
                // bytemuck::cast_slice(&measured.hit_points),
                //                 usage: wgpu::BufferUsages::VERTEX |
                // wgpu::BufferUsages::COPY_DST,             },
                //         ));
                //     }
                //     #[cfg(not(debug_assertions))]
                //     log::error!("Ray trajectories can only be drawn in debug
                // mode");
            }
            #[cfg(feature = "optix")]
            RtcMethod::Optix => {
                log::info!("OptiX is not supported for ray trajectory drawing");
            }
            RtcMethod::Grid => {
                log::info!("Grid is not supported for ray trajectory drawing");
            }
        }
    }

    /// Records the render process for the debugging draw calls.
    ///
    /// Only records the render pass if the debug renderer is enabled.
    pub fn record_render_pass(
        &self,
        ctx: &GpuContext,
        color_output: Option<wgpu::RenderPassColorAttachment>,
        depth_output: Option<wgpu::RenderPassDepthStencilAttachment>,
    ) -> Option<wgpu::CommandEncoder> {
        self.cache.read(|cache| {
            if !self.enabled {
                return None;
            }
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("debug-render-pass"),
                });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("debug-render-pass"),
                    color_attachments: &[color_output],
                    depth_stencil_attachment: depth_output,
                });

                let mut constants = [0.0f32; 20];
                // Draw emitter samples.
                if let Some(samples) = &self.emitter_samples_buffer {
                    if self.emitter_samples_drawing {
                        // Convert range in bytes to range in vertices.
                        render_pass.set_pipeline(&self.points_pipeline);
                        render_pass.set_bind_group(0, &self.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, samples.slice(..));
                        constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                        constants[16..20].copy_from_slice(&Self::EMITTER_SAMPLES_COLOR);
                        render_pass.set_push_constants(
                            wgpu::ShaderStages::VERTEX_FRAGMENT,
                            0,
                            bytemuck::cast_slice(&constants),
                        );
                        render_pass
                            .draw(0..self.emitter_samples.as_ref().unwrap().len() as u32, 0..1);
                    }

                    if self.emitter_rays_drawing && self.emitter_rays_buffer.is_some() {
                        let rays = self.emitter_rays_buffer.as_ref().unwrap();
                        render_pass.set_pipeline(&self.lines_pipeline);
                        render_pass.set_bind_group(0, &self.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, rays.slice(..));
                        constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                        constants[16..20].copy_from_slice(&Self::EMITTER_RAYS_COLOR);
                        render_pass.set_push_constants(
                            wgpu::ShaderStages::VERTEX_FRAGMENT,
                            0,
                            bytemuck::cast_slice(&constants),
                        );
                        render_pass.draw(
                            0..self.emitter_rays.as_ref().unwrap().len() as u32 * 2,
                            0..1,
                        );
                    }
                }

                if self.emitter_points_drawing && self.emitter_points_buffer.is_some() {
                    let buffer = self.emitter_points_buffer.as_ref().unwrap();
                    render_pass.set_pipeline(&self.points_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                    constants[16..20].copy_from_slice(&Self::EMITTER_POINTS_COLOR);
                    render_pass.set_push_constants(
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        0,
                        bytemuck::cast_slice(&constants),
                    );
                    render_pass.draw(0..buffer.size() as u32 / 12, 0..1);
                }

                log::trace!(
                    "Primitive drawing: {}, {}",
                    self.surface_primitive_drawing,
                    self.surface_of_interest.is_some()
                );
                if self.surface_primitive_drawing && self.surface_of_interest.is_some() {
                    let surface = self.surface_of_interest.unwrap();
                    let mesh = cache
                        .get_micro_surface_renderable_mesh_by_surface_id(surface)
                        .unwrap();
                    render_pass.set_pipeline(&self.triangles_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                    constants[16..20].copy_from_slice(&Self::SURFACE_PRIMITIVE_COLOR);
                    render_pass.set_push_constants(
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        0,
                        bytemuck::cast_slice(&constants),
                    );
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(
                        self.surface_primitive_id * 3..(self.surface_primitive_id + 1) * 3,
                        0,
                        0..1,
                    );
                }

                if self.surface_normals_buffer.is_some() && self.surface_normals_drawing {
                    let buffer = self.surface_normals_buffer.as_ref().unwrap();
                    render_pass.set_pipeline(&self.lines_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    render_pass.set_vertex_buffer(
                        0,
                        self.surface_normals_buffer.as_ref().unwrap().slice(..),
                    );
                    constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                    constants[16..20].copy_from_slice(&Self::SURFACE_NORMAL_COLOR);
                    render_pass.set_push_constants(
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        0,
                        bytemuck::cast_slice(&constants),
                    );
                    render_pass.draw(0..buffer.size() as u32 / 12, 0..1);
                }

                if self.ray_trajectories_drawing_reflected || self.ray_trajectories_drawing_missed {
                    render_pass.set_pipeline(&self.lines_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());

                    if self.ray_trajectories_drawing_missed {
                        if let Some(missed_buffer) = &self.ray_trajectories_missed_buffer {
                            constants[16..20].copy_from_slice(&Self::RAY_TRAJECTORIES_MISSED_COLOR);
                            render_pass.set_vertex_buffer(0, missed_buffer.slice(..));
                            render_pass.set_push_constants(
                                wgpu::ShaderStages::VERTEX_FRAGMENT,
                                0,
                                bytemuck::cast_slice(&constants),
                            );
                            render_pass.draw(0..missed_buffer.size() as u32 / 12, 0..1);
                        }
                    }

                    if self.ray_trajectories_drawing_reflected {
                        if let Some(reflected_buffer) = &self.ray_trajectories_reflected_buffer {
                            constants[16..20]
                                .copy_from_slice(&Self::RAY_TRAJECTORIES_REFLECTED_COLOR);
                            render_pass.set_vertex_buffer(0, reflected_buffer.slice(..));
                            render_pass.set_push_constants(
                                wgpu::ShaderStages::VERTEX_FRAGMENT,
                                0,
                                bytemuck::cast_slice(&constants),
                            );
                            render_pass.draw(0..reflected_buffer.size() as u32 / 12, 0..1);
                        }
                    }
                }

                if self.detector_ray_hit_points_drawing
                    && self.detector_ray_hit_points_buffer.is_some()
                {
                    let buffer = self.detector_ray_hit_points_buffer.as_ref().unwrap();
                    render_pass.set_pipeline(&self.points_pipeline);
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                    constants[16..20].copy_from_slice(&Self::RAY_HIT_POINTS_COLOR);
                    render_pass.set_push_constants(
                        wgpu::ShaderStages::VERTEX_FRAGMENT,
                        0,
                        bytemuck::cast_slice(&constants),
                    );
                    render_pass.draw(0..buffer.size() as u32 / 12, 0..1);
                }

                if self.detector_dome_drawing {
                    let (orbit_radius, _) = self.estimate_radii().unwrap();
                    render_pass.set_bind_group(0, &self.bind_group, &[]);
                    if self.detector_dome_vertex_buffer.is_some() {
                        let vertex_buffer = self.detector_dome_vertex_buffer.as_ref().unwrap();
                        render_pass.set_pipeline(&self.lines_pipeline);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        constants[0..16].copy_from_slice(
                            &Mat4::from_scale(Vec3::splat(orbit_radius)).to_cols_array(),
                        );
                        constants[16..20].copy_from_slice(&Self::COLLECTOR_COLOR);
                        render_pass.set_push_constants(
                            wgpu::ShaderStages::VERTEX_FRAGMENT,
                            0,
                            bytemuck::cast_slice(&constants),
                        );
                        render_pass.draw(0..vertex_buffer.size() as u32 / 12, 0..1);
                    }
                }
            }

            Some(encoder)
        })
    }
}

/// Context for rendering the UI.
pub struct GuiContext {
    //// The wgpu device.
    device: Arc<wgpu::Device>,
    //// The wgpu queue.
    queue: Arc<wgpu::Queue>,
    /// Context for GUI painting.
    context: RawGuiContext,
    /// Rendering state for the GUI.
    pub renderer: Arc<RwLock<GuiRenderer>>, // TODO: make private
}

pub struct GuiRenderOutput {
    pub user_cmds: Vec<wgpu::CommandBuffer>,
    pub ui_cmd: wgpu::CommandBuffer,
}

impl GuiContext {
    /// Creates related resources used for UI rendering.
    ///
    /// If the format passed is not a *Srgb format, the shader will
    /// automatically convert to sRGB colors in the shader.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        event_loop: &EventLoopWindowTarget<VgonioEvent>,
        msaa_samples: u32,
    ) -> Self {
        let context = RawGuiContext::new(event_loop);
        let renderer = Arc::new(RwLock::new(GuiRenderer::new(
            &device,
            surface_format,
            None,
            msaa_samples,
        )));
        Self {
            device,
            queue,
            context,
            renderer,
        }
    }

    /// Prepares the rendering. This should be called before any rendering.
    pub fn update(&mut self, window: &Window) { self.context.prepare(window); }

    /// Returns the encapsulated GUI context.
    pub fn ctx(&self) -> &RawGuiContext { &self.context }

    /// Run the UI and record the rendering commands.
    pub fn render(
        &mut self,
        window: &Window,
        screen_desc: ScreenDescriptor,
        target: &wgpu::TextureView,
        ui: impl FnOnce(&egui::Context),
    ) -> GuiRenderOutput {
        let mut ui_cmd_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gui-command-encoder"),
                });
        let output = self.context.run(ui);
        self.context
            .handle_platform_output(window, output.platform_output);

        let primitives = self.context.inner.tessellate(output.shapes);

        {
            let mut renderer = self.renderer.write().unwrap();
            let user_cmds = {
                for (id, image_delta) in &output.textures_delta.set {
                    renderer.update_texture(&self.device, &self.queue, *id, image_delta);
                }
                renderer.update_buffers(
                    &self.device,
                    &self.queue,
                    &mut ui_cmd_encoder,
                    &primitives,
                    &screen_desc,
                )
            };

            {
                let mut render_pass =
                    ui_cmd_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gui_render_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                renderer.render(&mut render_pass, &primitives, &screen_desc);
            }

            {
                for id in &output.textures_delta.free {
                    renderer.remove_texture(*id);
                }
            }

            GuiRenderOutput {
                user_cmds,
                ui_cmd: ui_cmd_encoder.finish(),
            }
        }
    }

    /// Update the context whenever there is a window event.
    pub fn on_window_event(&mut self, event: &WindowEvent) -> EventResponse {
        self.context.on_window_event(event)
    }
}

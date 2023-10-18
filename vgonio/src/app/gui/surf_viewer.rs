use egui::{Ui, WidgetText};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
};
use uuid::Uuid;

use vgcore::math::{Mat4, Vec4};
use vgsurf::MicroSurface;

use crate::app::{
    cache::{Handle, InnerCache},
    gfx::{
        camera::{ProjectionKind, ViewProjUniform},
        GpuContext, Texture,
    },
    gui::{
        data::MicroSurfaceProp,
        docking::{Dockable, WidgetKind},
        event::{EventLoopProxy, SurfaceViewerEvent, VgonioEvent},
        state::{camera::CameraState, debug::DebugDrawingState, InputState},
        theme::ThemeState,
        visual_grid::{VisualGridPipeline, VisualGridState},
    },
};

use super::state::{DepthMap, GuiRenderer};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadingMode {
    #[default]
    Wireframe,
    Flat,
    Smooth,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OverlayFlags {
    pub normals: bool,
    pub tangents: bool,
    pub bitangents: bool,
    pub uvs: bool,
}

/// State of a single surface viewer.
pub struct SurfaceViewerState {
    /// Color attachment ID.
    c_attachment_id: egui::TextureId,
    /// Color attachment.
    c_attachment: Texture,
    /// Depth attachment.
    d_attachment: DepthMap,
    /// Camera state.
    camera: CameraState,
    /// State of the micro surface rendering.
    surface: MicroSurfaceState,
    /// State of the visual grid rendering.
    visual_grid: VisualGridState,
    /// Overlay flags.
    overlay: OverlayFlags,
    /// Shading mode.
    shading: ShadingMode,
}

impl SurfaceViewerState {
    pub fn new(
        gpu: &GpuContext,
        gui: &Arc<RwLock<GuiRenderer>>,
        format: wgpu::TextureFormat,
        c_attachment_id: egui::TextureId,
        surf_globals_layout: &wgpu::BindGroupLayout,
        surf_locals_layout: &wgpu::BindGroupLayout,
        grid_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampling-debugger-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        let c_attachment = Texture::new(
            &gpu.device,
            &wgpu::TextureDescriptor {
                label: Some(format!("surf_viewer_{:?}_color_attachment", c_attachment_id).as_str()),
                size: wgpu::Extent3d {
                    width: SurfaceViewer::DEFAULT_WIDTH,
                    height: SurfaceViewer::DEFAULT_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Some(sampler),
        );
        let d_attachment = DepthMap::new(
            gpu,
            SurfaceViewer::DEFAULT_WIDTH,
            SurfaceViewer::DEFAULT_HEIGHT,
        );
        gui.write().unwrap().update_egui_texture_from_wgpu_texture(
            &gpu.device,
            &c_attachment.view,
            wgpu::FilterMode::Linear,
            c_attachment_id,
        );
        Self {
            c_attachment_id,
            c_attachment,
            d_attachment,
            camera: CameraState::default_with_size(
                SurfaceViewer::DEFAULT_WIDTH,
                SurfaceViewer::DEFAULT_HEIGHT,
            ),
            surface: MicroSurfaceState::new(gpu, surf_globals_layout, surf_locals_layout),
            visual_grid: VisualGridState::new(gpu, grid_bind_group_layout),
            overlay: Default::default(),
            shading: Default::default(),
        }
    }

    pub fn camera_view_proj(&self) -> ViewProjUniform { self.camera.uniform.view_proj }

    pub fn depth_attachment(&self) -> &Texture { &self.d_attachment.depth_attachment }

    pub fn colour_attachment(&self) -> &Texture { &self.c_attachment }

    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        gpu: &GpuContext,
        gui: &Arc<RwLock<GuiRenderer>>,
    ) {
        let sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampling-debugger-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        self.c_attachment = Texture::new(
            &gpu.device,
            &wgpu::TextureDescriptor {
                label: Some(
                    format!("surf_viewer_{:?}_color_attachment", self.c_attachment).as_str(),
                ),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Some(sampler),
        );
        gui.write().unwrap().update_egui_texture_from_wgpu_texture(
            &gpu.device,
            &self.c_attachment.view,
            wgpu::FilterMode::Linear,
            self.c_attachment_id,
        );
        self.d_attachment.resize(gpu, width, height);
        self.camera.projection.resize(width, height);
    }

    pub fn update(
        &mut self,
        surfaces: &[(&Handle<MicroSurface>, &MicroSurfaceProp)],
        input: &InputState,
        dt: std::time::Duration,
        theme: &ThemeState,
        gpu: &GpuContext,
        update_camera: bool,
    ) {
        if update_camera {
            self.camera.update(input, dt, ProjectionKind::Perspective);
        }
        let view_proj = self.camera.uniform.view_proj;
        let view_proj_inv = self.camera.uniform.view_proj_inv;
        self.surface.update(gpu, &view_proj, surfaces);
        self.visual_grid.update(
            gpu,
            &view_proj,
            &view_proj_inv,
            theme.visuals().grid_line_color,
            theme.kind(),
        )
    }
}

/// Pipeline for rendering micro surfaces.
pub(crate) struct SurfacePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub globals_bind_group_layout: wgpu::BindGroupLayout,
    pub locals_bind_group_layout: wgpu::BindGroupLayout,
    pub normals_pipeline: wgpu::RenderPipeline,
}

/// State of multiple surface viewers.
pub struct SurfaceViewerStates {
    /// Output format of the surface viewer.
    format: wgpu::TextureFormat,
    /// Render pipeline for rendering micro surfaces.
    surf_pipeline: SurfacePipeline,
    /// Render pipeline for rendering the visual grid.
    grid_pipeline: VisualGridPipeline,
    /// States of the micro surface rendering for each surface viewer.
    states: HashMap<Uuid, SurfaceViewerState>,
    /// List of surfaces ready to be rendered.
    surfaces: HashSet<Handle<MicroSurface>>,
}

impl SurfaceViewerStates {
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let surf_pipeline = MicroSurfaceState::create_pipeline(ctx, target_format);
        let grid_pipeline = VisualGridState::create_pipeline(ctx, target_format);
        Self {
            format: target_format,
            surf_pipeline,
            grid_pipeline,
            states: Default::default(),
            surfaces: Default::default(),
        }
    }

    pub fn viewer_state(&self, viewer: Uuid) -> Option<&SurfaceViewerState> {
        self.states.get(&viewer)
    }

    pub fn update_surfaces_list(&mut self, surfaces: &[Handle<MicroSurface>]) {
        for surface in surfaces {
            self.surfaces.insert(*surface);
        }
        for (_, state) in self.states.iter_mut() {
            state.surface.update_surfaces_list(surfaces.iter());
        }
    }

    /// Updates the overlay flags of a surface viewer.
    pub fn update_overlay(&mut self, viewer: Uuid, overlay: OverlayFlags) {
        if let Some(state) = self.states.get_mut(&viewer) {
            state.overlay = overlay;
        }
    }

    /// Updates the shading mode of a surface viewer.
    pub fn update_shading(&mut self, viewer: Uuid, shading: ShadingMode) {
        if let Some(state) = self.states.get_mut(&viewer) {
            state.shading = shading;
        }
    }

    /// Allocates resources for a new surface viewer.
    pub fn allocate_viewer_resources(
        &mut self,
        viewer: Uuid,
        tex_id: egui::TextureId,
        gpu: &GpuContext,
        gui: &Arc<RwLock<GuiRenderer>>,
    ) {
        assert!(
            self.states.get(&viewer).is_none(),
            "Surface viewer with the same UUID already exists"
        );
        let mut state = SurfaceViewerState::new(
            gpu,
            gui,
            self.format,
            tex_id,
            &self.surf_pipeline.globals_bind_group_layout,
            &self.surf_pipeline.locals_bind_group_layout,
            &self.grid_pipeline.bind_group_layout,
        );
        state.surface.update_surfaces_list(self.surfaces.iter());
        self.states.insert(viewer, state);
    }

    /// Resize the viewport of a surface viewer.
    pub fn resize_viewport(
        &mut self,
        viewer: Uuid,
        width: u32,
        height: u32,
        gpu: &GpuContext,
        gui: &Arc<RwLock<GuiRenderer>>,
    ) {
        if let Some(state) = self.states.get_mut(&viewer) {
            state.resize(width, height, self.format, gpu, gui);
        }
    }

    pub fn record_render_pass(
        &mut self,
        gpu: &GpuContext,
        active_viewer: Option<Uuid>,
        input: &InputState,
        dt: std::time::Duration,
        theme: &ThemeState,
        cache: &InnerCache,
        surfaces: &[(&Handle<MicroSurface>, &MicroSurfaceProp)],
    ) -> wgpu::CommandEncoder {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vgonio_render_encoder"),
            });
        for (viewer, state) in &mut self.states {
            if let Some(active) = active_viewer {
                if active == *viewer {
                    state.update(surfaces, input, dt, theme, gpu, true);
                }
            } else {
                state.update(surfaces, input, dt, theme, gpu, false);
            }
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(
                    format!("surface_viewer_render_pass_{:?}", state.c_attachment_id).as_str(),
                ),
                color_attachments: &[Some(
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &state.c_attachment.view,
                        // This is the texture that will receive the resolved output; will be
                        // the same as `view` unless multisampling.
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(theme.visuals().clear_color),
                            store: true,
                        },
                    },
                )],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.d_attachment.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            let aligned_micro_surface_uniform_size =
                MicroSurfaceUniforms::aligned_size(&gpu.device);

            if !surfaces.is_empty() {
                render_pass.set_pipeline(&self.surf_pipeline.pipeline);
                render_pass.set_bind_group(0, &state.surface.globals_bind_group, &[]);

                for (hdl, _) in surfaces.iter() {
                    let renderable = cache.get_micro_surface_renderable_mesh_by_surface_id(**hdl);
                    if renderable.is_none() {
                        log::debug!(
                            "Failed to get renderable mesh for surface {:?}, skipping.",
                            hdl
                        );
                        continue;
                    }
                    let buf_index = state
                        .surface
                        .locals_lookup
                        .iter()
                        .position(|x| x == *hdl)
                        .unwrap();
                    let renderable = renderable.unwrap();
                    render_pass.set_bind_group(
                        1,
                        &state.surface.locals_bind_group,
                        &[buf_index as u32 * aligned_micro_surface_uniform_size],
                    );
                    render_pass.set_vertex_buffer(0, renderable.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        renderable.index_buffer.slice(..),
                        renderable.index_format,
                    );
                    render_pass.draw_indexed(0..renderable.indices_count, 0, 0..1);

                    if state.overlay.normals {
                        let buffer = renderable.normal_buffer.as_ref().unwrap();
                        render_pass.set_pipeline(&self.surf_pipeline.normals_pipeline);
                        render_pass.set_vertex_buffer(0, buffer.slice(..));
                        let mut constants = [0.0; 16 + 4];
                        constants[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
                        constants[16..20].copy_from_slice(&DebugDrawingState::SURFACE_NORMAL_COLOR);
                        render_pass.set_push_constants(
                            wgpu::ShaderStages::VERTEX_FRAGMENT,
                            0,
                            bytemuck::cast_slice(&constants),
                        );
                        render_pass.draw(0..buffer.size() as u32 / 12, 0..1);
                    }
                }
            }

            {
                state
                    .visual_grid
                    .record_render_pass(&self.grid_pipeline.pipeline, &mut render_pass);
            }
        }
        encoder
    }
}

/// Rendering resources for loaded [`MicroSurface`].
pub struct MicroSurfaceState {
    /// Bind group containing global uniform buffer.
    globals_bind_group: wgpu::BindGroup,
    /// Bind group containing local uniform buffer.
    locals_bind_group: wgpu::BindGroup,
    /// Uniform buffer containing only view and projection matrices.
    global_uniform_buffer: wgpu::Buffer,
    /// Uniform buffer containing data subject to each loaded micro surface.
    local_uniform_buffer: wgpu::Buffer,
    /// Lookup table linking micro surface handle to its offset in the local
    /// uniform buffer.
    locals_lookup: Vec<Handle<MicroSurface>>,
}

impl MicroSurfaceState {
    pub const INITIAL_MICRO_SURFACE_COUNT: usize = 64;

    pub(crate) fn create_pipeline(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
    ) -> SurfacePipeline {
        let aligned_locals_size = MicroSurfaceUniforms::aligned_size(&ctx.device);
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("micro_surface_shader_module"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./assets/shaders/wgsl/micro_surface.wgsl").into(),
                ),
            });
        let globals_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("micro_surface_globals_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ViewProjUniform::SIZE_IN_BYTES as u64,
                            ),
                        },
                        count: None,
                    }],
                });

        let locals_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("micro_surface_locals_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(aligned_locals_size as u64),
                        },
                        count: None,
                    }],
                });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("micro_surface_pipeline_layout"),
                bind_group_layouts: &[&globals_bind_group_layout, &locals_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("micro_surface_render_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 3 * std::mem::size_of::<f32>() as u64,
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
                    polygon_mode: wgpu::PolygonMode::Line,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });
        let normals_pipeline = {
            let pipeline_layout =
                ctx.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("surface-normals-drawing-pipeline-layout"),
                        bind_group_layouts: &[
                            &globals_bind_group_layout,
                            &locals_bind_group_layout,
                        ],
                        push_constant_ranges: &[wgpu::PushConstantRange {
                            stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            range: 0..80,
                        }],
                    });
            let vert_state = wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_normals_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 3 * std::mem::size_of::<f32>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
            };
            let frag_state = wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_normals_main",
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
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("surface-normals-drawing-pipeline"),
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
                })
        };

        SurfacePipeline {
            pipeline,
            globals_bind_group_layout,
            locals_bind_group_layout,
            normals_pipeline,
        }
    }

    pub fn new(
        ctx: &GpuContext,
        globals_layout: &wgpu::BindGroupLayout,
        locals_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let global_uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro_surface_global_uniform_buffer"),
            size: std::mem::size_of::<ViewProjUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let aligned_locals_size = MicroSurfaceUniforms::aligned_size(&ctx.device);
        let local_uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro_surface_local_uniform_buffer"),
            size: aligned_locals_size as u64 * Self::INITIAL_MICRO_SURFACE_COUNT as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_globals_bind_group"),
            layout: globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        let locals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_locals_bind_group"),
            layout: locals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &local_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(aligned_locals_size as u64),
                }),
            }],
        });
        Self {
            globals_bind_group,
            locals_bind_group,
            global_uniform_buffer,
            local_uniform_buffer,
            locals_lookup: Default::default(),
        }
    }

    /// Update the list of surfaces.
    ///
    /// Updates the lookup table.
    pub fn update_surfaces_list<'a, S: Iterator<Item = &'a Handle<MicroSurface>>>(
        &mut self,
        surfs: S,
    ) {
        for hdl in surfs {
            if self.locals_lookup.contains(hdl) {
                continue;
            }
            self.locals_lookup.push(*hdl);
        }
    }

    pub fn update(
        &mut self,
        gpu: &GpuContext,
        view_proj: &ViewProjUniform,
        surfaces: &[(&Handle<MicroSurface>, &MicroSurfaceProp)],
    ) {
        // Update global uniform buffer.
        gpu.queue.write_buffer(
            &self.global_uniform_buffer,
            0,
            bytemuck::bytes_of(view_proj),
        );
        // Update per-surface uniform buffer.
        let aligned_size = MicroSurfaceUniforms::aligned_size(&gpu.device);
        for (hdl, state) in surfaces.iter() {
            let mut buf = [0.0; 20];
            let local_uniform_buf_index =
                self.locals_lookup.iter().position(|h| *h == **hdl).unwrap();
            buf[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
            buf[16..20].copy_from_slice(&[
                state.min + state.height_offset,
                state.max + state.height_offset,
                state.max - state.min,
                state.scale,
            ]);
            gpu.queue.write_buffer(
                &self.local_uniform_buffer,
                local_uniform_buf_index as u64 * aligned_size as u64,
                bytemuck::cast_slice(&buf),
            );
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MicroSurfaceUniforms {
    /// The model matrix.
    pub model: Mat4,
    /// Extra information: lowest, highest, span, scale.
    pub info: Vec4,
}

impl MicroSurfaceUniforms {
    /// Returns the size of the uniforms in bytes, aligned to the device's
    /// `min_uniform_buffer_offset_alignment`.
    pub fn aligned_size(device: &wgpu::Device) -> u32 {
        let alignment = device.limits().min_uniform_buffer_offset_alignment;
        let size = std::mem::size_of::<MicroSurfaceUniforms>() as u32;
        let remainder = size % alignment;
        if remainder == 0 {
            size
        } else {
            size + alignment - remainder
        }
    }
}

/// Surface viewer.
pub struct SurfaceViewer {
    /// Unique identifier across all widgets.
    uuid: Uuid,
    /// Size of the viewport.
    viewport_size: egui::Vec2,
    // /// Gizmo for navigating the scene.
    // navigator: NavigationGizmo,
    color_attachment_id: egui::TextureId,
    /// Shading mode.
    shading: ShadingMode,
    /// Overlay flags.
    overlay: OverlayFlags,
    event_loop: EventLoopProxy,
}

impl SurfaceViewer {
    /// Initial width of the viewer.
    const DEFAULT_WIDTH: u32 = 256;
    /// Initial height of the viewer.
    const DEFAULT_HEIGHT: u32 = 256;

    /// Creates a new [`SurfaceViewer`].
    ///
    /// The viewer's output is rendered to a texture, which can be accessed
    /// by egui through [`color_attachment_id`]. Resources needed for rendering
    /// are allocated by [`SurfaceViewerState`].
    pub fn new(gui: Arc<RwLock<GuiRenderer>>, event_loop: EventLoopProxy) -> Self {
        let color_attachment_id = gui.write().unwrap().pre_register_texture_id();
        Self {
            viewport_size: egui::Vec2::new(
                SurfaceViewer::DEFAULT_WIDTH as f32,
                SurfaceViewer::DEFAULT_HEIGHT as f32,
            ),
            // navigator: NavigationGizmo::new(GizmoOrientation::Global),
            color_attachment_id,
            shading: Default::default(),
            uuid: Uuid::new_v4(),
            event_loop,
            overlay: OverlayFlags::default(),
        }
    }

    /// Returns the texture ID of the viewer's output.
    pub fn color_attachment_id(&self) -> egui::TextureId { self.color_attachment_id }

    /// Resizes the viewport.
    pub fn resize_viewport(&mut self, new_size: egui::Vec2, scale_factor: Option<f32>) {
        let scale_factor = scale_factor.unwrap_or(1.0);
        if new_size == self.viewport_size || (new_size.x == 0.0 && new_size.y == 0.0) {
            return;
        }
        let width = (new_size.x * scale_factor) as u32;
        let height = (new_size.y * scale_factor) as u32;
        self.event_loop
            .send_event(VgonioEvent::SurfaceViewer(SurfaceViewerEvent::Resize {
                uuid: self.uuid,
                size: (width, height),
            }));
        self.viewport_size = new_size;
    }
}

impl Dockable for SurfaceViewer {
    fn kind(&self) -> WidgetKind { WidgetKind::SurfViewer }

    fn title(&self) -> WidgetText {
        format!("Surface Viewer - {}", &self.uuid.to_string().as_str()[..6]).into()
    }

    fn uuid(&self) -> Uuid { self.uuid }

    fn ui(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        let size = egui::Vec2::new(rect.width(), rect.height());

        let top = ui.clip_rect().top();
        let right = ui.clip_rect().right();
        let toolbar_width = ui.clip_rect().width() * 0.1;
        egui::Area::new(format!("surf_viewer_toolbar_{:?}", self.uuid))
            .anchor(
                egui::Align2::LEFT_TOP,
                egui::Vec2::new(right - toolbar_width, top),
            )
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    ui.menu_button("Overlay", |ui| {
                        let a = ui.checkbox(&mut self.overlay.normals, "Normals").changed();
                        let b = ui
                            .checkbox(&mut self.overlay.tangents, "Tangents")
                            .changed();
                        let c = ui
                            .checkbox(&mut self.overlay.bitangents, "Bitangents")
                            .changed();
                        let d = ui.checkbox(&mut self.overlay.uvs, "UVs").changed();
                        if a || b || c || d {
                            self.event_loop.send_event(VgonioEvent::SurfaceViewer(
                                SurfaceViewerEvent::UpdateOverlay {
                                    uuid: self.uuid,
                                    overlay: self.overlay,
                                },
                            ));
                        }
                    });
                    let shading = self.shading;
                    ui.menu_button("Shading", |ui| {
                        ui.selectable_value(&mut self.shading, ShadingMode::Wireframe, "Wireframe");
                        ui.selectable_value(&mut self.shading, ShadingMode::Flat, "Flat");
                        ui.selectable_value(&mut self.shading, ShadingMode::Smooth, "Smooth");
                    });
                    if shading != self.shading {
                        self.event_loop.send_event(VgonioEvent::SurfaceViewer(
                            SurfaceViewerEvent::UpdateShading {
                                uuid: self.uuid,
                                shading: self.shading,
                            },
                        ));
                    }
                });
            });

        self.resize_viewport(size, None);
        ui.image(egui::load::SizedTexture {
            id: self.color_attachment_id,
            size: self.viewport_size,
        });
    }
}

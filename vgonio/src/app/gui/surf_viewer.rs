use egui::{Ui, WidgetText};
use log::log;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use uuid::Uuid;

use vgcore::math::{Mat4, Vec3, Vec4};
use vgsurf::MicroSurface;

use crate::app::{
    cache::{Cache, Handle},
    gfx::{
        camera::{Camera, Projection, ProjectionKind, ViewProjUniform},
        GpuContext, Texture,
    },
    gui::{
        data::{MicroSurfaceProp, PropertyData},
        docking::{Dockable, WidgetKind},
        event::{EventLoopProxy, SurfaceViewerEvent, VgonioEvent},
        state::{camera::CameraState, InputState},
        theme::{ThemeKind, ThemeState, ThemeVisuals},
        visual_grid::VisualGridState,
    },
};

use super::{
    gizmo::NavigationGizmo,
    state::{DepthMap, GuiRenderer},
};

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
        }
    }

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
    ) {
        self.camera.update(&input, dt, ProjectionKind::Perspective);
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

/// State of multiple surface viewers.
pub struct SurfaceViewerStates {
    /// Output format of the surface viewer.
    format: wgpu::TextureFormat,
    /// Render pipeline for rendering micro surfaces.
    surf_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for surface rendering global uniform buffer.
    surf_globals_layout: wgpu::BindGroupLayout,
    /// Bind group layout for surface rendering local uniform buffer.
    surf_locals_layout: wgpu::BindGroupLayout,
    /// Render pipeline for rendering the visual grid.
    grid_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for visual grid rendering.
    grid_bind_group_layout: wgpu::BindGroupLayout,
    /// States of the micro surface rendering for each surface viewer.
    states: HashMap<Uuid, SurfaceViewerState>,
}

impl SurfaceViewerStates {
    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let (surf_pipeline, surf_globals_layout, surf_locals_layout) =
            MicroSurfaceState::create_pipeline(ctx, target_format);
        let (grid_pipeline, grid_bind_group_layout) =
            VisualGridState::create_pipeline(ctx, target_format);
        Self {
            format: target_format,
            surf_pipeline,
            surf_globals_layout,
            surf_locals_layout,
            grid_pipeline,
            grid_bind_group_layout,
            states: Default::default(),
        }
    }

    pub fn update_surfaces_list(&mut self, surfaces: &[Handle<MicroSurface>]) {
        for (_, state) in self.states.iter_mut() {
            state.surface.update_surfaces_list(surfaces);
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
        self.states.insert(
            viewer,
            SurfaceViewerState::new(
                gpu,
                gui,
                self.format,
                tex_id,
                &self.surf_globals_layout,
                &self.surf_locals_layout,
                &self.grid_bind_group_layout,
            ),
        );
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

    // TODO: only update focused viewer's camera
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        input: &InputState,
        dt: std::time::Duration,
        theme: &ThemeState,
        cache: &Cache,
        encoder: &mut wgpu::CommandEncoder,
        surfaces: &[(&Handle<MicroSurface>, &MicroSurfaceProp)],
    ) {
        for (_, state) in &mut self.states {
            state.update(surfaces, input, dt, theme, gpu);
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
                render_pass.set_pipeline(&self.surf_pipeline);
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
                }
            }
            {
                state
                    .visual_grid
                    .render(&self.grid_pipeline, &mut render_pass);
            }
        }
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

    pub fn create_pipeline(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
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
        (
            pipeline,
            globals_bind_group_layout,
            locals_bind_group_layout,
        )
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
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        let locals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_locals_bind_group"),
            layout: &locals_layout,
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
    pub fn update_surfaces_list(&mut self, surfs: &[Handle<MicroSurface>]) {
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
    /// Cache for all kinds of resources.
    cache: Arc<RwLock<Cache>>,
    // /// Debug drawing state.
    // debug_draw_state: DebugDrawState,
    /// Size of the viewport.
    viewport_size: egui::Vec2,
    // /// Gizmo for navigating the scene.
    // navigator: NavigationGizmo,
    output_format: wgpu::TextureFormat,
    proj_view_model: Mat4,
    focused: bool,
    color_attachment_id: egui::TextureId,
    event_loop: EventLoopProxy,

    prop_data: Arc<RwLock<PropertyData>>, // TODO: remove
}

impl SurfaceViewer {
    const DEFAULT_WIDTH: u32 = 256;
    const DEFAULT_HEIGHT: u32 = 256;

    pub fn new(
        gui: Arc<RwLock<GuiRenderer>>,
        format: wgpu::TextureFormat,
        cache: Arc<RwLock<Cache>>,
        event_loop: EventLoopProxy,
        prop_data: Arc<RwLock<PropertyData>>,
    ) -> Self {
        let color_attachment_id = gui.write().unwrap().pre_register_texture_id();
        Self {
            cache,
            // debug_draw_state,
            viewport_size: egui::Vec2::new(
                SurfaceViewer::DEFAULT_WIDTH as f32,
                SurfaceViewer::DEFAULT_HEIGHT as f32,
            ),
            // navigator: NavigationGizmo::new(GizmoOrientation::Global),
            output_format: format,
            color_attachment_id,
            proj_view_model: Mat4::IDENTITY,
            focused: false,
            uuid: Uuid::new_v4(),
            prop_data,
            event_loop,
        }
    }

    pub fn color_attachment_id(&self) -> egui::TextureId { self.color_attachment_id }

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
            }))
            .unwrap();
        self.viewport_size = new_size;
    }

    #[deprecated]
    fn render(&mut self, size: egui::Vec2, scale_factor: Option<f32>) {
        // Resize if needed
        self.resize_viewport(size, scale_factor);
        // TODO: update camera
        // self.camera.uniform.update(
        //     &self.camera.camera,
        //     &self.camera.projection,
        //     ProjectionKind::Perspective,
        // );
        // self.visual_grid_state.update_uniforms(
        //     &self.gpu,
        //     &self.camera.uniform.view_proj,
        //     &self.camera.uniform.view_proj_inv,
        //     wgpu::Color {
        //         r: 0.4,
        //         g: 0.4,
        //         b: 0.4,
        //         a: 1.0,
        //     },
        //     ThemeKind::Dark,
        // );
        // self.navigator.update_matrices(
        //     Mat4::IDENTITY,
        //     Mat4::look_at_rh(self.camera.camera.eye, Vec3::ZERO,
        // self.camera.camera.up),     Mat4::orthographic_rh(-1.0, 1.0, -1.0,
        // 1.0, 0.1, 100.0), );
        // let view_proj = self.camera.uniform.view_proj;
        // let prop_data = self.prop_data.read().unwrap();
        //     let visible_surfaces = prop_data.visible_surfaces();
        //     if !visible_surfaces.is_empty() {
        //         self.gpu.queue.write_buffer(
        //             &self.surf_state.global_uniform_buffer,
        //             0,
        //             bytemuck::bytes_of(&view_proj),
        //         );
        //         // Update per-surface uniform buffer.
        //         let aligned_size =
        // MicroSurfaceUniforms::aligned_size(&self.gpu.device);
        //         for (hdl, prop) in visible_surfaces.iter() {
        //             let local_uniform_buf_index = if let Some(idx) =
        //                 self.surf_state.locals_lookup.iter().position(|h| h
        // == *hdl)             {
        //                 idx
        //             } else {
        //                 self.surf_state.locals_lookup.push(**hdl);
        //                 self.surf_state.locals_lookup.len() - 1
        //             };

        //             let mut buf = [0.0; 20];
        //
        // buf[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
        //             buf[16..20].copy_from_slice(&[
        //                 prop.min + prop.height_offset,
        //                 prop.max + prop.height_offset,
        //                 prop.max - prop.min,
        //                 prop.scale,
        //             ]);
        //             self.gpu.queue.write_buffer(
        //                 &self.surf_state.local_uniform_buffer,
        //                 local_uniform_buf_index as u64 * aligned_size as u64,
        //                 bytemuck::cast_slice(&buf),
        //             );
        //         }
        //     }

        //     let cache = self.cache.read().unwrap();
        //     // Update rendered texture.
        //     let mut encoder = self
        //         .gpu
        //         .device
        //         .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        //             label: Some("surf_viewer_update"),
        //         });
        //     {
        //         let mut render_pass =
        // encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //             label: Some("surf_viewer_update"),
        //             color_attachments: &[Some(wgpu::RenderPassColorAttachment
        // {                 view: &self.color_attachment.view,
        //                 resolve_target: None,
        //                 ops: wgpu::Operations {
        //                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        //                     store: true,
        //                 },
        //             })],
        //             depth_stencil_attachment:
        // Some(wgpu::RenderPassDepthStencilAttachment {
        // view: &self.depth_map.depth_attachment.view,
        // depth_ops: Some(wgpu::Operations {                     load:
        // wgpu::LoadOp::Clear(1.0),                     store: true,
        //                 }),
        //                 stencil_ops: None,
        //             }),
        //         });

        //         let aligned_micro_surface_uniform_size =
        //             MicroSurfaceUniforms::aligned_size(&self.gpu.device);

        //         if !visible_surfaces.is_empty() {
        //             render_pass.set_pipeline(&self.surf_state.pipeline);
        //             render_pass.set_bind_group(0,
        // &self.surf_state.globals_bind_group, &[]);

        //             for (hdl, _) in visible_surfaces.iter() {
        //                 let renderable =
        // cache.get_micro_surface_renderable_mesh_by_surface_id(**hdl);
        //                 if renderable.is_none() {
        //                     log::debug!(
        //                         "Failed to get renderable mesh for surface
        // {:?}, skipping.",
        //                         hdl
        //                     );
        //                     continue;
        //                 }
        //                 let buf_index = self
        //                     .surf_state
        //                     .locals_lookup
        //                     .iter()
        //                     .position(|x| &x == hdl)
        //                     .unwrap();
        //                 let renderable = renderable.unwrap();
        //                 render_pass.set_bind_group(
        //                     1,
        //                     &self.surf_state.locals_bind_group,
        //                     &[buf_index as u32 *
        // aligned_micro_surface_uniform_size],                 );
        //                 render_pass.set_vertex_buffer(0,
        // renderable.vertex_buffer.slice(..));
        // render_pass.set_index_buffer(
        // renderable.index_buffer.slice(..),
        // renderable.index_format,                 );
        //                 render_pass.draw_indexed(0..renderable.indices_count,
        // 0, 0..1);             }
        //         }

        //         self.visual_grid_state.render(&mut render_pass);

        //         // // Draw visual grid.
        //         //
        // render_pass.set_pipeline(&self.visual_grid_state.pipeline);
        //         // render_pass.set_bind_group(0,
        // &self.visual_grid_state.bind_group,         // &[]);
        // render_pass.draw(0..6, 0..1);     }
        //     self.gpu.queue.submit(Some(encoder.finish()));
    }
}

impl Dockable for SurfaceViewer {
    fn kind(&self) -> WidgetKind { WidgetKind::SurfViewer }

    fn title(&self) -> WidgetText { "Surface Viewer".into() }

    fn uuid(&self) -> Uuid { self.uuid }

    fn ui(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        let size = egui::Vec2::new(rect.width(), rect.height());
        // Resize if needed.
        self.resize_viewport(size, None);
        ui.image(self.color_attachment_id, self.viewport_size);
    }
}

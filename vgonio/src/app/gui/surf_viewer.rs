use crate::{
    app::{
        cache::{Cache, Handle},
        gfx::{
            camera::{Camera, Projection, ProjectionKind, ViewProjUniform},
            GpuContext, Texture,
        },
        gui::{
            gizmo::NavigationGizmo,
            state::{camera::CameraState, DebugDrawState, DepthMap, GuiRenderer, InputState},
            ui::Dockable,
            MicroSurfaceUniforms, VgonioEventLoop, VisualGridState,
        },
    },
    error::RuntimeError,
};
use chrono::Duration;
use egui::{PointerButton, Ui, WidgetText, WidgetType::Label};
use egui_gizmo::GizmoOrientation;
use std::sync::{Arc, RwLock};
use vgcore::math::Vec3;
use vgsurf::MicroSurface;
use winit::dpi::PhysicalSize;

// TODO: remove crate public visibility
/// Rendering resources for loaded [`MicroSurface`].
pub struct MicroSurfaceState {
    /// Render pipeline for rendering micro surfaces.
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// Bind group containing global uniform buffer.
    pub(crate) globals_bind_group: wgpu::BindGroup,
    /// Bind group containing local uniform buffer.
    pub(crate) locals_bind_group: wgpu::BindGroup,
    /// Uniform buffer containing only view and projection matrices.
    pub(crate) global_uniform_buffer: wgpu::Buffer,
    /// Uniform buffer containing data subject to each loaded micro surface.
    pub(crate) local_uniform_buffer: wgpu::Buffer,
    /// Lookup table linking [`MicroSurface`] to its offset in the local uniform
    /// buffer.
    pub(crate) locals_lookup: Vec<Handle<MicroSurface>>,
}

impl MicroSurfaceState {
    pub const INITIAL_MICRO_SURFACE_COUNT: usize = 64;

    pub fn new(ctx: &GpuContext, target_format: wgpu::TextureFormat) -> Self {
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("micro_surface_shader_module"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./assets/shaders/wgsl/micro_surface.wgsl").into(),
                ),
            });
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

        let globals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_globals_bind_group"),
            layout: &globals_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: global_uniform_buffer.as_entire_binding(),
            }],
        });

        let locals_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("micro_surface_locals_bind_group"),
            layout: &locals_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &local_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(aligned_locals_size as u64),
                }),
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
        Self {
            pipeline,
            globals_bind_group,
            locals_bind_group,
            global_uniform_buffer,
            local_uniform_buffer,
            locals_lookup: Default::default(),
        }
    }

    /// Only updates the lookup table. The actual data is updated in the
    /// [`VgonioGuiApp::update`] method.
    pub fn update_locals_lookup(&mut self, surfs: &[Handle<MicroSurface>]) {
        for hdl in surfs {
            if self.locals_lookup.contains(hdl) {
                continue;
            }
            self.locals_lookup.push(*hdl);
        }
    }
}

/// Surface viewer.
pub struct SurfViewer {
    /// GPU context.
    gpu: Arc<GpuContext>,
    gui: Arc<RwLock<GuiRenderer>>,
    /// State of the camera.
    camera: CameraState,
    /// State of the visual grid rendering.
    visual_grid_state: VisualGridState,
    /// Cache for all kinds of resources.
    cache: Arc<RwLock<Cache>>,
    /// Micro-surface rendering state.
    surf_state: MicroSurfaceState,
    /// Depth texture.
    depth_map: DepthMap,
    /// Debug drawing state.
    debug_draw_state: DebugDrawState,
    /// Size of the viewport.
    viewport_size: egui::Vec2,
    /// Gizmo for navigating the scene.
    navigator: NavigationGizmo,
    output_format: wgpu::TextureFormat,
    /// Color attachment.
    color_attachment: Texture,
    color_attachment_id: egui::TextureId,
    // depth_attachment: Texture,
    // depth_attachment_id: egui::TextureId,
    id_counter: u32,
}

impl SurfViewer {
    pub fn new(
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        cache: Arc<RwLock<Cache>>,
        event_loop: VgonioEventLoop,
        id_counter: u32,
    ) -> Self {
        let surf_state = MicroSurfaceState::new(&gpu, format);
        let depth_map = DepthMap::new(&gpu, width, height);
        let debug_draw_state = DebugDrawState::new(&gpu, format, event_loop, cache.clone());
        let camera = {
            let camera = Camera::new(Vec3::new(0.0, 4.0, 10.0), Vec3::ZERO, Vec3::Y);
            let projection = Projection::new(0.1, 100.0, 75.0f32.to_radians(), width, height);
            CameraState::new(camera, projection, ProjectionKind::Perspective)
        };
        let visual_grid_state = VisualGridState::new(&gpu, format);
        // TODO: improve
        let sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampling-debugger-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        let color_attachment = Texture::new(
            &gpu.device,
            &wgpu::TextureDescriptor {
                label: Some("surf_viewer_color_attachment"),
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
        let color_attachment_id = gui.write().unwrap().register_native_texture(
            &gpu.device,
            &color_attachment.view,
            wgpu::FilterMode::Linear,
        );
        Self {
            gpu,
            gui,
            camera,
            visual_grid_state,
            cache,
            surf_state,
            depth_map,
            debug_draw_state,
            viewport_size: egui::Vec2::new(width as f32, height as f32),
            navigator: NavigationGizmo::new(GizmoOrientation::Global),
            output_format: format,
            color_attachment,
            color_attachment_id,
            id_counter,
        }
    }

    pub fn resize(&mut self, new_size: egui::Vec2, scale_factor: Option<f32>) {
        let scale_factor = scale_factor.unwrap_or(1.0);
        if new_size == self.viewport_size || (new_size.x == 0.0 && new_size.y == 0.0) {
            return;
        }
        println!("resize: {:?}", new_size);
        let width = (new_size.x * scale_factor) as u32;
        let height = (new_size.y * scale_factor) as u32;
        let sampler = Arc::new(self.gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampling-debugger-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        self.color_attachment = Texture::new(
            &self.gpu.device,
            &wgpu::TextureDescriptor {
                label: Some("surf_viewer_color_attachment"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.output_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Some(sampler),
        );
        self.gui
            .write()
            .unwrap()
            .update_egui_texture_from_wgpu_texture(
                &self.gpu.device,
                &self.color_attachment.view,
                wgpu::FilterMode::Linear,
                self.color_attachment_id,
            );
        self.depth_map.resize(&self.gpu, width, height);
        self.camera.projection.resize(width, height);
        self.viewport_size = new_size;
    }

    // pub fn update(&mut self, input: &InputState, dt: Duration) -> Result<(),
    // RuntimeError> {     self.camera.update(input, dt,
    // ProjectionKind::Perspective); }
}

impl Dockable for SurfViewer {
    fn title(&self) -> WidgetText { WidgetText::from("Surface View") }

    fn ui(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        self.resize(rect.size(), None);
        let response = egui::Area::new("surf_viewer")
            .show(ui.ctx(), |ui| {
                ui.push_id(format!("surf_viewer_{}", self.id_counter), |ui| {
                    ui.image(self.color_attachment_id, rect.size());
                });
            })
            .response;

        if response.dragged_by(PointerButton::Primary) {
            println!("dragged");
        }
    }
}

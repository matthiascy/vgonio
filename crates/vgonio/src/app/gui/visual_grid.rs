use base::math::{Mat4, Vec3, IDENTITY_MAT4};
use wgut::{
    camera::ViewProjUniform, context::GpuContext, render_pass::DEFAULT_BIND_GROUP_LAYOUT_DESC,
    texture::Texture,
};

use super::theme::ThemeKind;

// TODO: unify VisualGrid and VisualGridState

pub struct VisualGrid {
    pub transform: Mat4,
    /// Extent of grid in X, Y plane
    pub grid_size: f32,
    /// Minimum size of one cell
    pub cell_size: f32,
    /// sRGB color and alpha of thin lines
    pub thin_lines_color: Vec3,
    /// sRGB color and alpha of thick lines (every tenth line)
    pub thick_lines_color: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisualGridUniforms {
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub view_inv: [f32; 16],
    pub proj_inv: [f32; 16],
    pub grid_line_color: [f32; 4],
}

impl Default for VisualGridUniforms {
    fn default() -> Self {
        Self {
            view: IDENTITY_MAT4,
            proj: IDENTITY_MAT4,
            view_inv: IDENTITY_MAT4,
            proj_inv: IDENTITY_MAT4,
            grid_line_color: [0.6, 0.6, 0.6, 1.0],
        }
    }
}

/// Visual grid rendering.
pub struct VisualGridState {
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    pub(crate) visible: bool,
}

pub(crate) struct VisualGridPipeline {
    /// Render pipeline for rendering the visual grid.
    pub pipeline: wgpu::RenderPipeline,
    /// Bind group layout for visual grid rendering.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl VisualGridState {
    pub(crate) fn create_pipeline(
        ctx: &GpuContext,
        target_format: wgpu::TextureFormat,
    ) -> VisualGridPipeline {
        let vert_shader = ctx
            .device
            .create_shader_module(wgpu::include_spirv!(concat!(
                env!("OUT_DIR"),
                "/visual_grid.vert.spv"
            )));
        let frag_shader = ctx
            .device
            .create_shader_module(wgpu::include_spirv!(concat!(
                env!("OUT_DIR"),
                "/visual_grid.frag.spv"
            )));
        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&DEFAULT_BIND_GROUP_LAYOUT_DESC);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visual_grid_render_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visual_grid_render_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vert_shader,
                    entry_point: "main",
                    buffers: &[],
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
                fragment: Some(wgpu::FragmentState {
                    module: &frag_shader,
                    entry_point: "main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });
        VisualGridPipeline {
            pipeline,
            bind_group_layout,
        }
    }
}

impl VisualGridState {
    pub fn new(ctx: &GpuContext, bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        use wgpu::util::DeviceExt;
        let uniform_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visual_grid_uniform_buffer"),
                contents: bytemuck::bytes_of(&VisualGridUniforms::default()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visual_grid_bind_group"),
            layout: bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        Self {
            bind_group,
            uniform_buffer,
            visible: true,
        }
    }

    pub fn update(
        &self,
        ctx: &GpuContext,
        view_proj: &ViewProjUniform,
        view_proj_inv: &ViewProjUniform,
        color: wgpu::Color,
        theme_kind: ThemeKind,
    ) {
        let alpha = match theme_kind {
            ThemeKind::Light => 0.0,
            ThemeKind::Dark => 1.0,
        };
        ctx.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&VisualGridUniforms {
                view: view_proj.view.to_cols_array(),
                proj: view_proj.proj.to_cols_array(),
                view_inv: view_proj_inv.view.to_cols_array(),
                proj_inv: view_proj_inv.proj.to_cols_array(),
                grid_line_color: [color.r as f32, color.g as f32, color.b as f32, alpha],
            }),
        );
    }

    /// Render the visual grid.
    pub fn record_render_pass<'a>(
        &'a self,
        pipeline: &'a wgpu::RenderPipeline,
        pass: &mut wgpu::RenderPass<'a>,
    ) {
        if !self.visible {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    pub fn visible(&self) -> bool { self.visible }
}

use crate::{context::GpuContext, vertex::VertexLayout};
use base::{
    math::{Aabb, Vec3},
    Asset,
};
use surf::{HeightOffset, MicroSurface, MicroSurfaceMesh, TriangulationPattern};
use uuid::Uuid;

/// A mesh of triangles that can be rendered with a [`wgpu::RenderPipeline`].
#[derive(Debug)]
pub struct RenderableMesh {
    pub uuid: Uuid,
    pub msurf: Option<Uuid>,
    pub vertices_count: u32,
    pub indices_count: u32,
    pub extent: Aabb,
    pub vertex_layout: VertexLayout,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub normal_buffer: Option<wgpu::Buffer>,
    pub index_format: wgpu::IndexFormat,
    pub topology: wgpu::PrimitiveTopology,
}

impl Asset for RenderableMesh {}

// TODO: create a separate method to extract face normals of an heightfield
impl RenderableMesh {
    pub fn from_micro_surface_with_id(
        ctx: &GpuContext,
        surf: &MicroSurface,
        id: Uuid,
        offset: HeightOffset,
        pattern: TriangulationPattern,
    ) -> Self {
        let mesh = surf.as_micro_surface_mesh(offset, pattern);
        Self::from_micro_surface_mesh_with_id(ctx, &mesh, id)
    }

    /// Creates a new [`RenderableMesh`] directly from a [`MicroSurface`].
    pub fn from_micro_surface(
        ctx: &GpuContext,
        surf: &MicroSurface,
        offset: HeightOffset,
        pattern: TriangulationPattern,
    ) -> Self {
        Self::from_micro_surface_with_id(ctx, surf, Uuid::new_v4(), offset, pattern)
    }

    pub fn from_micro_surface_mesh_with_id(
        ctx: &GpuContext,
        mesh: &MicroSurfaceMesh,
        id: Uuid,
    ) -> Self {
        use wgpu::util::DeviceExt;
        log::debug!(
            "Creating mesh view with vertex count: {}, expected size: {} bytes",
            mesh.num_verts,
            mesh.verts.len() * 3 * 4
        );
        let vertex_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_view_vertex_buffer"),
                contents: bytemuck::cast_slice(&mesh.verts),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh_view_index_buffer"),
                contents: bytemuck::cast_slice(&mesh.facets),
                usage: wgpu::BufferUsages::INDEX,
            });

        let vertex_layout = VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None);

        let normals = mesh
            .facet_normals
            .iter()
            .zip(mesh.facets.chunks(3))
            .flat_map(|(n, f)| {
                let center = f
                    .iter()
                    .fold(Vec3::ZERO, |acc, v| acc + mesh.verts[*v as usize] / 3.0);
                [center, center + *n * 0.35]
            })
            .collect::<Vec<_>>();

        let normal_buffer = Some(ctx.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("mesh_view_normal_buffer"),
                contents: bytemuck::cast_slice(&normals),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));

        log::debug!(
            "TriangleMesh --> MeshView, num verts: {}, num faces: {}, num indices: {}",
            mesh.num_verts,
            mesh.num_facets,
            mesh.facets.len()
        );

        Self {
            uuid: id,
            msurf: Some(mesh.msurf),
            vertices_count: mesh.num_verts as u32,
            indices_count: mesh.facets.len() as u32,
            extent: mesh.bounds,
            vertex_layout,
            vertex_buffer,
            index_buffer,
            normal_buffer,
            index_format: wgpu::IndexFormat::Uint32,
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }

    pub fn from_micro_surface_mesh(ctx: &GpuContext, mesh: &MicroSurfaceMesh) -> Self {
        Self::from_micro_surface_mesh_with_id(ctx, mesh, Uuid::new_v4())
    }
}
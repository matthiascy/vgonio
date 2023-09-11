use crate::app::{cache::Asset, gfx::VertexLayout};
use bytemuck::{Pod, Zeroable};
use std::ops::Index;
use uuid::Uuid;
use vgcore::math::Aabb;
use vgsurf::{HeightOffset, MicroSurface, MicroSurfaceMesh, TriangulationPattern};

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
    pub index_format: wgpu::IndexFormat,
    pub topology: wgpu::PrimitiveTopology,
}

impl Asset for RenderableMesh {}

// TODO: create a separate method to extract face normals of an heightfield
impl RenderableMesh {
    pub fn from_micro_surface_with_id(
        device: &wgpu::Device,
        surf: &MicroSurface,
        id: Uuid,
        offset: HeightOffset,
        pattern: TriangulationPattern,
    ) -> Self {
        use wgpu::util::DeviceExt;
        // Number of triangles = 2 * rows * cols
        let (cols, rows) = (surf.cols, surf.rows);
        let (positions, extent) = surf.generate_vertices(offset.eval(surf.min, surf.max));
        let vertices_count = positions.len();
        let indices_count = 2 * (rows - 1) * (cols - 1) * 3;
        let indices: Vec<u32> = vgsurf::regular_grid_triangulation(rows, cols, pattern);
        debug_assert_eq!(indices.len(), indices_count);
        log::debug!(
            "Heightfield--> MeshView, num verts: {}, num faces: {}, num indices: {}",
            vertices_count,
            indices_count / 3,
            indices.len()
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_vertex_buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let vertex_layout = VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None);

        Self {
            uuid: id,
            msurf: Some(surf.uuid),
            vertices_count: vertices_count as u32,
            indices_count: indices_count as u32,
            extent,
            vertex_layout,
            vertex_buffer,
            index_buffer,
            index_format: wgpu::IndexFormat::Uint32,
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }

    /// Creates a new [`RenderableMesh`] directly from a [`MicroSurface`].
    pub fn from_micro_surface(
        device: &wgpu::Device,
        surf: &MicroSurface,
        offset: HeightOffset,
        pattern: TriangulationPattern,
    ) -> Self {
        Self::from_micro_surface_with_id(device, surf, Uuid::new_v4(), offset, pattern)
    }

    pub fn from_micro_surface_mesh_with_id(
        device: &wgpu::Device,
        mesh: &MicroSurfaceMesh,
        id: Uuid,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_vertex_buffer"),
            contents: bytemuck::cast_slice(&mesh.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_index_buffer"),
            contents: bytemuck::cast_slice(&mesh.facets),
            usage: wgpu::BufferUsages::INDEX,
        });

        let vertex_layout = VertexLayout::new(&[wgpu::VertexFormat::Float32x3], None);

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
            index_format: wgpu::IndexFormat::Uint32,
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }

    pub fn from_micro_surface_mesh(device: &wgpu::Device, mesh: &MicroSurfaceMesh) -> Self {
        Self::from_micro_surface_mesh_with_id(device, mesh, Uuid::new_v4())
    }
}

impl RenderableMesh {
    pub fn new<V: Pod + Zeroable + Index<usize, Output = f32>, I: Pod + Zeroable>(
        device: &wgpu::Device,
        layout: VertexLayout,
        vertices: &[V],
        indices: &[I],
        topology: wgpu::PrimitiveTopology,
        index_format: wgpu::IndexFormat,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let mut extent = Aabb::default();
        for v in vertices.iter() {
            for k in 0..3 {
                if v[k] > extent.max[k] {
                    extent.max[k] = v[k];
                }
                if v[k] < extent.min[k] {
                    extent.min[k] = v[k];
                }
            }
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_index_buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_index_buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            uuid: Uuid::new_v4(),
            msurf: None,
            vertices_count: vertices.len() as _,
            indices_count: indices.len() as _,
            extent,
            vertex_layout: layout,
            vertex_buffer: vb,
            index_buffer: ib,
            index_format,
            topology,
        }
    }
}

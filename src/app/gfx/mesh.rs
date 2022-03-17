use crate::app::gfx::VertexLayout;
use crate::isect::Aabb;
use wgpu::util::DeviceExt;

/// Surface mesh which contains only triangles.
#[derive(Debug)]
pub struct MeshView {
    pub vertices_count: u32,
    pub indices_count: u32,
    pub vertex_layout: VertexLayout,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub topology: wgpu::PrimitiveTopology,
}

impl MeshView {
    pub fn new<V: bytemuck::Pod + bytemuck::Zeroable>(
        device: &wgpu::Device,
        layout: VertexLayout,
        vertices: &[V],
        indices: &[u32],
        topology: wgpu::PrimitiveTopology,
    ) -> Self {
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
            vertices_count: vertices.len() as _,
            indices_count: indices.len() as _,
            vertex_layout: layout,
            vertex_buffer: vb,
            index_buffer: ib,
            topology,
        }
    }
}

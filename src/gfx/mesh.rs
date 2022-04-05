use crate::gfx::VertexLayout;
use crate::htfld::{regular_triangulation, Heightfield};
use crate::isect::Aabb;
use bytemuck::{Pod, Zeroable};
use std::ops::Index;
use wgpu::util::DeviceExt;
use wgpu::{PrimitiveTopology, VertexFormat};

/// Surface mesh which contains only triangles.
#[derive(Debug)]
pub struct MeshView {
    pub vertices_count: u32,
    pub indices_count: u32,
    pub extent: Aabb,
    pub vertex_layout: VertexLayout,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub topology: wgpu::PrimitiveTopology,
}

// TODO: create a separate method to extract face normals of an heightfield
impl MeshView {
    pub fn from_height_field(device: &wgpu::Device, hf: &Heightfield) -> Self {
        // Number of triangles = 2 * rows * cols
        let (rows, cols) = (hf.cols, hf.rows);
        let (positions, extent) = hf.generate_vertices();
        let vertices_count = positions.len();
        let indices_count = 2 * (rows - 1) * (cols - 1) * 3;

        let indices: Vec<u32> = regular_triangulation(&positions, rows, cols);

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

        let vertex_layout = VertexLayout::new(&[VertexFormat::Float32x3], None);

        Self {
            vertices_count: vertices_count as u32,
            indices_count: indices_count as u32,
            extent,
            vertex_layout,
            vertex_buffer,
            index_buffer,
            index_format: wgpu::IndexFormat::Uint32,
            topology: PrimitiveTopology::TriangleList,
        }
    }
}

impl MeshView {
    pub fn new<V: Pod + Zeroable + Index<usize, Output = f32>, I: Pod + Zeroable>(
        device: &wgpu::Device,
        layout: VertexLayout,
        vertices: &[V],
        indices: &[I],
        topology: wgpu::PrimitiveTopology,
        index_format: wgpu::IndexFormat,
    ) -> Self {
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

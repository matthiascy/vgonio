use crate::gfx::VertexLayout;
use crate::htfld::Heightfield;
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

// TODO: create a separate method to extract face normals of an heightfield

impl MeshView {
    pub fn from_height_field(device: &wgpu::Device, hf: &Heightfield) -> Self {
        // Number of triangles = 2 * rows * cols
        let (rows, cols) = (hf.cols, hf.rows);
        let (positions, extent) = hf.generate_vertices();
        let vertices_count = positions.len();
        let indices_count = 2 * (rows - 1) * (cols - 1) * 3;

        let mut indices: Vec<u32> = vec![];
        indices.resize(indices_count, 0);
        let mut tri = 0;
        for i in 0..rows * cols {
            let row = i / rows;
            let col = i % rows;

            // last row
            if row == cols - 1 {
                continue;
            }

            if col == 0 {
                indices[tri] = i as u32;
                indices[tri + 1] = (i + 1) as u32;
                indices[tri + 2] = (i + rows) as u32;
                tri += 3;
            } else if col == rows - 1 {
                indices[tri] = i as u32;
                indices[tri + 1] = (i + rows) as u32;
                indices[tri + 2] = (i + rows - 1) as u32;
                tri += 3;
            } else {
                indices[tri] = i as u32;
                indices[tri + 1] = (i + rows) as u32;
                indices[tri + 2] = (i + rows - 1) as u32;
                indices[tri + 3] = i as u32;
                indices[tri + 4] = (i + 1) as u32;
                indices[tri + 5] = (i + rows) as u32;
                tri += 6;
            }
        }

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

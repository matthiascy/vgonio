pub mod desc;
pub mod ior;
mod occlusion;

pub use occlusion::*;
use std::str::FromStr;

use crate::gfx::camera::{Projection, ProjectionKind};
use crate::htfld::{regular_triangulation, Heightfield};
use crate::isect::Aabb;
use crate::Error;
use glam::Vec3;
use wgpu::util::DeviceExt;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum Medium {
    Air,
    Vacuum,
    Aluminium,
    Copper,
}

impl FromStr for Medium {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "air" => Ok(Self::Air),
            "vacuum" => Ok(Self::Vacuum),
            "al" => Ok(Self::Aluminium),
            "cu" => Ok(Self::Copper),
            &_ => Err(Error::Any("unknown medium".to_string())),
        }
    }
}

/// Light source used for acquisition of shadowing and masking function.
pub struct LightSource {
    pub pos: Vec3,
    pub proj: Projection,
    pub proj_kind: ProjectionKind,
}

/// Light space matrix used for generation of depth map.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightSourceRaw([f32; 16]);

impl LightSource {
    pub fn to_raw(&self) -> LightSourceRaw {
        let forward = -self.pos;
        let up = if forward == -Vec3::Y {
            Vec3::new(1.0, 1.0, 0.0).normalize()
        } else {
            Vec3::Y
        };
        let view = glam::Mat4::look_at_rh(self.pos, Vec3::ZERO, up);
        let proj = self.proj.matrix(self.proj_kind);
        LightSourceRaw((proj * view).to_cols_array())
    }
}

/// Micro-surface mesh used to measure geometric masking/shadowing function.
/// Position is the only attribute that a vertex possesses. See
/// [`Self::VERTEX_FORMAT`]. Indices is stored as `u32`. See
/// [`Self::INDEX_FORMAT`]. The mesh is constructed from a heightfield.
///
/// Indices are generated in the following order: azimuthal angle first then
/// polar angle.
pub struct MicroSurfaceView {
    /// Dimension of the micro-surface.
    pub extent: Aabb,

    /// Buffer stores all the vertices.
    pub vertex_buffer: wgpu::Buffer,

    pub facets: Vec<[u32; 3]>,

    pub facet_normals: Vec<Vec3>,
}

impl MicroSurfaceView {
    pub const VERTEX_FORMAT: wgpu::VertexFormat = wgpu::VertexFormat::Float32;

    pub const VERTEX_BUFFER_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: Self::VERTEX_FORMAT.size(),
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: Self::VERTEX_FORMAT,
            offset: 0,
            shader_location: 0,
        }],
    };

    pub const INDEX_FORMAT: wgpu::IndexFormat = wgpu::IndexFormat::Uint32;
}

impl MicroSurfaceView {
    pub fn from_height_field(device: &wgpu::Device, hf: &Heightfield) -> Self {
        let (rows, cols) = (hf.cols, hf.rows);
        let (positions, extent) = hf.generate_vertices();
        let indices: Vec<u32> = regular_triangulation(&positions, rows, cols);
        let facets = (0..indices.len())
            .step_by(3)
            .map(|i| [indices[i], indices[i + 1], indices[i + 2]])
            .collect();
        let facets_count = indices.len() / 3;

        // Generate triangles' normals
        let mut normals: Vec<Vec3> = vec![];
        normals.resize(facets_count, Vec3::ZERO);

        for i in 0..facets_count {
            let i0 = indices[3 * i] as usize;
            let i1 = indices[3 * i + 1] as usize;
            let i2 = indices[3 * i + 2] as usize;

            let v0 = positions[i0];
            let v1 = positions[i1];
            let v2 = positions[i2];

            let n = (v1 - v0).cross(v2 - v0).normalize();
            normals[i] = n;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_vertex_buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            extent,
            vertex_buffer,
            facets,
            facet_normals: normals,
        }
    }
}

fn measure_micro_surface_geometric_term(
    device: wgpu::Device,
    queue: wgpu::Queue,
    view: &MicroSurfaceView,
) {
    unimplemented!()
}

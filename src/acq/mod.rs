//! Acquisition related.

pub mod bsdf;
mod collector;
pub mod desc;
mod embree_rt;
mod emitter;
pub mod fresnel;
mod grid_rt;
pub mod ior;
pub mod ndf;
mod occlusion;
pub mod scattering;
pub mod util;
mod si_units;

pub use collector::{Collector, Patch};
pub use embree_rt::EmbreeRayTracing;
pub use emitter::Emitter;
pub use grid_rt::GridRayTracing;
pub use si_units::*;

pub use occlusion::*;
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::{
    gfx::camera::{Projection, ProjectionKind},
    htfld::{regular_triangulation, Heightfield},
    isect::Aabb,
    Error,
};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Representation of a ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// The origin of the ray.
    pub o: Vec3,

    /// The direction of the ray.
    pub d: Vec3,
}

impl Ray {
    /// Create a new ray (direction will be normalised).
    pub fn new(o: Vec3, d: Vec3) -> Self {
        let d = d.normalize();
        // let inv_dir_z = 1.0 / d.z;
        // let kz = Axis::max_axis(d.abs());
        // let kx = kz.next_axis();
        // let ky = kz.next_axis();
        Self { o, d }
    }
}

impl From<embree::Ray> for Ray {
    fn from(ray: embree::Ray) -> Self {
        let o = Vec3::new(ray.org_x, ray.org_y, ray.org_z);
        let d = Vec3::new(ray.dir_x, ray.dir_y, ray.dir_z);
        Self { o, d }
    }
}

impl From<Ray> for embree::Ray {
    fn from(ray: Ray) -> Self { Self::new(ray.o.into(), ray.d.into()) }
}

/// Implemented ray tracing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // todo: case_insensitive
pub enum RayTracingMethod {
    /// Standard using embree.
    Standard,
    /// Customised grid tracing method.
    Grid,
}

/// Struct used to record ray path.
#[derive(Debug, Copy, Clone)]
pub struct TrajectoryNode {
    /// The ray of the node.
    pub ray: Ray,

    /// The cosine of the angle between the ray and the normal (always positive).
    pub cos: f32,
}

/// Ray tracing record.
#[derive(Debug)]
pub struct RtcRecord {
    /// Path of traced ray.
    pub trajectory: Vec<TrajectoryNode>,

    /// Energy of the ray with different wavelengths at each bounce.
    /// Inner vector is the energy of the ray of different wavelengths.
    /// Outer vector is the number of bounces.
    pub energy_each_bounce: Vec<Vec<f32>>,
}

impl RtcRecord {
    /// Returns the bounces of traced ray.
    pub fn bounces(&self) -> usize { self.trajectory.len() - 1 }
}

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
/// [`Self::INDEX_FORMAT`]. The mesh is constructed from a height field.
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

/// Resolves the path for the given base path.
///
/// # Note
///
/// Resolved path is not guaranteed to exist.
///
/// # Arguments
///
/// * `input_path` - Path of the input file.
///
/// * `output_path` - Output path.
///
/// # Returns
///
/// A `PathBuf` indicating the resolved path. It differs according to the
/// base path and patterns inside of `path`.
///
///   1. `path` is `None`
///
///      Returns the current working directory.
///
///   2. `path` is prefixed with `//`
///
///      Returns the a path which is relative to `base_path`'s directory, with
///      the remaining of the `path` appended.
///
///   3. `path` is not `None` but is not prefixed with `//`
///
///      Returns the `path` as is.
pub(crate) fn resolve_file_path(base_path: &Path, path: Option<&Path>) -> PathBuf {
    path.map_or_else(
        || std::env::current_dir().unwrap(),
        |path| {
            if let Ok(prefix_stripped) = path.strip_prefix("//") {
                // Output path is relative to input file's directory
                let mut resolved = base_path
                    .parent()
                    .unwrap()
                    .to_path_buf()
                    .canonicalize()
                    .unwrap();
                resolved.push(prefix_stripped);
                resolved
            } else {
                // Output path is at somewhere else.
                path.to_path_buf()
            }
        },
    )
}

// fn measure_micro_surface_geometric_term(
//     device: wgpu::Device,
//     queue: wgpu::Queue,
//     view: &MicroSurfaceView,
// ) {
//     unimplemented!()
// }

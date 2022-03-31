use crate::gfx::camera::{Projection, ProjectionKind};
use glam::Vec3;
use wgpu::util::DeviceExt;
use crate::htfld::{Heightfield, regular_triangulation};
use crate::isect::Aabb;

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

    /// Buffer stores all the indices. Indices are stored following the order
    /// the order of azimuthal angle and polar angle when the mesh is
    /// generated.
    pub index_buffer: wgpu::Buffer,

    /// Index ranges for each set of triangles with the same normal.
    /// Note: only the start of the range is stored.
    pub index_ranges: Vec<(u32, u32)>,

    /// Number of bins in azimuthal angle of micro-facets' normal.
    pub num_azimuth_bins: usize,

    /// Number of bins in polar angle of micro-facets' normal.
    pub num_zenith_bins: usize,
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
    pub fn from_height_field(
        device: &wgpu::Device,
        hf: &Heightfield,
        azimuth_bin_size: f32,
        zenith_bin_size: f32,
    ) -> Self {
        let (rows, cols) = (hf.cols, hf.rows);
        let (positions, extent) = hf.generate_vertices();
        let indices: Vec<u32> = regular_triangulation(&positions, rows, cols);
        let triangles_count = indices.len() / 3;

        // Generate triangles' normals
        let mut normals: Vec<Vec3> = vec![];
        normals.resize(triangles_count, Vec3::ZERO);

        for i in 0..triangles_count {
            let i0 = indices[3 * i + 0] as usize;
            let i1 = indices[3 * i + 1] as usize;
            let i2 = indices[3 * i + 2] as usize;

            let v0 = positions[i0];
            let v1 = positions[i1];
            let v2 = positions[i2];

            let n = (v1 - v0).cross(v2 - v0).normalize();
            normals[i] = n;
        }

        let num_azimuth_bins = ((2.0 * std::f32::consts::PI) / azimuth_bin_size).ceil() as usize;
        let num_zenith_bins = ((std::f32::consts::PI / 2.0) / zenith_bin_size).ceil() as usize;
        let bins_count = num_azimuth_bins * num_zenith_bins;
        // Classify triangles by azimuth and zenith
        let mut sorted_faces: Vec<Vec<usize>> = vec![vec![]; bins_count];
        for i in 0..normals.len() {
            let n = normals[i];
            // atan2 is defined for [-pi, pi], so we need to convert the range to [0, 2pi]
            let azimuth = n.z.atan2(n.x) + std::f32::consts::PI;
            let zenith = n.y.acos();
            let mut azimuth_bin = (azimuth / azimuth_bin_size).floor() as usize;
            let mut zenith_bin = (zenith / zenith_bin_size).floor() as usize;
            if azimuth_bin > num_azimuth_bins {
                panic!("azimuth bin out of range: {}", azimuth_bin);
            }
            if zenith_bin > num_zenith_bins {
                panic!("zenith bin out of range: {}", zenith_bin);
            }
            // Sometimes because of the floating point precision, the azimuth bin may equal to the number of bins.
            if azimuth_bin == num_azimuth_bins {
                azimuth_bin = num_azimuth_bins - 1;
            }
            if zenith_bin == num_zenith_bins {
                zenith_bin = num_zenith_bins - 1;
            }
            sorted_faces[azimuth_bin * num_zenith_bins + zenith_bin].push(i);
        }
        println!("sorted_faces:");
        for faces in &sorted_faces {
            if !faces.is_empty() {
                println!("{:?}", faces);
            }
        }

        let mut index_ranges = vec![(0, 0); bins_count];
        // Reorder indices
        let mut ordered_indices = vec![0; indices.len()];
        let mut index_range_start: usize = 0;
        // Iterate over the bins
        for i in 0..sorted_faces.len() {
            let faces = &sorted_faces[i];
            // No faces classified in this bin
            if faces.is_empty() {
                index_ranges[i] = (index_range_start as u32, index_range_start as u32);
            } else {
                for j in 0..faces.len() {
                    for k in 0..3 {
                        ordered_indices[index_range_start + j * 3 + k] = indices[faces[j] * 3 + k];
                    }
                }
                index_ranges[i] = (index_range_start as u32, (index_range_start + faces.len() * 3) as u32);
                index_range_start += faces.len() * 3;
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_vertex_buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_view_index_buffer"),
            contents: bytemuck::cast_slice(&ordered_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            extent,
            vertex_buffer,
            index_buffer,
            index_ranges,
            num_azimuth_bins,
            num_zenith_bins
        }
    }
}

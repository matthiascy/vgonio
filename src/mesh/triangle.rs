use crate::isect::Aabb;
use embree::Geometry;
use glam::Vec3;
use std::sync::Arc;

/// Surface triangulation method.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TriangulationMethod {
    /// Regular triangulation.
    Regular,

    /// Delaunay triangulation.
    Delaunay,
}

/// Triangle representation of the surface mesh.
#[derive(Debug)]
pub struct TriangleMesh {
    /// Axis-aligned bounding box of the mesh.
    pub extent: Aabb,

    /// Number of triangles in the mesh.
    pub num_tris: usize,

    /// Number of vertices in the mesh.
    pub num_verts: usize,

    /// Vertices of the mesh.
    pub verts: Vec<Vec3>,

    /// Triangle indices.
    pub faces: Vec<u32>,

    /// Normal vectors of each triangle.
    pub normals: Vec<Vec3>,
}

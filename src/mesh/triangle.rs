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
    pub extent: Aabb,
    pub num_tris: usize,
    pub num_verts: usize,
    pub verts: Vec<Vec3>,
    pub faces: Vec<u32>,
}

use crate::isect::Aabb;
use glam::Vec3;

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
pub struct MicroSurfaceTriMesh {
    /// Axis-aligned bounding box of the mesh.
    pub extent: Aabb,

    /// Number of triangles in the mesh.
    pub num_facets: usize,

    /// Number of vertices in the mesh.
    pub num_verts: usize,

    /// Vertices of the mesh.
    pub verts: Vec<Vec3>,

    /// Vertex indices forming the facets which are triangles.
    pub facets: Vec<u32>,

    /// Normal vectors of each triangle.
    pub facet_normals: Vec<Vec3>,
}

impl MicroSurfaceTriMesh {
    /// Calculate the surface area of a facet.
    ///
    /// # Arguments
    ///
    /// * `facet` - Index of the facet.
    ///
    /// TODO(yang): unit of surface area
    pub fn facet_surface_area(&self, facet: usize) -> f32 {
        let v0 = self.verts[self.facets[facet * 3] as usize];
        let v1 = self.verts[self.facets[facet * 3 + 1] as usize];
        let v2 = self.verts[self.facets[facet * 3 + 2] as usize];
        let e0 = v1 - v0;
        let e1 = v2 - v0;
        0.5 * e0.cross(e1).length()
    }

    /// Calculate the macro surface area of the mesh.
    ///
    /// REVIEW(yang): temporarily the surface mesh is generated on XZ plane in
    /// right-handed Y up coordinate system.
    /// TODO(yang): unit of surface area
    pub fn macro_surface_area(&self) -> f32 {
        (self.extent.max.x - self.extent.min.x) * (self.extent.max.z - self.extent.min.z)
    }
}

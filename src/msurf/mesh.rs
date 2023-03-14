use crate::{app::cache::Asset, isect::Aabb, msurf::AxisAlignment};
use glam::Vec3;
use uuid::Uuid;

/// Triangle representation of the surface mesh.
///
/// It has the same length unit as the [`MicroSurface`] from which it is
/// generated.
#[derive(Debug)]
pub struct MicroSurfaceMesh {
    /// Unique identifier.
    pub uuid: Uuid,

    /// Uuid of the [`MicroSurface`] from which the mesh is generated.
    pub msurf: Uuid,

    /// Axis-aligned bounding box of the mesh.
    pub extent: Aabb,

    /// Axis alignment of the mesh.
    pub alignment: AxisAlignment,

    /// Number of triangles in the mesh.
    pub num_facets: usize,

    /// Number of vertices in the mesh.
    pub num_verts: usize,

    /// Vertices of the mesh.
    pub verts: Vec<Vec3>,

    /// Vertex indices forming the facets which are triangles.
    pub facets: Vec<u32>,

    /// Normal vectors of each facet.
    pub facet_normals: Vec<Vec3>,

    /// Surface area of each facet.
    pub facet_areas: Vec<f32>,
}

impl Asset for MicroSurfaceMesh {}

impl MicroSurfaceMesh {
    /// Returns the surface area of a facet.
    ///
    /// # Arguments
    ///
    /// * `facet` - Index of the facet.
    ///
    /// TODO(yang): unit of surface area
    pub fn facet_surface_area(&self, facet: usize) -> f32 { self.facet_areas[facet] }

    /// Calculate the macro surface area of the mesh.
    ///
    /// REVIEW(yang): temporarily the surface mesh is generated on XZ plane in
    /// right-handed Y up coordinate system.
    /// TODO(yang): unit of surface area
    pub fn macro_surface_area(&self) -> f32 {
        (self.extent.max.x - self.extent.min.x) * (self.extent.max.z - self.extent.min.z)
    }
}

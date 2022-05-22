use crate::htfld::Heightfield;
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

impl TriangleMesh {
    /// Create a `embree::TriangleMesh` from the triangle mesh.
    pub fn to_embree_mesh(&self, device: Arc<embree::Device>) -> Arc<embree::TriangleMesh> {
        let mut mesh = embree::TriangleMesh::unanimated(device, self.num_tris, self.num_verts);
        {
            let mut mesh_ref_mut = Arc::get_mut(&mut mesh).unwrap();
            {
                let mut verts_buffer = mesh_ref_mut.vertex_buffer.map();
                let mut indxs_buffer = mesh_ref_mut.index_buffer.map();
                for (i, vert) in self.verts.iter().enumerate() {
                    verts_buffer[i] = [vert.x, vert.y, vert.z, 1.0];
                }
                // TODO: replace with slice::as_chunks when it's stable.
                (0..self.num_tris).for_each(|tri| {
                    indxs_buffer[tri] = [
                        self.faces[tri * 3],
                        self.faces[tri * 3 + 1],
                        self.faces[tri * 3 + 2],
                    ];
                })
            }
            mesh_ref_mut.commit()
        }
        mesh
    }
}

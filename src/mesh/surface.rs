//! # Attributes
//!
//! + Vertex attributes:
//!   - Position (mandatory)
//!   - Normal
//!   - Color
//!   - Texture Coordinates
//! + Face attributes
//!   - Normal
//!   - Color
//!   - Texture
//! + Edge attributes (rarely used)
//!   - Visible line

use crate::{
    error::Error,
    gfx::MeshView,
    htfld::Heightfield,
    isect::Aabb,
    math::Vec3,
    mesh::half_edge::{HEDart, HEEdge, HEFace, HEVert},
};
use std::{
    any::Any,
    collections::HashMap,
    ops::{Deref, DerefMut},
};

/// Vertex attributes container.
trait AttribContainer {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct AttribArray<T>(Vec<T>);

impl<T> AttribArray<T> {
    pub fn new() -> Self { Self(vec![]) }
}

impl<T> Deref for AttribArray<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> DerefMut for AttribArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T: 'static> AttribContainer for AttribArray<T> {
    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

type VertAttribs = HashMap<String, Box<dyn AttribContainer>>;
type FaceAttribs = HashMap<String, Box<dyn AttribContainer>>;
type EdgeAttribs = HashMap<String, Box<dyn AttribContainer>>;

/// Half-edge representation of a surface mesh.
pub struct SurfaceMesh {
    /// Axis-aligned Bounding Box
    extent: Aabb,
    /// Vertices' connectivity.
    verts: Vec<HEVert>,
    /// Faces' connectivity.
    faces: Vec<HEFace>,
    /// Half-edge connectivity.
    darts: Vec<HEDart>,
    /// Edge connectivity.
    edges: Vec<HEEdge>,
    /// Vertex attributes
    vert_attribs: VertAttribs,
    /// Face attributes
    face_attribs: FaceAttribs,
    /// Edge attributes
    edge_attribs: EdgeAttribs,
    /// Specifies whether if the surface has been modified.
    is_dirty: bool,
}

// VertexIterator, HalfEdgeIterator, EdgeIterator, FaceIterator
// VertexAroundVertexCirculator, HalfEdgeAroundVertexCirculator,
// FaceAroundVertexCirculator, VertexAroundFaceCirculator,
// HalfEdgeAroundFaceCirculator SurfaceCurvature -- compute per-vertex curvature
// (min, max, mean, Gaussian) SurfaceGeodesic -- compute geodesic distance from
// a set of seed vertices SurfaceHoleFilling -- close simple holes
// SurfaceNormals -- compute surface normals
// SurfaceSimplification -- surface mesh simplification
// SurfaceSmoothing -- surface subdivision algorithm
// SurfaceTriangulation -- triangulate polygons to get a pure triangle mesh.

impl Default for SurfaceMesh {
    fn default() -> Self { Self::new() }
}

impl SurfaceMesh {
    pub fn new() -> Self {
        let mut vert_attribs = VertAttribs::new();
        let mut face_attribs = FaceAttribs::new();
        let edge_attribs = EdgeAttribs::new();

        // Where stores position of each vertex.
        vert_attribs.insert("positions".into(), Box::new(AttribArray::<Vec3>::new()));
        vert_attribs.insert("normals".into(), Box::new(AttribArray::<Vec3>::new()));
        face_attribs.insert("normals".into(), Box::new(AttribArray::<Vec3>::new()));

        Self {
            extent: Default::default(),
            verts: vec![],
            faces: vec![],
            darts: vec![],
            edges: vec![],
            vert_attribs,
            face_attribs,
            edge_attribs,
            is_dirty: false,
        }
    }

    /// Constructs a `SurfaceMesh` from samples of a `HeightField`.
    ///
    /// This method uses regular grid triangulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use vgonio::{
    ///     htfld::{AxisAlignment, Heightfield},
    ///     mesh::SurfaceMesh,
    /// };
    /// let hf = Heightfield::new_by(6, 6, 0.1, 0.1, AxisAlignment::XZ, |x, y| {
    ///     (x * y) as f32 * 0.5
    /// });
    /// let mesh = SurfaceMesh::from_height_field(&hf);
    /// ```
    pub fn from_height_field(hf: &Heightfield) -> Self {
        // todo: orientation
        let (rows, cols) = (hf.rows, hf.cols);
        let (vert_positions, extent) = hf.generate_vertices();
        // Regular triangulation, compute indices for each triangle (counter clock-wise)
        let faces_count = (rows - 1) * (cols - 1);
        let darts_count = 6 * (rows - 1) * (cols - 1);
        let edges_count = 3 * (rows - 1) * (cols - 1) + (rows - 1) + (cols - 1);
        let mut he_darts: Vec<HEDart> = Vec::with_capacity(darts_count);
        let mut he_verts: Vec<HEVert> = Vec::with_capacity(vert_positions.len());
        let mut he_edges: Vec<HEEdge> = Vec::with_capacity(edges_count);
        let mut he_faces: Vec<HEFace> = Vec::with_capacity(faces_count);

        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                let is_last_col = c == cols - 2;
                let is_last_row = r == rows - 2;
                let curr_dart = he_darts.len();
                let curr_vert = r * cols + c;
                let curr_face = (r * (cols - 1) + c) * 2;
                //  v2 - v3
                //  | \  |
                // v0 - v1
                let (v0, v1, v2, v3) = {
                    (
                        curr_vert,
                        curr_vert + 1,
                        curr_vert + cols,
                        curr_vert + 1 + cols,
                    )
                };

                let lower_twin = if r == 0 {
                    usize::MAX
                } else {
                    curr_dart - 6 * (cols - 1)
                };
                let left_twin = if c == 0 { usize::MAX } else { curr_dart - 3 };
                let right_twin = if is_last_col {
                    usize::MAX
                } else {
                    curr_dart + 8
                };
                let top_twin = if is_last_row {
                    usize::MAX
                } else {
                    curr_dart + 6 * (cols - 1)
                };
                he_darts.extend_from_slice(&[
                    // 1st triangle
                    HEDart::new(v0, curr_face, curr_dart + 1, curr_dart + 2, lower_twin), // curr_dart + 0
                    HEDart::new(v1, curr_face, curr_dart + 2, curr_dart, curr_dart + 5),  // curr_dart + 1
                    HEDart::new(v2, curr_face, curr_dart, curr_dart + 1, left_twin),      // curr_dart + 2
                    // 2nd triangle
                    HEDart::new(v1, curr_face + 1, curr_dart + 4, curr_dart + 5, right_twin), // curr_dart + 3
                    HEDart::new(v3, curr_face + 1, curr_dart + 5, curr_dart + 3, top_twin),   // curr_dart + 4
                    HEDart::new(v2, curr_face + 1, curr_dart + 3, curr_dart + 4, curr_dart + 1), // // curr_dart + 5
                ]);
                he_edges.extend_from_slice(&[
                    HEEdge(curr_dart, lower_twin),
                    HEEdge(curr_dart + 1, curr_dart + 5),
                    HEEdge(curr_dart + 2, left_twin),
                ]);
                he_verts.push(HEVert::new(v0, curr_dart));

                if is_last_col {
                    he_edges.push(HEEdge(curr_dart + 3, usize::MAX));
                    he_verts.push(HEVert::new(v1, curr_dart + 3));
                }

                if is_last_row {
                    he_edges.push(HEEdge(curr_dart + 4, usize::MAX));
                    he_verts.push(HEVert::new(v2, curr_dart + 2));
                }

                if is_last_col && is_last_row {
                    he_verts.push(HEVert::new(v3, curr_dart + 4));
                }

                he_faces.extend_from_slice(&[
                    HEFace::new(curr_face, curr_dart),
                    HEFace::new(curr_face + 1, curr_dart + 3),
                ]);
            }
        }

        let mut vert_attribs = VertAttribs::new();
        vert_attribs.insert("positions".into(), Box::new(AttribArray(vert_positions)));
        vert_attribs.insert("normals".into(), Box::new(AttribArray::<Vec3>::new()));
        let mut face_attribs = FaceAttribs::new();
        face_attribs.insert("normals".into(), Box::new(AttribArray::<Vec3>::new()));

        Self {
            extent,
            verts: he_verts,
            faces: he_faces,
            darts: he_darts,
            edges: he_edges,
            vert_attribs,
            face_attribs,
            edge_attribs: EdgeAttribs::new(),
            is_dirty: true,
        }
    }

    pub fn update_vertex_normals(&mut self) {
        if self.is_dirty {
            // let vert_normals = self.vert_attribs.get(&"normals")
            //     .into_any().downcast_mut::<AttribArray<Normal3f>>()?;
            // vert_normals.0.resize(self.verts.len(), Normal3f::new(0.0, 0.0,
            // 0.0)); for vert in &self.verts {
            //     let mut start = self.darts[face]
            //     vert_normals[vert] =
            // }
        }
    }

    pub fn update_face_normals(&mut self) {
        if self.is_dirty {
            let face_normals = self
                .face_attribs
                .get_mut(&"normals".to_string())
                .unwrap()
                .as_any_mut()
                .downcast_mut::<AttribArray<Vec3>>()
                .unwrap();
            face_normals
                .0
                .resize(self.faces.len(), Vec3::new(0.0, 0.0, 0.0));
            let vert_positions = self
                .vert_attribs
                .get_mut(&"positions".to_string())
                .unwrap()
                .as_any_mut()
                .downcast_mut::<AttribArray<Vec3>>()
                .unwrap();
            for face in self.faces.iter() {
                let start = face.dart;
                let mut curr = self.darts[start].next;
                let mut verts: Vec<usize> = vec![self.darts[start].vert];
                while start != curr {
                    verts.push(self.darts[curr].vert);
                    curr = self.darts[curr].next;
                }

                let face_idx = self.darts[face.dart].face;
                let positions: Vec<Vec3> = verts.iter().map(|idx| vert_positions[*idx]).collect();
                face_normals[face_idx] = {
                    let e0 = positions[1] - positions[0];
                    let e1 = positions[2] - positions[0];
                    e0.cross(e1).normalize()
                }
            }
        }
    }

    // TODO: add_vertex
    // TODO: add_triangle

    pub fn dump_vertex_normals(&self) -> Result<(), Error> { Ok(()) }

    pub fn dump_face_normals(&self) -> Result<(), Error> { Ok(()) }

    pub fn view(&self) -> MeshView { todo!() }
}

//! Implementation of a doubly connected edge list (DCEL) data structure, known
//! also as half-edge data structure, which is used to represent the geometry of
//! a surface mesh.
//! Any valid half-edge mesh must be manifold, oriented, and closed.

use glam::{Vec2, Vec3};
use std::{borrow::Cow, collections::HashMap};

/// Vertex representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vert {
    /// Position of the vertex.
    pub pos: u32,
    /// Outgoing half-edge.
    pub dart: u32,
}

impl Vert {
    /// Invalid vertex.
    pub const fn invalid() -> Self {
        Self {
            pos: u32::MAX,
            dart: u32::MAX,
        }
    }

    /// A valid vertex means that the position and the outgoing half-edge are
    /// both valid.
    pub fn is_valid(&self) -> bool { self.pos != u32::MAX && self.dart != u32::MAX }
}

/// Face representation.
///
/// Each half-edge is associated with a face where the half-edge points along
/// the counter-clockwise direction of the face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Face {
    pub dart: u32,
}

impl Face {
    /// Invalid face.
    pub const fn invalid() -> Self { Self { dart: u32::MAX } }

    /// A valid face means that the half-edge is valid.
    pub fn is_valid(&self) -> bool { self.dart != u32::MAX }
}

/// Half-edge representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dart {
    /// The origin vertex of the half-edge.
    pub vert: u32,
    /// The incident face of the half-edge.
    pub face: u32,
    /// The opposite half-edge.
    pub twin: u32,
    /// The next half-edge.
    pub next: u32,
}

impl Dart {
    /// Invalid half-edge.
    pub const fn invalid() -> Self {
        Self {
            twin: u32::MAX,
            next: u32::MAX,
            vert: u32::MAX,
            face: u32::MAX,
        }
    }

    /// A valid half-edge means that the origin vertex, the incident face and
    /// the next half-edge are all valid.
    pub fn is_valid(&self) -> bool {
        self.vert != u32::MAX && self.face != u32::MAX && self.next != u32::MAX
    }
}

/// Doubly connected edge list (DCEL) data structure.
pub struct HalfEdgeMesh<'a> {
    /// Positions of the vertices.
    pub positions: Cow<'a, [Vec3]>,
    /// Vertices of the mesh.
    pub verts: Box<[Vert]>,
    /// Faces of the mesh.
    pub faces: Box<[Face]>,
    /// Half-edges of the mesh.
    pub darts: Box<[Dart]>,
}

impl<'a> HalfEdgeMesh<'a> {
    /// Create a new half-edge mesh from the given positions and triangles.
    pub fn new(positions: &'a [Vec3], tris: &'a [u32]) -> Self {
        assert_eq!(tris.len() % 3, 0, "The input triangles must be valid.");
        let num_verts = positions.len();
        let num_faces = tris.len() / 3;

        let positions = Cow::Borrowed(positions);
        let mut verts = vec![Vert::invalid(); num_verts].into_boxed_slice();
        let mut faces = vec![Face::invalid(); num_faces].into_boxed_slice();
        let mut darts = vec![Dart::invalid(); num_faces * 3].into_boxed_slice();
        // The key is the vertex indices of the two ends of the half-edge.
        // The value is the two half-edges that are twins.
        let mut twins: HashMap<(u32, u32), (u32, u32)> = HashMap::new();

        for (face_idx, (tri, face)) in tris.chunks(3).zip(faces.iter_mut()).enumerate() {
            let base_dart_idx = face_idx * 3;
            face.dart = base_dart_idx as u32;

            for i in 0..3 {
                let dart_idx = base_dart_idx + i;
                let vert_idx = tri[i] as usize;

                darts[dart_idx].vert = vert_idx as u32;
                darts[dart_idx].face = face_idx as u32;
                darts[dart_idx].next = (base_dart_idx + (i + 1) % 3) as u32;

                let dart_head = tri[i];
                let dart_tail = tri[(i + 1) % 3];
                let twin_key = (dart_head.min(dart_tail), dart_head.max(dart_tail));
                if let Some((_, b)) = twins.get_mut(&twin_key) {
                    *b = dart_idx as u32;
                } else {
                    twins.insert(twin_key, (dart_idx as u32, u32::MAX));
                }

                verts[vert_idx].pos = vert_idx as u32;
                // Set the half-edge of the vertex if it's not set yet
                if !verts[vert_idx].is_valid() {
                    verts[vert_idx].dart = dart_idx as u32;
                }
            }
        }

        for (a, b) in twins.values() {
            if *a == u32::MAX || *b == u32::MAX {
                continue;
            }
            darts[*a as usize].twin = *b;
            darts[*b as usize].twin = *a;
        }

        Self {
            positions,
            verts,
            faces,
            darts,
        }
    }
}

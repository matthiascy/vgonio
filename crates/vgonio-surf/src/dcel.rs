//! Implementation of a doubly connected edge list (DCEL) data structure, known
//! also as half-edge data structure, which is used to represent the geometry of
//! a surface mesh.
//! Any valid half-edge mesh must be manifold, oriented, and closed.

use crate::{TriangleUVSubdivision, TriangulationPattern, VertLoc};
use glam::{DVec3, Vec2, Vec3};
use log::log;
use std::{borrow::Cow, collections::VecDeque};

use rayon::iter::{IntoParallelRefMutIterator, ParallelBridge, ParallelIterator};
use std::{collections::HashMap, ptr::addr_of_mut, sync::RwLock};

type NoHashMap<K, V> = HashMap<K, V, nohash_hasher::BuildNoHashHasher<K>>;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dart {
    /// The origin vertex of the half-edge.
    pub vert: u32,
    /// The incident face of the half-edge.
    pub face: u32,
    /// The opposite half-edge.
    pub twin: u32,
    /// The next half-edge.
    pub next: u32,
    /// The previous half-edge.
    pub prev: u32,
}

impl Dart {
    /// Invalid half-edge.
    pub const fn invalid() -> Self {
        Self {
            twin: u32::MAX,
            next: u32::MAX,
            vert: u32::MAX,
            face: u32::MAX,
            prev: u32::MAX,
        }
    }

    /// A valid half-edge means that the origin vertex, and
    /// the next half-edge is valid.
    pub fn is_valid(&self) -> bool { self.vert != u32::MAX && self.next != u32::MAX }

    /// Check if the half-edge is on the boundary.
    pub fn is_boundary(&self) -> bool { self.face == u32::MAX }

    /// Check if the half-edge has a twin.
    pub fn has_twin(&self) -> bool { self.twin != u32::MAX }
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
    pub fn new<'b>(positions: Cow<'a, [Vec3]>, tris: &'b [u32]) -> Self {
        log::trace!("Constructing a half-edge mesh with tris: {:?}", tris);
        #[cfg(feature = "bench")]
        let start = std::time::Instant::now();
        if positions.is_empty() || tris.is_empty() {
            log::error!("Constructing a half-edge mesh with empty positions or triangles!");
        }
        assert_eq!(tris.len() % 3, 0, "The input triangles must be valid.");
        let num_verts = positions.len();
        let num_faces = tris.len() / 3;

        let mut verts = vec![Vert::invalid(); num_verts].into_boxed_slice();
        let mut faces = vec![Face::invalid(); num_faces].into_boxed_slice();
        let mut darts = vec![Dart::invalid(); num_faces * 3];

        // Initialize the vertices with the position index to avoid modifying inside
        // loop constructing darts.
        for (i, vert) in verts.iter_mut().enumerate() {
            vert.pos = i as u32;
        }

        // The pointer to the darts as one dart will not be accessed by multiple threads
        // in the later loop.
        struct DartsPtr(pub *mut Dart);
        unsafe impl Send for DartsPtr {}
        unsafe impl Sync for DartsPtr {}

        let darts_ptr = DartsPtr(darts.as_mut_ptr());

        // The key is the vertex indices of the two ends of the half-edge.
        // The value is the two half-edges that are twins.
        let mut twins: HashMap<(u32, u32), (u32, u32)> = HashMap::new();

        // Parallelize the construction of the darts.
        tris.chunks(3 * 1024)
            .zip(faces.chunks_mut(1024))
            .enumerate()
            .par_bridge()
            .for_each(|(chunk_idx, (tris_chunk, faces_chunk))| {
                tris_chunk
                    .chunks(3)
                    .zip(faces_chunk.iter_mut())
                    .enumerate()
                    .for_each(|(face_local_idx, (tri, face))| {
                        let _ = &darts_ptr;
                        let face_idx = chunk_idx * 1024 + face_local_idx;
                        let base_dart_idx = face_idx * 3;
                        face.dart = base_dart_idx as u32;
                        unsafe {
                            for i in 0..3 {
                                let dart_idx = base_dart_idx + i;
                                let vert_idx = *tri.get_unchecked(i);

                                let dart = darts_ptr.0.add(dart_idx);
                                (*dart).vert = vert_idx;
                                (*dart).face = face_idx as u32;
                                (*dart).next = (base_dart_idx + (i + 1) % 3) as u32;
                                (*dart).prev = (base_dart_idx + (i + 2) % 3) as u32;
                            }
                        }
                    })
            });

        // Process the vertices and the twins.
        for (face_idx, tri) in tris.chunks(3).enumerate() {
            let base_dart_idx = face_idx * 3;

            for i in 0..3 {
                let dart_idx = base_dart_idx + i;
                let vert_idx = tri[i] as usize;

                let vert = unsafe { verts.get_unchecked_mut(vert_idx) };
                // Set the half-edge of the vertex if it's not set yet
                if !vert.is_valid() {
                    vert.dart = dart_idx as u32;
                }

                let dart_head = tri[i];
                let dart_tail = tri[(i + 1) % 3];
                let twin_key = (dart_head.min(dart_tail), dart_head.max(dart_tail));
                if let Some((_, b)) = twins.get_mut(&twin_key) {
                    *b = dart_idx as u32;
                } else {
                    twins.insert(twin_key, (dart_idx as u32, u32::MAX));
                }
            }
        }

        log::trace!("Twins: {:?}", twins);

        // Number of darts incident on a face.
        let n_darts = darts.len();
        let mut boundary_darts = Vec::with_capacity(num_verts);
        // The boundary darts are stored in a hashmap where the key is the dart index,
        // and the value is the start and end vertices indices of the boundary.
        let mut boundary_darts_ends: HashMap<u32, (u32, u32)> = HashMap::new();
        for (a, b) in twins.values() {
            // The twin of the half-edge is on the boundary.
            if *b == u32::MAX {
                // The half-edge located on the face.
                let dart = darts[*a as usize];
                let next_dart_vert = darts[dart.next as usize].vert;
                let boundary_dart_idx = (n_darts + boundary_darts.len()) as u32;
                darts[*a as usize].twin = boundary_dart_idx;

                // Construct the dart on the boundary.
                let mut boundary_dart = Dart::invalid();
                boundary_dart.twin = *a;
                boundary_dart.vert = next_dart_vert;
                boundary_dart.face = u32::MAX;
                boundary_dart.next = u32::MAX;
                boundary_dart.prev = u32::MAX;
                boundary_darts.push(boundary_dart);
                boundary_darts_ends.insert(boundary_dart_idx, (boundary_dart.vert, dart.vert));
                continue;
            }
            darts[*a as usize].twin = *b;
            darts[*b as usize].twin = *a;
        }

        log::trace!(
            "Boundary darts ({}): {:?}",
            boundary_darts.len(),
            boundary_darts
        );
        log::trace!(
            "Boundary darts ends ({}): {:?}",
            boundary_darts_ends.len(),
            boundary_darts_ends
        );

        darts.append(&mut boundary_darts);
        darts.shrink_to_fit();

        // Iterate over the boundary darts and set the next and previous darts.
        for (dart_idx, (start, end)) in boundary_darts_ends.iter() {
            let mut dart = &mut darts[*dart_idx as usize];
            dart.next = boundary_darts_ends
                .iter()
                .find(|(_, (s, _))| s == end)
                .unwrap()
                .0
                .clone();
            dart.prev = boundary_darts_ends
                .iter()
                .find(|(_, (_, e))| e == start)
                .unwrap()
                .0
                .clone();
        }

        #[cfg(feature = "bench")]
        log::debug!(
            "Constructing the half-edge mesh took: {:?} ms",
            start.elapsed().as_millis()
        );

        Self {
            positions,
            verts,
            faces,
            darts: darts.into_boxed_slice(),
        }
    }

    /// Gets all the half-edges of the face together with the dart index.
    pub fn darts_of_face(&self, face_idx: usize) -> Box<[(u32, Dart)]> {
        let face = &self.faces[face_idx];
        let mut darts = Vec::new();
        let mut dart_idx = face.dart;
        loop {
            let dart = &self.darts[dart_idx as usize];
            darts.push((dart_idx, *dart));
            dart_idx = dart.next;
            if dart_idx == face.dart {
                break;
            }
        }
        darts.into_boxed_slice()
    }

    /// Get the faces on which the vertex is incident.
    pub fn faces_of_vert(&self, vert_idx: usize) -> Box<[u32]> {
        let vert = &self.verts[vert_idx];
        let mut faces = Vec::new();
        let mut dart_idx = vert.dart;
        loop {
            let dart = &self.darts[dart_idx as usize];
            if !dart.is_boundary() {
                faces.push(dart.face);
            }
            let prev = &self.darts[dart.prev as usize];
            dart_idx = prev.twin;
            if dart_idx == vert.dart {
                break;
            }
        }
        faces.into_boxed_slice()
    }

    /// Subdivide the mesh by looping over the faces and subdividing each face
    /// into smaller faces according to the given uv coordinates of the
    /// triangle.
    ///
    /// # Arguments
    ///
    /// * `uvs` - The uv coordinates of the desired interpolation points on the
    ///   face; the input UVs contain also the UVs of the original vertices. The
    ///   uv coordinates are in the range [0, 1]; the first vertex of the face
    ///   is the origin, the edge from the second to the origin is the u-axis,
    ///   and the edge from the third to the origin is the v-axis. See
    ///   [`TriangleUVSubdivision::calc_pnts_uvs`] for more details.
    ///
    /// * `interp` - The interpolation function that takes the uv coordinates,
    ///   the positions of the vertices of the face, and the output positions of
    ///   the new vertices.
    pub fn subdivide_by_uvs<I>(
        &self,
        sub: &TriangleUVSubdivision,
        interp: I,
    ) -> HalfEdgeMesh<'static>
    where
        I: Fn(&[Vec3], &[Vec2], &mut [DVec3]),
    {
        log::debug!("Subdividing the mesh by uvs.: {:?}", sub);
        let new_num_faces = sub.n_tris as usize * self.faces.len();
        let mut new_positions = self.positions.to_vec();
        let mut new_triangles = vec![u32::MAX; new_num_faces * 3];

        // The new triangles that will be added to the mesh.
        let mut interp_pnts_per_face = vec![DVec3::ZERO; sub.uvs.len()].into_boxed_slice();
        let mut new_tris_per_face = sub.indices.clone();

        // Records the shared edges of the face.
        // u32 - the dart index of the shared edge.
        // VertLoc - the location of the shared edge.
        // Box<u32> - the indices of the new vertices.
        let mut shared: NoHashMap<u32, (VertLoc, VecDeque<u32>)> = NoHashMap::default();
        let mut shared_to_remove = [u32::MAX; 3];

        // Vertex index replacement table from local to global.
        let mut replacement = vec![u32::MAX; sub.n_pnts as usize].into_boxed_slice();
        let is_bottom_left_to_top_right = sub.pattern == TriangulationPattern::BottomLeftToTopRight;
        // Loop over the faces and subdivide each face.
        for (old_face_idx, (face, new_tris)) in self
            .faces
            .iter()
            .zip(new_triangles.chunks_mut(sub.n_tris as usize * 3))
            .enumerate()
        {
            log::trace!("Processing face: {}", old_face_idx);
            // 1. Get the vertices of the face.
            let mut old_tri_verts = [Vec3::ZERO; 3];
            let mut old_tri_indis = [u32::MAX; 3];
            let darts_of_face = self.darts_of_face(old_face_idx);
            assert_eq!(darts_of_face.len(), 3, "The face must be a triangle.");
            old_tri_verts
                .iter_mut()
                .zip(darts_of_face.iter())
                .zip(old_tri_indis.iter_mut())
                .for_each(|((v, (_, d)), i)| {
                    let vert_idx = d.vert as usize;
                    *v = self.positions[vert_idx];
                    *i = vert_idx as u32;
                });

            // 2. Interpolate the new vertices.
            interp(&old_tri_verts, &sub.uvs, &mut interp_pnts_per_face);

            // Index is the edge, value is dart index.
            let mut local_shared = [u32::MAX; 3];
            // 3. Append the new vertices to the mesh & update the shared edges.
            for (i, (dart_idx, dart)) in darts_of_face.iter().enumerate() {
                if dart.has_twin() {
                    local_shared[i] = *dart_idx;
                    if !shared.contains_key(&dart.twin) {
                        // We don't know yet the actual vertex index of the newly created vertex.
                        shared.insert(
                            *dart_idx,
                            (
                                VertLoc::Edge(i as u32),
                                VecDeque::with_capacity(sub.n_ctrl_pnts as usize - 2),
                            ),
                        );
                    }
                }
            }
            log::trace!("Construct shared edges: {:?}", shared);
            log::trace!("Construct shared edges (local): {:?}", local_shared);

            let new_tri_vert_idx_offset = new_positions.len() as u32;
            log::trace!("num verts start: {}", new_positions.len());
            let mut next_new_tri_vert_idx = new_tri_vert_idx_offset;
            for (new_pnt, loc) in interp_pnts_per_face.iter().zip(sub.locations.iter()) {
                match loc {
                    VertLoc::Vert(_) => {
                        // The vertex comes from the original face.
                        // Do nothing.
                    }
                    l @ (VertLoc::Edge(_) | VertLoc::Inside) => {
                        // Update the shared edges.
                        if let Some(edge_loc) = l.edge_idx() {
                            // The new vertex is on the edge.
                            let dart_idx = local_shared[edge_loc as usize];
                            // This edge is shared.
                            if dart_idx != u32::MAX {
                                log::trace!(" - Shared edge: {}, dart: {}", edge_loc, dart_idx);

                                // Do not construct new vertices for the vertices that are shared.
                                if shared.contains_key(&self.darts[dart_idx as usize].twin) {
                                    continue;
                                }

                                // The first time we encounter the shared edge, we need to
                                // construct the new vertices.
                                shared
                                    .get_mut(&dart_idx)
                                    .unwrap()
                                    .1
                                    .push_back(next_new_tri_vert_idx);
                            }
                        }

                        new_positions.push(Vec3::new(
                            new_pnt.x as f32,
                            new_pnt.y as f32,
                            new_pnt.z as f32,
                        ));
                        next_new_tri_vert_idx += 1;
                    }
                }
            }

            log::trace!("shared edges: {:?}", shared);
            log::trace!("num verts after: {}", new_positions.len());

            // 4. Build vertex index replacement table.
            // let mut replacement = vec![u32::MAX; sub.n_pnts as usize].into_boxed_slice();
            let mut n_shared = 0;
            let mut shared_to_remove_idx = 0;
            replacement
                .iter_mut()
                .zip(sub.locations.iter())
                .enumerate()
                .for_each(|(idx, (rep, loc))| {
                    match loc {
                        VertLoc::Vert(i) => {
                            // The vertex comes from the original face.
                            *rep = old_tri_indis[*i as usize];
                            return;
                        }
                        VertLoc::Inside => {}
                        e @ VertLoc::Edge(i) => {
                            // Shared edge.
                            if local_shared[*i as usize] != u32::MAX {
                                let dart = self.darts[local_shared[*i as usize] as usize];
                                // Can we find the vertex index from the shared edges?
                                if let Some((_, verts)) = shared.get_mut(&dart.twin) {
                                    *rep = if e == &VertLoc::E02
                                        || (e == &VertLoc::E12 && is_bottom_left_to_top_right)
                                    {
                                        verts.pop_back().unwrap()
                                    } else {
                                        verts.pop_front().unwrap()
                                    };

                                    if verts.is_empty() {
                                        shared_to_remove[shared_to_remove_idx] = dart.twin;
                                        shared_to_remove_idx += 1;
                                    }
                                    n_shared += 1;
                                    return;
                                }
                            }
                        }
                    }

                    if idx as u32 > sub.n_ctrl_pnts - 1 {
                        // The rest of the vertices come from the new vertices.
                        *rep = idx as u32 + new_tri_vert_idx_offset - 2 - n_shared;
                    } else {
                        // The rest of the vertices come from the new vertices.
                        *rep = idx as u32 + new_tri_vert_idx_offset - 1 - n_shared;
                    }
                });
            log::trace!("Replacement table: {:?}", replacement);

            // 5. Convert the newly created triangle indices into global vertex indices.
            new_tris_per_face.iter_mut().for_each(|vidx| {
                *vidx = replacement[*vidx as usize];
            });

            log::trace!("New triangles - local: {:?}", sub.indices);
            log::trace!("New triangles - global: {:?}", new_tris_per_face);
            log::trace!("Old triangle vertices: {:?}", old_tri_indis);

            // 6. Append the new triangles to the mesh.
            new_tris.copy_from_slice(&new_tris_per_face);

            // Cleanup the shared edges.
            for dart_idx in shared_to_remove.iter_mut() {
                if *dart_idx == u32::MAX {
                    continue;
                }
                shared.remove(&dart_idx);
                *dart_idx = u32::MAX;
            }

            // Reset the replacement table.
            replacement.fill(u32::MAX);

            log::trace!("Shared edges after cleanup: {:?}", shared);

            // Reset the new triangles.
            new_tris_per_face.copy_from_slice(&sub.indices);
        }

        new_positions.shrink_to_fit();
        new_triangles.shrink_to_fit();
        HalfEdgeMesh::new(Cow::Owned(new_positions), &new_triangles)
    }

    /// Get the number of vertices.
    pub fn n_verts(&self) -> usize { self.verts.len() }

    /// Get the number of faces.
    pub fn n_faces(&self) -> usize { self.faces.len() }

    pub fn debug_print(&self) {
        log::debug!("Vertices: {:?}", self.verts);
        log::debug!("Faces: {:?}", self.faces);
        log::debug!("Darts: {:?}", self.darts);
    }
}

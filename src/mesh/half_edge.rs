/// Representation of the half-edge.
/// Each `Dart` contains:
///   + origin vertex index
///   + incident face index
///   + next half-edge index
///   + prev half-edge index
///   + twin half-edge index
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HEDart {
    /// Vertex index of the origin of the [`HalfEdge`] (there starts the edge)
    pub vert: usize,
    /// Incident `Face` index, may be empty (set to usize::MAX)
    pub face: usize,
    /// Succeeding `Dart` index (counter clock-wise)
    pub next: usize,
    /// Previous `Dart` index (clock-wise)
    pub prev: usize,
    /// Twin `Dart` in range [0, usize::MAX), usize::MAX means that this dart
    /// doesn't have twin (boundary).
    pub twin: usize,
}

impl HEDart {
    pub fn new(vert: usize, face: usize, next: usize, prev: usize, twin: usize) -> Self {
        Self {
            vert,
            face,
            next,
            prev,
            twin,
        }
    }
}

/// Representation of a complete edge.
///
/// It contains two indices of the twin [`Dart`]s.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HEEdge(pub usize, pub usize);

/// Representation of a face
///
/// Each `HEFace` contains:
///   + face index
///   + adjacent half-edge index
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HEFace {
    /// Face index
    pub face: usize,
    /// Adjacent `Dart` index (first `Dart`'s index)
    pub dart: usize,
}

impl HEFace {
    pub fn new(face: usize, dart: usize) -> Self {
        Self { face, dart }
    }
}

/// Representation of a vertex.
///
/// Each `HEVert` contains :
///   + position index
///   + out-going half-edge index
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HEVert {
    /// Vertex index
    pub vert: usize,
    /// Outgoing [`Dart`] index (following the counter clock-wise direction)
    pub dart: usize,
}

impl HEVert {
    pub fn new(vert: usize, dart: usize) -> Self {
        Self { vert, dart }
    }
}

//! Surface smoothing algorithms.
pub mod cpnt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subdivision {
    /// Subdivide the surface using curved PN triangles (3D Bezier triangles).
    Curved(u32),
    /// Regularly subdivide the surface, adding variation to the height of new
    /// vertices.
    Wiggly(u32),
    /// No subdivision.
    None,
}

/// The kind of subdivision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivisionKind {
    Curved,
    Wiggly,
}

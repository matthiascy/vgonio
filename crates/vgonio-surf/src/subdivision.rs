//! Surface smoothing algorithms.
pub mod curved;
pub mod wiggle;

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

impl Subdivision {
    /// Get the subdivision level.
    pub fn level(&self) -> u32 {
        match self {
            Subdivision::Curved(level) | Subdivision::Wiggly(level) => *level,
            Subdivision::None => 0,
        }
    }
}

/// The kind of subdivision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivisionKind {
    Curved,
    Wiggly,
}

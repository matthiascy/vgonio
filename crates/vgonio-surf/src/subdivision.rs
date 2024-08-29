//! Surface smoothing algorithms.
pub mod curved;
pub mod wiggle;

/// Subdivision scheme with a level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subdivision {
    /// Subdivide the surface using curved PN triangles (3D Bezier triangles).
    Curved(u32),
    /// Regularly subdivide the surface, adding variation to the height of new
    /// vertices.
    Wiggly(u32),
}

impl Subdivision {
    /// Get the subdivision level.
    pub fn level(&self) -> u32 {
        match self {
            Subdivision::Curved(level) | Subdivision::Wiggly(level) => *level,
        }
    }
}

/// The kind of subdivision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SubdivisionKind {
    Curved,
    Wiggly,
}

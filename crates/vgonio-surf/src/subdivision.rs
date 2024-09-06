//! Surface smoothing algorithms.

pub mod curved;
pub mod wiggle;

/// Subdivision scheme with a level.
///
/// The level is the number of extra vertices to add along the edges of the
/// triangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subdivision {
    /// Subdivide the surface using curved PN triangles (3D Bezier triangles).
    Curved(u32),
    /// Regularly subdivide the surface, adding variation to the height of new
    /// vertices.
    /// Optionally, the offset can be specified to add randomness to the height
    /// of the new vertices it's a percentage value.
    /// The actual offset is calculated as `offset * √ (du^2, dv^2)`, where
    /// `du` and `dv` are the horizontal and vertical distances of the sampled
    /// point from the original heightfield.
    /// The default offset is 100%.
    Wiggly {
        /// The level of subdivision.
        level: u32,
        /// The offset to add randomly to the z coordinate of the new points.
        offset: u32,
    },
}

impl Subdivision {
    /// Get the subdivision level.
    pub fn level(&self) -> u32 {
        match self {
            Subdivision::Curved(level) | Subdivision::Wiggly { level, .. } => *level,
        }
    }

    /// Get the kind of subdivision.
    pub fn kind(&self) -> SubdivisionKind {
        match self {
            Subdivision::Curved(_) => SubdivisionKind::Curved,
            Subdivision::Wiggly { .. } => SubdivisionKind::Wiggly,
        }
    }

    pub fn offset(&self) -> Option<u32> {
        match self {
            Subdivision::Curved(_) => None,
            Subdivision::Wiggly { offset, .. } => Some(*offset),
        }
    }
}

/// The kind of subdivision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SubdivisionKind {
    /// Subdivide the surface using curved PN triangles (3D Bezier triangles).
    Curved,
    /// Regularly subdivide the surface, adding variation to the height of new
    /// vertices.
    Wiggly,
}

//! Geometric and physical optics related computations and data structures.

pub mod fresnel;
pub mod ior;

use base::math::Vec3A;

/// Orients a vector to point away from a surface as defined by its
/// normal.
pub fn face_forward(n: Vec3A, i: Vec3A) -> Vec3A {
    if n.dot(i) < 0.0 {
        n
    } else {
        -n
    }
}

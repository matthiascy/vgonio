//! Implementation of optics-related computations and data structures.

mod fresnel;
mod refractive_index;

pub use fresnel::*;
use glam::Vec3A;
pub use refractive_index::*;

/// Reflects a vector `v` about a normal `n`.
///
/// # Arguments
///
/// * `i` - Vector to be reflected. Ends up on the point of incidence.
/// * `n` - Normal vector.
#[inline(always)]
pub fn reflect_i(i: Vec3A, n: Vec3A) -> Vec3A { i - 2.0 * i.dot(n) * n }

/// Reflects a viewing vector `v` about a normal `n`.
///
/// # Arguments
///
/// * `v` - Viewing vector. Starts from the point of incidence.
/// * `n` - Normal vector.
#[inline(always)]
pub fn reflect_v(v: Vec3A, n: Vec3A) -> Vec3A { 2.0 * v.dot(n) * n - v }

/// Reflects a vector `v` about a normal `n` with the cosine between `v` and `n`
/// already known.
///
/// # Arguments
///
/// * `i` - Vector to be reflected. Ends up on the point of incidence.
/// * `n` - Normal vector.
#[inline(always)]
pub fn reflect_i_cos(i: Vec3A, n: Vec3A, cos_theta: f32) -> Vec3A { i - 2.0 * cos_theta * n }

/// Reflects a viewing vector `v` about a normal `n` with the cosine between `v`
/// and `n` already known.
///
/// # Arguments
///
/// * `v` - Viewing vector. Starts from the point of incidence.
/// * `n` - Normal vector.
#[inline(always)]
pub fn reflect_v_cos(v: Vec3A, n: Vec3A, cos_theta: f32) -> Vec3A { 2.0 * cos_theta * n - v }

// /// Refracts a viewing vector `v` about a normal `n` using the relative
// /// refraction index `eta`. `eta` is the refraction index of outside medium
// /// (where `n` is pointing) divided by the refraction index of the inside
// /// medium. The vectors `v` and `n` are pointing towards the same side of the
// /// surface. The cosine between `v` and `n` is given as input and the cosine
// of /// `-n` and transmission ray is computed as output.
// pub fn refract_v();

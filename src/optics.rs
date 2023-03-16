//! Geometric and physical optics related computations and data structures.

mod fresnel;
mod refractive_index;

pub use fresnel::*;
use glam::Vec3A;
pub use refractive_index::*;

/// Reflects a vector `i` about a normal `n`.
///
/// # Arguments
///
/// * `i` - Incident direction (normalised), ends up on the point of incidence.
/// * `n` - Normal vector (normalised).
///
/// # Notes
///
/// The direction of the incident ray `i` is determined from the point of
/// intersection rather than the ray's origin.
#[inline(always)]
pub fn reflect_i(i: Vec3A, n: Vec3A) -> Vec3A { w_i - 2.0 * i.dot(n) * n }

/// Reflects a viewing vector `v` about a normal `n`.
///
/// # Arguments
///
/// * `v` - Viewing direction (normalised), starts from the point of incidence.
/// * `n` - Normal vector (normalised).
///
/// # Notes
///
/// The direction of the viewing ray `v` is determined from the ray's origin
/// rather than the point of intersection.
#[inline(always)]
pub fn reflect_v(v: Vec3A, n: Vec3A) -> Vec3A { 2.0 * v.dot(n) * n - v }

/// Reflects a vector `i` about a normal `n` with the cosine between `i` and `n`
/// already known.
///
/// # Arguments
///
/// * `i` - Vector to be reflected (normalised), ends up on the point of
///   incidence.
/// * `n` - Normal vector (normalised).
/// * `cos_i` - Cosine of the angle between `i` and `n`. This will always be
///   negative because `i` ends up on the point of incidence.
///
/// # Notes
///
/// The direction of the incident ray `i` is determined from the point of
/// intersection rather than the ray's origin.
#[inline(always)]
pub fn reflect_i_cos(i: Vec3A, n: Vec3A, cos_i: f32) -> Vec3A {
    debug_assert!(cos_i <= 0.0 && cos_i >= -1.0);
    i - 2.0 * cos_i * n
}

/// Reflects a viewing vector `v` about a normal `n` with the cosine between `v`
/// and `n` already known.
///
/// # Arguments
///
/// * `v` - Viewing vector (normalised), starts from the point of incidence.
/// * `n` - Normal vector (normalised).
/// * `cos_theta` - Cosine of the angle between `v` and `n`. This will always be
///   positive because `v` starts from the point of incidence.
#[inline(always)]
pub fn reflect_v_cos(v: Vec3A, n: Vec3A, cos_i: f32) -> Vec3A {
    debug_assert!(cos_i >= 0.0 && cos_i <= 1.0);
    2.0 * cos_i * n - v
}

/// Result of a refraction computation.
pub enum RefractionResult {
    TotalInternalReflection {
        /// Direction of the reflected ray.
        dir_r: Vec3A,
        /// Cosine of the reflected angle. It's the angle between the reflected
        /// direction and the inverse of the normal. This will always be
        /// positive.
        cos_r: f32,
    },
    Refraction {
        /// Direction of the refracted ray. It's the direction of the
        /// transmitted ray. In the case of total internal reflection,
        /// this will be the reflected direction.
        dir_t: Vec3A,
        /// Cosine of the transmitted angle. It's the angle between the
        /// refracted direction and the inverse of the normal. This will
        /// always be positive.
        cos_t: f32,
    },
}

/// Refracts an viewing vector `v` at a normal `n` using the relative
/// refraction index.
///
/// # Arguments
///
/// * `v` - Incident vector (normalised), starts from the point of incidence.
/// * `n` - Normal vector (normalised), always points towards the outside of the
///   incident medium.
/// * `eta` - Relative refraction index, the refractive index of outside medium
///   (where `n` is pointing, primary medium or incident medium) divided by the
///   refractive index of the inside medium (secondary medium, transmitted
///   medium), i.e. `eta = n_i / n_t`, where `n_i` is the refractive index of
///   the incident medium and `n_t` is the refractive index of the transmitted
///   medium.
/// * `cos_i` - Cosine of the incident angle, NOT the angle between `v` and `n`.
///   Should always be positive.
/// # Notes
///
/// The vectors `v` and `n` are pointing towards the same side of the surface.
/// The cosine between `v` and `n` is given as input and the cosine of `-n` and
/// transmission ray is computed as output.
pub fn refract_v_cos_relative(v: Vec3A, n: Vec3A, eta: f32, cos_i: f32) -> RefractionResult {
    debug_assert!(
        cos_i >= 0.0 && cos_i <= 1.0,
        "cos_i should be the cosine of the incident angle and should be positive"
    );
    let cos_t_sqr = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

    if cos_t_sqr < 0.0 {
        // Total internal reflection.
        let cos_r = cos_i;
        let dir_r = reflect_v_cos(v, n, cos_r);
        RefractionResult::TotalInternalReflection { dir_r, cos_r }
    } else {
        // Refraction.
        let cos_t = cos_t_sqr.sqrt();
        let dir_t = (eta * cos_i - cos_t) * n - eta * v;
        RefractionResult::Refraction { dir_t, cos_t }
    }
}

/// Refracts an incident vector `i` at a normal `n` using the relative
/// refraction index.
///
/// # Arguments
///
/// * `i` - Incident vector (normalised), ends up on the point of incidence.
/// * `n` - Normal vector (normalised), always points towards the outside of the
///   incident medium.
/// * `eta` - Relative refraction index, the refractive index of outside medium
///   (where `n` is pointing, primary medium or incident medium) divided by the
///   refractive index of the inside medium (secondary medium, transmitted
///   medium), i.e. `eta = n_i / n_t`, where `n_i` is the refractive index of
///   the incident medium and `n_t` is the refractive index of the transmitted
///   medium.
/// * `cos_i` - Cosine of the incident angle, NOT the angle between `i` and `n`.
///   Should always be positive.
/// # Notes
///
/// The vectors `i` and `n` are not pointing towards the same side of the
/// surface. The cosine of the incident angle, equals to the negative of the
/// cosine between `i` and `n` is given as input and the cosine of `-n` and
/// transmission ray is computed as output.
pub fn refract_i_cos_relative(i: Vec3A, n: Vec3A, eta: f32, cos_i: f32) -> RefractionResult {
    debug_assert!(
        cos_i >= 0.0 && cos_i <= 1.0,
        "cos_i should be the cosine of the incident angle and should be positive."
    );
    let cos_t_sqr = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

    if cos_t_sqr < 0.0 {
        // Total internal reflection.
        let cos_r = cos_i;
        let dir_r = reflect_i_cos(i, n, cos_r);
        RefractionResult::TotalInternalReflection { dir_r, cos_r }
    } else {
        // Refraction.
        let cos_t = cos_t_sqr.sqrt();
        let dir_t = (eta * cos_i - cos_t) * n - eta * i;
        RefractionResult::Refraction { dir_t, cos_t }
    }
}

pub fn refract_i(i: Vec3A, n: Vec3A, eta_i: f32, eta_t: f32) -> RefractionResult { todo!() }

pub fn refract_v(v: Vec3A, n: Vec3A, eta_i: f32, eta_t: f32) -> RefractionResult { todo!() }

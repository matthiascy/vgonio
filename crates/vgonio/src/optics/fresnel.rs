//! Fresnel equations and related functions.
//!
//! # Reflection
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../misc/imgs/reflect.svg"))]
//!
//! See [`reflect`] and [`reflect_cos_abs`] for computing the reflection of a
//! vector with respect to a surface normal.
//!
//! ## Angle of Incidence
//!
//! The angle of incidence of a ray to a surface is measured as the difference
//! in angle between the ray and the normal vector of the surface at the point
//! of incidence.
//!
//! ## Angle of Reflection
//!
//! The angle of reflection of a ray from a surface is measured as the
//! difference in angle between the reflected ray and the normal vector of the
//! surface at the point of reflection.
//!
//! # Refraction
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../misc/imgs/refract.svg"))]
//!
//! See [`refract`] for computing the refraction of a vector with respect to a
//! surface normal without knowing if the ray is located outside the surface.
//!
//! See [`refract2`] for computing the refraction of a vector with respect to a
//! surface normal knowing that the ray is located outside the surface.
//!
//! See [`refract_cos`] for computing the refraction of a vector with respect to
//! a surface normal knowing the cosine of the incident angle, the relative
//! index of refraction and that the ray is located outside the surface.
//!
//! # Fresnel equations
//!
//! Fresnel equations describe the amount of light reflected from a surface;
//! they are solution to Maxwell's equations at smooth surfaces.
//!
//! The fraction of the incident power that is reflected from the interface is
//! called the reflectance (or "reflectivity", or "power reflection
//! coefficient"), and the fraction that is referred into the second medium is
//! called the transmittance (or "transmissivity", or "power transmission
//! coefficient")
//!
//! Given the index of refraction and the angle which the incident ray makes
//! with the surface normal, the Fresnel equations specify the materials'
//! corresponding reflectance for two different polarisation states of the
//! incident light.
//!
//! If the we make the assumption that light is un-polarized (randomly oriented
//! with respect to the light wave), the `Fresnel reflectance` is the average of
//! the squares of the _parallel_ and _perpendicular_ polarisation terms.
//!
//! ## "S" polarisation
//!
//! "S" polarisation (electric field) is the perpendicular polarisation, and it
//! sticks up out of the plane of incidence (the plane that contains the
//! incident and reflected rays). "S" comes from senkrecht, German for
//! perpendicular.
//!
//! + reflection and transmission coefficients for perpendicularly polarized
//!   light at the interface of two *dielectric* media:
//!
//!   $$r_\perp=
//! \frac{\eta_i\cos\theta_i -
//! \eta_t\cos\theta_t}{\eta_i\cos\theta_i + \eta_t\cos\theta_t}$$
//!
//!   $$t_\perp=\frac{2\eta_i\cos\theta_i}{\eta_i\cos\theta_i +
//! \eta_t\cos\theta_t}$$
//!
//! + reflection and transmission coefficients for perpendicularly polarized
//!   light at the interface between a *conductor* and a *dielectric* medium:
//!
//!   $$r_\perp=\frac{a^2+b^2-2a\cos\theta+\cos^2\theta}{a^2+b^2+2a\cos\theta+\
//! cos^2\theta}$$
//!
//!   where
//!
//!   $$a^2 + b^2 = \sqrt{(\eta^2-k^2-\sin^2\theta)^2+4\eta^2k^2}$$
//!
//! ## "P" polarisation
//!
//! "P" polarisation (electric field) is the parallel polarisation, and it lies
//! parallel to the plane of incidence.
//!
//! + reflection and transmission coefficients for parallel polarized light at
//! the interface of two *dielectric* media are related by the formula:
//!
//!   $$r_\parallel =
//! \frac{\eta_t\cos\theta_i-\eta_i\cos\theta_t}{\eta_t\cos\theta_i +
//! \eta_i\cos\theta_t}$$
//!
//!   $$t_\parallel = \frac{2\eta_i\cos\theta_i}{\eta_t\cos\theta_i +
//! \eta_i\cos\theta_t}$$
//!
//! + reflection and transmission coefficients for parallel polarized light at
//!   the interface between a *conductor* and a *dielectric* medium:
//!
//!   $$r_\parallel=r_\perp\frac{\cos^2\theta(a^2+b^2)-2a\cos\theta\sin^2\
//! theta+\ sin^4\theta}{\cos^2\theta(a^2+b^2)-2a\cos\theta\sin^2\theta+\sin^4\
//! theta}$$
//!
//!   where
//!
//!   $$a^2 + b^2 = \sqrt{(\eta^2-k^2-\sin^2\theta)^2+4\eta^2k^2}$$
//!
//! and $\eta + ik = \eta_t / \eta_i$ is the relative index of refraction
//! computed using a complex division operation. Generally $\eta_i$ will be a
//! dielectric so that a normal real division can be used instead.
//! See [`reflectance_dielectric_conductor`] for details.
//!
//! For both polarisations: $\eta_i\sin\theta_i = \eta_t\sin\theta_t$.
//!
//! The cosine terms should all be greater than or equal to zero; for the
//! purposes of computing these values, the geometric normal should be flipped
//! to be on the same side as $\omega_i$ and $\omega_t$ when computing
//! $\cos\theta_i$ and $\cos\theta_t$ respectively.
//!
//! For unpolarised light, the reflectance is the average of the squares of the
//! parallel and perpendicular terms:
//!
//! $F_r = \frac{1}{2}(r_\parallel^2 + r_\perp^2)$
//!
//! Due to conservation of energy, the energy transmitted by a dielectric is
//! $1 - F_r$.
//!
//! The cosine terms should all be greater than or equal to zero;
//! for the purpose of computation, the geometric normal should be
//! flipped to be on the same side as $w_i$ and $w_t$ when computing
//! $cos_{\theta_i}$ and $cos_{\theta_t}$.
//!
//! ## refractive indices
//!
//! + Dielectrics/Insulators Dielectrics dont' conduct electricity, they have
//!   real-valued indices of refraction and transmit a portion of the incident
//!   illumination.
//!
//! + Conductors In contrast to dielectrics, conductors have a complex-valued
//!   index of refraction ῆ = η + ik
//!
//! Give the incident direction $\omega_i$, and indices of refraction of two
//! mediums, compute the reflection coefficients for waves parallel and
//! perpendicular to the plane of incidence.
//!
//! ## Schlick's approximation
//!
//! Schlick proposes the following equation for reflectance:
//!
//! $$R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5$$
//!
//! where
//!
//! $$R_0 = (\frac{\eta_i - \eta_t}{\eta_i + \eta_t})^2$$
//!
//! However, this approximation fails to model the reflectance when
//! $\eta_i > \eta_t$. This can be fixed by using $\cos\theta_t$ instead of
//! $\cos\theta_i$.
//!
//! See [`reflectance_schlick_approx`].

// TODO: unify fresnel calculation (using complex refractive index).

use crate::optics::ior::RefractiveIndex;
use base::{math, math::Vec3A};

/// Reflects a vector `wi` with respect to surface normal `n`.
///
/// # Arguments
///
/// * `wi` - Incident direction (normalised), ends up on the point of incidence.
/// * `n` - Normal vector (normalised).
///
/// # Notes
///
/// The direction of the incident ray `wi` is determined from ray's origin
/// rather than intersection point.
#[inline(always)]
pub fn reflect(wi: Vec3A, n: Vec3A) -> Vec3A {
    debug_assert!(
        wi.dot(n) <= 0.0,
        "wi should point towards the point of incidence"
    );
    wi - 2.0 * wi.dot(n) * n
}

/// Reflects a vector `wi` about a normal `n` with the cosine of incident angle
/// already known.
///
/// # Arguments
///
/// * `wi` - Vector to be reflected (normalised), ends up on the point of
///   incidence.
/// * `n` - Normal vector (normalised).
/// * `cos` - Cosine of the incident angle, should always be positive; this is
///   *NOT* the angle between the two vectors. It should be the absolute value
///   of the cosine of the angle between `wi` and `n`.
///
/// # Notes
///
/// The direction of the incident ray `i` is determined from ray's origin rather
/// than the point of intersection.
#[inline(always)]
pub fn reflect_cos_abs(wi: Vec3A, n: Vec3A, cos_abs: f32) -> Vec3A {
    debug_assert!((0.0..=1.0).contains(&cos_abs));
    wi + 2.0 * cos_abs * n
}

/// Result of a refraction computation.
pub enum RefractionResult {
    /// Total internal reflection, the ray is reflected.
    TotalInternalReflection,
    /// Refraction, the ray is partially reflected and partially refracted.
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

/// Refracts an incident vector `wi` at the surface with a normal `n` using the
/// relative refraction index.
///
/// Use this function when you are sure that the incident vector `wi` is at the
/// same side of the surface as the normal `n`; the relative refraction index
/// should be the ratio of the refractive indices of the incident medium
/// (outside transmitted medium) over the transmitted medium (inside
/// transmitted medium).
///
/// # Arguments
///
/// * `wi` - Incident vector (normalised), ends up on the point of incidence.
/// * `n` - Normal vector (normalised), always points towards the outside of the
///   incident medium.
/// * `eta` - Relative refraction index, the refractive index of outside medium
///   (where `n` is pointing, primary medium or incident medium) divided by the
///   refractive index of the inside medium (secondary medium, transmitted
///   medium), i.e. `eta = eta_i / eta_t`, where `eta_i` is the refractive index
///   of the incident medium and `eta_t` is the refractive index of the
///   transmitted medium.
/// * `cos_i` - Cosine of the incident angle, NOT the angle between `wi` and
///   `n`. Should always be positive.
///
/// # Notes
///
/// The vectors `wi` and `n` are at the same side of the boundary.
/// The cosine of the incident angle, equals to the negative of the
/// cosine between `wi` and `n` is given as input and the cosine of `-n` and
/// transmission ray (angle of transmittance) is computed as output.
pub fn refract_cos(wi: Vec3A, n: Vec3A, eta: f32, cos: f32) -> RefractionResult {
    debug_assert!(
        (0.0..=1.0).contains(&cos),
        "cos_i should be the cosine of the incident angle and should be positive."
    );
    let cos_t_sqr = 1.0 - eta * eta * (1.0 - cos * cos);

    if cos_t_sqr < 0.0 {
        RefractionResult::TotalInternalReflection
    } else {
        // Refraction.
        let cos_t = cos_t_sqr.sqrt();
        let dir_t = (eta * cos - cos_t) * n + eta * wi;
        RefractionResult::Refraction { dir_t, cos_t }
    }
}

/// Refracts `wi` with respect to a given surface normal `n` and the refractive
/// indices of the incident and transmitted media.
///
/// This function assumes that the incident vector `wi` is at the same side of
/// the surface as the normal `n`, i.e. the incident vector `wi` is pointing
/// towards the inside of the incident medium. Hence, the `eta_i` is the
/// refractive index of the incident medium and `eta_t` is the refractive index
/// of the transmitted medium. If you are not sure about the side of the
/// incident vector `wi`, use [`refract`].
///
/// # Arguments
///
/// * `wi` - Incident vector (normalised), ends up on the point of incidence.
/// * `n` - Normal vector (normalised), always points towards the outside of the
///  incident medium.
/// * `eta_i` - Refractive index of the incident medium.
/// * `eta_t` - Refractive index of the transmitted medium.
pub fn refract2(wi: Vec3A, n: Vec3A, eta_i: f32, eta_t: f32) -> RefractionResult {
    debug_assert!(
        n.dot(wi) <= 0.0,
        "The incident vector `i` should point towards the opposite side of the surface as the \
         normal `n`."
    );
    debug_assert!(
        eta_i > 0.0 && eta_t > 0.0,
        "The refractive indices should be positive and non-zero."
    );
    let cos_i = -n.dot(wi);
    refract_cos(wi, n, eta_i / eta_t, cos_i)
}

/// Refracts an incident vector `wi` at the boundary of two
/// media with the given refractive indices.
///
/// Use this function when you are NOT sure at which side of the surface the
/// incident vector `wi` lies.
///
/// # Arguments
///
/// * `wi` - Incident vector (normalised), ends up on the point of incidence.
/// * `n` - Normal vector (normalised), always pointing towards the outside of
///   the incident medium (surface).
/// * `eta_o` - Refractive index of the outside medium of the interface.
/// * `eta_i` - Refractive index of the inside medium of the interface.
///
/// # Notes
///
/// The normal vector `n` is not necessarily pointing towards the outside of the
/// incident medium. Thus, we need to check the dot product of `i` and `n` to
/// determine which side of the surface the incident vector is pointing to.
pub fn refract(wi: Vec3A, n: Vec3A, eta_o: f32, eta_i: f32) -> RefractionResult {
    debug_assert!(
        eta_o > 0.0 && eta_i > 0.0,
        "Refractive indices should be positive and non-zero."
    );
    let cos_i = n.dot(wi);
    if cos_i < 0.0 {
        // The ray is on the outside of the interface, `cos_i` is negative.
        refract_cos(wi, n, eta_o / eta_i, -cos_i)
    } else {
        // The ray is on the inside of the interface, normal and refractive indices need
        // to be flipped.
        refract_cos(wi, -n, eta_i / eta_o, cos_i)
    }
}

/// Computes the Schlick's approximation of the Fresnel specular (reflection)
/// factor.
///
/// In case of incident medium is air, `eta_i` is 1.0.
///
/// # Arguments
///
/// * `cos_i` - absolute cosine of the angle between the direction from which
///   the incident light is coming and the normal of the interface between the
///   two media.
/// * `eta_i` - refractive index of incident medium.
/// * `eta_t` - refractive index of transmitted medium.
pub fn reflectance_schlick_approx(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    debug_assert!((0.0..=1.0).contains(&cos_i), "cos_i must be in [0, 1]");
    let mut r0 = (eta_i - eta_t) / (eta_i + eta_t);

    r0 *= r0;

    let a = 1.0 - cos_i;

    r0 + (1.0 - r0) * a * a * a * a * a
}

/// Computes the Schlick's approximation of the Fresnel reflectance coefficient
/// for rays with different wavelengths(different refractive index).
///
/// See [`reflectance_schlick_approx`] for details.
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between the direction from which the
///   incident light is coming and the normal of the interface between the two
///   media.
/// * `eta_i` - slice of refractive index of incident medium.
/// * `eta_t` - slice of refractive index of transmitted medium.
///
/// # Note
///
/// The length of the slices must be the same.
pub fn reflectance_schlick_approx_spectrum(cos_i: f32, eta_i: &[f32], eta_t: &[f32]) -> Vec<f32> {
    debug_assert_eq!(
        eta_i.len(),
        eta_t.len(),
        "eta_i and eta_t must have the same length"
    );
    debug_assert!((0.0..=1.0).contains(&cos_i), "cos_i must be in [0, 1]");

    let mut output = vec![1.0; eta_i.len()];

    for (i, r) in output.iter_mut().enumerate() {
        *r = reflectance_schlick_approx(cos_i, eta_i[i], eta_t[i]);
    }

    output
}

/// Computes the unpolarised Fresnel reflection coefficient at a planar
/// interface between two dielectric materials.
///
/// This function does *NOT* assume that the incident ray is located on the
/// outside of the interface (same side as the normal). The sign of the cosine
/// of the angle between the incident ray and the normal indicates on which side
/// of the interface the incident ray is located. If the cosine is between -1
/// and 0, the ray is on the outside (same as the normal), and if the cosine is
/// between 0 and 1, the ray is on the inside (opposite to the normal).
///
/// If you are sure that the incident ray is always on the outside of the
/// interface, use [`reflectance_dielectric2`] instead.
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between the normal and the incident
///   direction (originated from the ray's origin).
/// * `eta_i` - refractive index of the incident medium.
/// * `eta_t` - refractive index of the transmitted medium.
///
/// # Note
///
/// The incident direction is always originated from the ray's origin. The sign
/// of the cosine of the incident angle indicates on which side of the medium
/// the incident ray lines. If the cosine is between -1 and 0, the ray is on
/// the outside (same as the normal), and if the cosine is between 0 and 1, the
/// ray is on the inside (opposite to the normal).
pub fn reflectance_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    debug_assert!((-1.0..=1.0).contains(&cos_i), "cos_i must be in [-1, 1]");
    let cos_i = cos_i.clamp(-1.0, 1.0);
    if cos_i < 0.0 {
        // The incident ray is on the outside of the interface entering the medium.
        reflectance_dielectric2(-cos_i, eta_i, eta_t)
    } else {
        // The incident ray is on the inside of the interface leaving the medium.
        reflectance_dielectric2(cos_i, eta_t, eta_i)
    }
}

/// Computes the unpolarised Fresnel reflection coefficient at a planar
/// interface between two dielectric materials.
///
/// This function assumes that the incident ray is located on the outside of
/// the interface (same side as the normal). If you are not sure about the
/// location of the incident ray, use [`reflectance_dielectric`] instead.
///
/// # Arguments
///
/// * `cos_i` - cosine of the incident angle, should always be positive; this is
///   *NOT* the angle between the two vectors. It should be the absolute value
///   of the cosine of the angle between `wi` and `n`.
/// * `eta_i` - refractive index of the incident medium (outside).
/// * `eta_t` - refractive index of the transmitted medium (inside).
pub fn reflectance_dielectric2(cos_i_abs: f32, eta_i: f32, eta_t: f32) -> f32 {
    debug_assert!(
        (0.0..=1.0).contains(&cos_i_abs),
        "cos_i_abs must be in [0, 1]"
    );

    // Compute the angle between the normal and the transmitted direction.
    let sin_i = (1.0 - cos_i_abs * cos_i_abs).sqrt();
    let sin_t = eta_i / eta_t * sin_i;

    // Handle total internal reflection.
    if sin_t >= 1.0 {
        return 1.0;
    }

    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let r_parl =
        (eta_t * cos_i_abs - eta_i * cos_t) * math::rcp_f32(eta_t * cos_i_abs + eta_i * cos_t);
    let r_perp =
        (eta_i * cos_i_abs - eta_t * cos_t) * math::rcp_f32(eta_i * cos_i_abs + eta_t * cos_t);

    // No polarisation.
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Computes the unpolarised Fresnel reflection coefficient at a planar
/// interface between two dielectric materials for rays with different
/// wavelengths.
///
/// See [`reflectance_dielectric`] for details.
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between the normal and the incident
///   direction (originated from the ray's origin).
/// * `eta_i` - slice of refractive index of incident medium.
pub fn reflectance_dielectric_spectrum(cos_i: f32, eta: &[f32]) -> Vec<f32> {
    debug_assert!((-1.0..=1.0).contains(&cos_i), "cos_i must be in [-1, 1]");
    let mut output = vec![1.0; eta.len()];
    for (i, r) in output.iter_mut().enumerate() {
        *r = reflectance_dielectric(cos_i, eta[i], eta[i]);
    }
    output
}

// TODO: from conductor to dielectric

/// Fresnel reflectance of unpolarised light between dielectric and conductor.
///
/// Modified from "Optics" by K.D. Moeller, University Science Books, 1988
///
/// # Arguments
///
/// * `cos_i` - cosine of the incident angle (should always be positive).
/// * `eta_i` - refractive index of the incident medium.
/// * `eta_t` - refractive index of the transmitted medium.
/// * `k` - absorption coefficient of the transmitted medium.
pub fn reflectance_dielectric_conductor(cos_i_abs: f32, eta_i: f32, eta_t: f32, k_t: f32) -> f32 {
    assert!(
        (-f32::EPSILON..=1.0 + f32::EPSILON).contains(&cos_i_abs),
        "the cosine of the incident angle should always be positive {}",
        cos_i_abs
    );
    // Computes relative index of refraction.
    let eta = eta_t / eta_i;
    let k = k_t / eta_i;

    let cos_i_2 = cos_i_abs * cos_i_abs;
    let sin_i_2 = 1.0 - cos_i_2;
    let eta_2 = eta * eta;
    let k_2 = k * k;
    let t0 = eta_2 - k_2 - sin_i_2;
    let a2_plus_b2 = (t0 * t0 + 4.0 * k_t * k_t * eta_t * eta_t).sqrt();
    let t1 = a2_plus_b2 + cos_i_2;
    let a = (0.5 * (a2_plus_b2 + t0)).sqrt();
    let t2 = 2.0 * a * cos_i_abs;
    let rs = (t1 - t2) / (t1 + t2);
    let t3 = a2_plus_b2 * cos_i_2 + sin_i_2 * sin_i_2;
    let t4 = t2 * sin_i_2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    0.5 * (rp + rs)
}

/// Computes the unpolarised Fresnel reflection coefficient at a planar
/// interface between two unknown media.
///
/// This function is a wrapper around [`reflectance_dielectric`] and
/// [`reflectance_dielectric_conductor`].
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between the normal and the incident
///  direction (originated from the ray's origin).
/// * `ior_i` - refractive index of the outside medium.
/// * `ior_t` - refractive index of the inside medium.
///
/// # Notes
///
/// Because reflectance from a conductor to a dielectric is not implemented yet,
/// thus `cos_i` must be positive in case from a dielectric to a conductor.
pub fn reflectance(cos_i: f32, ior_i: RefractiveIndex, ior_t: RefractiveIndex) -> f32 {
    debug_assert!(
        (-1.0 - f32::EPSILON..=1.0 + f32::EPSILON).contains(&cos_i),
        "cos_i must be in [-1, 1]"
    );
    match (ior_i.is_dielectric(), ior_t.is_dielectric()) {
        // Both are dielectrics.
        (true, true) => reflectance_dielectric(cos_i, ior_i.eta, ior_t.eta),
        // One is a dielectric and the other is a conductor (entering the conductor).
        (true, false) => {
            reflectance_dielectric_conductor(cos_i.abs(), ior_i.eta, ior_t.eta, ior_t.k)
        }
        // One is a conductor and the other is a dielectric.
        (false, true) => {
            unimplemented!("reflectance from a conductor to a dielectric is not implemented")
        }
        // Both are conductors.
        (false, false) => unimplemented!("reflectance between two conductors is not implemented"),
    }
}

/// Computes the unpolarised Fresnel reflectance of unpolarised light between
/// dielectric and conductor medium for rays with different wavelengths.
///
/// See [`reflectance_dielectric_conductor`] for details.
///
/// # Arguments
///
/// * `cos` - cosine of the incident angle (should always be positive).
pub fn reflectance_dielectric_conductor_spectrum(
    cos: f32,
    eta_i: f32,
    ior_t: &[RefractiveIndex],
) -> Vec<f32> {
    assert!(
        (0.0..=1.0).contains(&cos),
        "the angle between normal and incident light should be positive"
    );
    let mut output = vec![1.0; ior_t.len()];
    for (i, r) in output.iter_mut().enumerate() {
        *r = reflectance_dielectric_conductor(cos, eta_i, ior_t[i].eta, ior_t[i].k);
    }
    output
}

#[cfg(test)]
mod tests {
    use crate::optics::ior::RefractiveIndex;
    use base::units::nm;
    use std::{fs::OpenOptions, io::Write};

    #[test]
    fn reflectance_dielectric_test() {
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("./reflectance_air_to_glass.csv")
                .unwrap();
            file.write_all(b"angle,reflectance\n").unwrap();
            for i in 0..1000 {
                let cos_i = (1000 - i) as f32 / 1000.0;
                let angle = cos_i.acos();
                let eta_i = 1.0; // air
                let eta_t = 1.5; // glass
                let r = super::reflectance_dielectric2(cos_i, eta_i, eta_t);
                file.write_all(format!("{},{}\n", angle.to_degrees(), r).as_bytes())
                    .unwrap();
            }
        }
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("./reflectance_air_to_glass_inv.csv")
                .unwrap();
            file.write_all(b"angle,reflectance\n").unwrap();
            for i in 0..1000 {
                let cos_i = (1000 - i) as f32 / 1000.0;
                let angle = cos_i.acos();
                let eta_i = 1.0; // air
                let eta_t = 1.5; // glass
                let r = super::reflectance_dielectric(cos_i, eta_i, eta_t);
                file.write_all(format!("{},{}\n", angle.to_degrees(), r).as_bytes())
                    .unwrap();
            }
        }
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("./reflectance_glass_to_air.csv")
                .unwrap();
            file.write_all(b"angle,reflectance\n").unwrap();
            for i in 0..1000 {
                let cos_i = (1000 - i) as f32 / 1000.0;
                let angle = cos_i.acos();
                let eta_t = 1.0; // air
                let eta_i = 1.5; // glass
                let r = super::reflectance_dielectric2(cos_i, eta_i, eta_t);
                file.write_all(format!("{},{}\n", angle.to_degrees(), r).as_bytes())
                    .unwrap();
            }
        }
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open("./reflectance_glass_to_air_inv.csv")
                .unwrap();
            file.write_all(b"angle,reflectance\n").unwrap();
            for i in 0..1000 {
                let cos_i = (1000 - i) as f32 / 1000.0;
                let angle = cos_i.acos();
                let eta_t = 1.0; // air
                let eta_i = 1.5; // glass
                let r = super::reflectance_dielectric(cos_i, eta_i, eta_t);
                file.write_all(format!("{},{}\n", angle.to_degrees(), r).as_bytes())
                    .unwrap();
            }
        }
    }

    #[test]
    fn reflectance_dielectric_conductor_test() {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("./reflectance_dielectric_conductor.csv")
            .unwrap();
        file.write_all(b"angle,reflectance\n").unwrap();
        for i in 0..1000 {
            let cos_i = (1000 - i) as f32 / 1000.0;
            let angle = cos_i.acos();
            let eta_i = 1.0; // air
            let eta_t = 1.1893; // al, 600nm
            let k_t = 6.9762; // al, 600nm
            let r = super::reflectance_dielectric_conductor(cos_i, eta_i, eta_t, k_t);
            file.write_all(format!("{},{}\n", angle.to_degrees(), r).as_bytes())
                .unwrap();
        }
    }

    #[test]
    fn reflectance_no_output() {
        let ior_vacuum = RefractiveIndex::VACUUM;
        let ior_al = RefractiveIndex::new(nm!(600.0), 1.1893, 6.9762);
        let ior_glass = RefractiveIndex::new(nm!(600.0), 1.5, 0.0);
        for i in 0..1000 {
            let cos_i = (1000 - i) as f32 / 1000.0;
            let angle = cos_i.acos();
            let r0 = super::reflectance_dielectric_conductor(
                cos_i,
                ior_vacuum.eta,
                ior_al.eta,
                ior_al.k,
            );
            let r1 = super::reflectance(cos_i, ior_vacuum, ior_al);
            assert!(
                (r0 - r1).abs() < 0.0001,
                "angle: {}, r0: {}, r1: {}",
                angle,
                r0,
                r1
            );

            let r2 = super::reflectance(cos_i, ior_vacuum, ior_glass);
            let r3 = super::reflectance_dielectric(cos_i, ior_vacuum.eta, ior_glass.eta);
            assert!(
                (r2 - r3).abs() < 0.0001,
                "angle: {}, r2: {}, r3: {}",
                angle,
                r2,
                r3
            );
        }
    }
}

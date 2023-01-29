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
//! corresponding reflectance for two different polarization states of the
//! incident light.
//!
//! If the we make the assumption that light is un-polarized (randomly oriented
//! with respect to the light wave), the `Fresnel reflectance` is the average of
//! the squares of the _parallel_ and _perpendicular_ polarization terms.
//!
//! ## "S" and "P" polarization
//!
//! "S" polarization (electric field) is the perpendicular polarization, and it
//! sticks up out of the plane of incidence (the plane that contains the
//! incident and reflected rays). "S" comes from senkrecht, German for
//! perpendicular.
//!
//! + reflection and transmission coefficients for perpendicularly polarized
//! light at the interface of two *dielectric* media:
//!
//!   $r_\perp=\frac{\eta_i\cos\theta_i - \eta_t\cos\theta_t}{\eta_i\cos\theta_i
//! + \eta_t\cos\theta_t}$
//!
//!   $t_\perp=\frac{2\eta_i\cos\theta_i}{\eta_i\cos\theta_i +
//! \eta_t\cos\theta_t}$
//!
//! + reflection and transmission coefficients for perpendicularly polarized
//! light at the interface between a *conductor* and a *dielectric* medium:
//!
//!   $r_\perp=\frac{a^2+b^2-2a\cos\theta+\cos^2\theta}{a^2+b^2+2a\cos\theta+\
//! cos^2\theta}$
//!
//! "P" polarization (electric field) is the parallel polarization, and it lies
//! parallel to the plane of incidence.
//!
//! + reflection and transmission coefficients for parallel polarized light at
//! the interface of two *dielectric* media are related by the formula:
//!
//!   $r_\parallel =
//! \frac{\eta_t\cos\theta_i-\eta_i\cos\theta_t}{\eta_t\cos\theta_i +
//! \eta_i\cos\theta_t}$
//!
//!   $t_\parallel = \frac{2\eta_i\cos\theta_i}{\eta_t\cos\theta_i +
//! \eta_i\cos\theta_t}$
//!
//! + reflection and transmission coefficients for parallel polarized
//! light at the interface between a *conductor* and a *dielectric* medium:
//!
//!   $r_\parallel=r_\perp\frac{\cos^2\theta(a^2+b^2)-2a\cos\theta\sin^2\theta+\
//! sin^4\theta}{\cos^2\theta(a^2+b^2)-2a\cos\theta\sin^2\theta+\sin^4\theta}$
//!
//! where
//!
//!   $a^2 + b^2 = \sqrt{(\eta^2-k^2-\sin^2\theta)^2+4\eta^2k^2}$
//!
//! and $\eta + ik = \eta_t / \eta_i$ is the relative index of refraction
//! computed using a complex division operation. Generally $\eta_i$ will be a
//! dielectric so that a normal real division can be used instead.
//! See [`reflectance_dielectric_conductor`] for details.
//!
//! For both polarizations: $\eta_i\sin\theta_i = \eta_t\sin\theta_t$.
//!
//! The cosine terms should all be greater than or equal to zero; for the
//! purposes of computing these values, the geometric normal should be flipped
//! to be on the same side as $\omega_i$ and $\omega_t$ when computing
//! $\cos\theta_i$ and $\cos\theta_t$ respectively.
//!
//! For unpolarized light, the reflectance is the average of the squares of the
//! parallel and perpendicular terms:
//!
//! $F_r = \frac{1}{2}(r_\parallel^2 + r_\perp^2)$
//!
//! Due to conservation of energy, the energy transmitted by a dielectric is $1
//! - F_r$.
//!
//! The cosine terms should all be greater than or equal to zero;
//! for the purpose of computation, the geometric normal should be
//! flipped to be on the same side as w_i and w_t when computing
//! cos_theta_i and cos_theta_t.
//!
//! ## refractive indices
//!
//! + Dielectrics/Insulators
//!   Dielectrics dont' conduct electricity, they have real-valued indices
//!   of refraction and transmit a portion of the incident illumination.
//!
//! + Conductors
//!   In contrast to dielectrics, conductors have a complex-valued index of
//!   refraction ῆ = η + ik
//!
//! Give the incident direction $\omega_i$, and indices of refraction of two
//! mediums, compute the reflection coefficients for waves parallel and
//! perpendicular to the plane of incidence.
//!
//! ## Schlick's approximation
//!
//! Schlick proposes the following equation for reflectance:
//!
//! $R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5$
//!
//! $R_0 = (\frac{\eta_i - \eta_t}{\eta_i + \eta_t})^2$
//!
//! However, this approximation fails to model the reflectance when $\eta_i >
//! \eta_t$. This can be fixed by using $\cos\theta_t$ instead of
//! $\cos\theta_i$.

// TODO: unify fresnel calculation (using complex refractive index).

use crate::optics::RefractiveIndex;

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
    assert_eq!(
        eta_i.len(),
        eta_t.len(),
        "eta_i and eta_t must have the same length"
    );

    let mut output = vec![1.0; eta_i.len()];

    for (i, r) in output.iter_mut().enumerate() {
        *r = reflectance_schlick_approx(cos_i, eta_i[i], eta_t[i]);
    }

    output
}

/// Computes the unpolarised Fresnel reflection coefficient at a planar
/// interface between two dielectric materials.
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between the normal and the incident
///   direction.
/// * `eta_i` - refractive index of the incident medium.
/// * `eta_t` - refractive index of the transmitted medium.
///
/// # Note
///
/// The incident direction is not originated from the intersection point.
pub fn reflectance_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    // The sign of the cosine of the incident angle indicates on which side of the
    // medium the incident ray lines. If the cosine is between -1 and 0, the ray
    // is on the outside, and if the cosine is between 0 and 1, the ray is on
    // the inside. `eta_i` and `eta_t` are adjusted such that `eta_i`
    // has the refractive index of the incident medium and thus makes sure that
    // `cos_i` is non-negative.
    let (eta_i, eta_t) = if cos_i >= 0.0 {
        (eta_t, eta_i)
    } else {
        (eta_i, eta_t)
    };

    // Compute the angle between the normal and the transmitted direction.
    let sin_i = (1.0 - cos_i * cos_i).sqrt();
    let sin_t = eta_i / eta_t * sin_i;

    // Handle total internal reflection.
    if sin_t >= 1.0 {
        return 1.0;
    }

    let cos_t = (1.0 - sin_t * sin_t).sqrt();

    let r_parl = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let r_perp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);

    // No polarization.
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
///   direction.
/// * `eta_i` - slice of refractive index of incident medium.
pub fn reflectance_dielectric_spectrum(cos_i: f32, eta: &[f32]) -> Vec<f32> {
    let mut output = vec![1.0; eta.len()];
    for (i, r) in output.iter_mut().enumerate() {
        *r = reflectance_dielectric(cos_i, eta[i], eta[i]);
    }
    output
}

/// Fresnel reflectance of unpolarised light between dielectric and conductor.
///
/// Modified from "Optics" by K.D. Moeller, University Science Books, 1988
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between normal and incident light (should be
///   positive).
/// * `eta_i` - refractive index of the incident medium.
/// * `eta_t` - refractive index of the transmitted medium.
/// * `k` - absorption coefficient of the transmitted medium.
pub fn reflectance_dielectric_conductor(cos_i: f32, eta_i: f32, eta_t: f32, k_t: f32) -> f32 {
    assert!(
        cos_i >= 0.0,
        "the angle between normal and incident light should be positive"
    );
    // Computes relative index of refraction.
    let eta = eta_t / eta_i;
    let k = k_t / eta_i;

    let cos_i_2 = cos_i * cos_i;
    let sin_i_2 = 1.0 - cos_i_2;
    let eta_2 = eta * eta;
    let k_2 = k * k;
    let t0 = eta_2 - k_2 - sin_i_2;
    let a2_plus_b2 = (t0 * t0 + 4.0 * k_t * k_t * eta_t * eta_t).sqrt();
    let t1 = a2_plus_b2 + cos_i_2;
    let a = (0.5 * (a2_plus_b2 + t0)).sqrt();
    let t2 = 2.0 * a * cos_i;
    let rs = (t1 - t2) / (t1 + t2);
    let t3 = a2_plus_b2 * cos_i_2 + sin_i_2 * sin_i_2;
    let t4 = t2 * sin_i_2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    0.5 * (rp + rs)
}

/// Computes the unpolarised Fresnel reflectance of unpolarised light between
/// dielectric and conductor medium for rays with different wavelengths.
///
/// See [`reflectance_dielectric_conductor`] for details.
///
/// # Arguments
///
/// * `cos` - cosine of the angle between normal and incident light (should be
///   positive).
pub fn reflectance_dielectric_conductor_spectrum(
    cos: f32,
    eta_i: f32,
    ior_t: &[RefractiveIndex],
) -> Vec<f32> {
    let mut output = vec![1.0; ior_t.len()];
    for (i, r) in output.iter_mut().enumerate() {
        *r = reflectance_dielectric_conductor(cos, eta_i, ior_t[i].eta, ior_t[i].k);
    }
    output
}

/// Computes the unpolarised Fresnel reflectance of unpolarised light between
/// the air and a conductor medium.
///
/// # Arguments
///
/// * `cos_i` - cosine of the angle between normal and incident light (should be
///   positive).
/// * `eta_t` - refractive index of the transmitted medium.
/// * `k_t` - absorption coefficient of the transmitted medium.
pub fn reflectance_air_conductor(cos_i: f32, eta_t: f32, k_t: f32) -> f32 {
    reflectance_dielectric_conductor(cos_i, 1.0, eta_t, k_t)
}

/// Computes the unpolarised Fresnel reflectance of unpolarised light between
/// the air and a conductor medium for rays with different wavelengths.
///
/// See [`reflectance_air_conductor`] for details.
pub fn reflectance_air_conductor_spectrum(cos: f32, ior_t: &[RefractiveIndex]) -> Vec<f32> {
    reflectance_dielectric_conductor_spectrum(cos, 1.0, ior_t)
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Write};

    #[test]
    fn reflectance_dielectric_test() {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("./reflectance_dielectric.csv")
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
            let eta_t = 1.1978; // al, 587.6nm
            let k_t = 7.0488;
            let r = super::reflectance_dielectric_conductor(cos_i, eta_i, eta_t, k_t);
            file.write_all(format!("{},{}\n", angle.to_degrees(), r).as_bytes())
                .unwrap();
        }
    }
}

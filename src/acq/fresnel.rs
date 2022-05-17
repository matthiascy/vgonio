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
//! light:
//!
//!   $r_\perp=\frac{n_i\cos\theta_i - n_t\cos\theta_t}{n_i\cos\theta_i +
//! n_t\cos\theta_t}$
//!
//!   $t_\perp=\frac{2n_i\cos\theta_i}{\n_i\cos\theta_i + n_t\cost\theta_t}$
//!
//! "P" polarization (electric field) is the parallel polarization, and it lies
//! parallel to the plane of incidence.
//!
//! + reflection and transmission coefficients for parallel polarized light:
//!
//!   $r_\parallel = \frac{n_t\cos\theta_i-n_i\cos\theta_t}{n_t\cos\theta_i +
//! n_i\cos\theta_t}$
//!
//!   $t_\parallel = \frac{2n_i\cos\theta_i}{n_t\cos\theta_i + n_i\cos\theta_t}$
//!
//! For both polarizations: $n_i\sin\theta_i = n_t\sin\theta_t$
//!
//! ## Refractive indices
//!
//! + Dielectrics/Insulators
//!   Dielectrics dont' conduct electricity, they have real-valued indices
//!   of refraction and transmit a portion of the incident illumination.
//!
//! + Conductors
//!   In contrast to dielectrics, conductors have a complex-valued index of
//!   refraction ῆ = η + ik
//!
//! Give the incident direction w_i, and indices of refraction of two mediums,
//! compute the reflection coefficients for waves parallel and perpendicular
//! to the plane of incidence.
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
//!
//! * `w_i` - normalized incident light direction.
//! * `n` - normalized surface normal.
//! * `eta_i` - index of refraction of incident medium
//! * `eta_t` - index of refraction of transmitted medium.
//!
//! Note: the cosine terms should all be greater than or equal to zero;
//!       for the purpose of computation, the geometric normal should be
//!       flipped to be on the same side as w_i and w_t when computing
//!       cos_theta_i and cos_theta_t

// #[cfg(target_arch = "x86")]
// use core::arch::x86 as arch;
//
// #[cfg(target_arch = "x86_64")]
// use core::arch::x86_64 as arch;

// TODO: unify fresnel calculation (using complex refractive index).

use crate::acq::ior::RefractiveIndex;
use glam::Vec3;

/// Compute the Schlick's approximation of the Fresnel specular (reflection)
/// factor.
///
/// # Arguments
///
/// * `eta_i` - refractive index of incident medium.
/// * `eta_t` - refractive index of transmitted medium.
/// * `n` - normalized surface normal.
/// * `w_i` - normalized incident direction
pub fn schlick_reflectance(w_i: Vec3, n: Vec3, eta_i: f32, eta_t: f32) -> f32 {
    let cos = w_i.dot(n);
    let mut r0 = (eta_i - eta_t) / (eta_i + eta_t);

    r0 *= r0;

    let a = 1.0 - cos;

    r0 + (1.0 - r0) * a * a * a * a * a
}

pub fn schlick_reflectance_spectrum(w_i: Vec3, n: Vec3, eta_i: &[f32], eta_t: &[f32]) -> Vec<f32> {
    assert_eq!(
        eta_i.len(),
        eta_t.len(),
        "eta_i and eta_t must have the same length"
    );

    let cos = w_i.dot(n);

    let mut output = Vec::with_capacity(eta_i.len());

    for (i, r) in output.iter_mut().enumerate() {
        let r0 = (eta_i[i] - eta_t[i]) / (eta_i[i] + eta_t[i]);

        let a = 1.0 - cos;

        *r = r0 + (1.0 - r0) * a * a * a * a * a;
    }

    output
}

pub fn fresnel_schlick_approx(w_i: Vec3, n: Vec3, eta_i: f32, eta_t: f32) -> f32 {
    schlick_reflectance(w_i, n, eta_i, eta_t)
}

pub fn fresnel_schlick_approx_spectrum(
    w_i: Vec3,
    n: Vec3,
    eta_i: &[f32],
    eta_t: &[f32],
) -> Vec<f32> {
    schlick_reflectance_spectrum(w_i, n, eta_i, eta_t)
}

/// Fresnel reflectance of unpolarised light between dielectric and conductor.
/// Modified from "Optics" by K.D. Moeller, Unisersity Science Books, 1988
/// `cos_theta`: noticed this is the cos of the angle between normal and
/// incident light (should be positive)
pub fn reflectance_dielectric_conductor(cos_theta: f32, eta_t: f32, k_t: f32) -> f32 {
    let cos_theta_2 = cos_theta * cos_theta;
    let sin_theta_2 = 1.0 - cos_theta_2;
    let sin_theta_4 = sin_theta_2 * sin_theta_2;

    let tmp1 = eta_t * eta_t - k_t * k_t - sin_theta_2;
    let a2pb2 = (tmp1 * tmp1 + 4.0 * k_t * k_t * eta_t * eta_t).sqrt();
    let a = (0.5 * (a2pb2 + tmp1)).sqrt();

    let term1 = a2pb2 + cos_theta_2;
    let term2 = 2.0 * a * cos_theta;

    let rs2 = (term1 - term2) / (term1 + term2);
    let term3 = a2pb2 * cos_theta_2 + sin_theta_4;
    let term4 = term2 * sin_theta_2;
    let rp2 = rs2 * (term3 - term4) / (term3 + term4);

    0.5 * (rp2 + rs2)
}

pub fn reflectance_dielectric_conductor_spectrum(
    cos_theta: f32,
    iors_t: &[RefractiveIndex],
) -> Vec<f32> {
    todo!()
}

// pub fn fresnel_schlick_approx2(w_i: &Vec3f, n: &Vec3f, eta_i: f32, eta_t:
// f32) -> FresnelCoeff {     let mut r0 = (eta_i - eta_t) / (eta_i + eta_t);
//     r0 *= r0;
//     let cos = -n.dot(w_i);
//     if eta_i > eta_t {
//         let eta = eta_i / eta_t;
//         let sin = n * n * (1.0 - )
//     }
// }

// pub fn s_polarized_reflection_coeff() ->

// fn fresnel_eq_dielectric_dielectric(w_i: &Vec3f, n: &Vec3f, eta_i: f32,
// eta_t: f32) -> FresnelPowerCoeff {     assert!(ior_i.is_same_wavelength(&
// ior_t));     unimplemented!()
// }
//
// fn fresnel_eq_dielectric_conductor(w_i: &Vec3f, n: &Vec3f, eta_i: f32, eta_t:
// f32, k_t: f32) -> FresnelPowerCoeff {     assert!(ior_i.is_same_wavelength(&
// ior_t));     unimplemented!()
// }
//
// fn fresnel_eq_conductor_dielectric(w_i: &Vec3f, n: &Vec3f, eta_i: f32, k_i:
// f32, eta_t: f32) -> FresnelPowerCoeff {     assert!(ior_i.
// is_same_wavelength(&ior_t));     unimplemented!()
// }
//
// fn fresnel_eq_conductor_conductor(w_i: &Vec3f, n: &Vec3f, eta_i: f32, k_i:
// f32, eta_t: f32, k_t: f32) -> FresnelPowerCoeff {     assert!(ior_i.
// is_same_wavelength(&ior_t));     unimplemented!()
// }

// /// Compute Fresnel equations for the case when incident medium is air and
// transmitted medium is /// a conductor.
// ///
// /// # Examples
// /// ```
// /// ```
// pub fn fresnel_eq_air_conductor(w_i: &Vec3f, n: &Vec3f, eta_t: f32, k_t: f32)
// -> FresnelCoeff {     // let cos_theta_i = w_i.dot(n).abs();
//     let cos_theta_i = unsafe {
//         let mut dot = [0.0f32; 4];
//         let a = arch::_mm_mul_ps(
//             arch::_mm_setr_ps(w_i.x, w_i.y, w_i.z, 0.0),
//             arch::_mm_setr_ps(n.x, n.y, n.z, 0.0),
//         );
//         arch::_mm_storeu_ps(dot.as_mut_ptr(), a);
//         dot[0] + dot[1] + dot[2]
//     };
//
//     let cos_theta_i_2 = cos_theta_i * cos_theta_i;
//     let two_eta_cos_theta = 2.0 * eta_t * cos_theta_i;
//
//     let t0 = eta_t * eta_t + k_t * k_t;
//     let t1 = t0 * cos_theta_i_2;
//     let r_s = (t0 - two_eta_cos_theta + cos_theta_i_2) / (t0 +
// two_eta_cos_theta + cos_theta_i_2);     let r_p = (t1 - two_eta_cos_theta +
// 1.0) / (t1 + two_eta_cos_theta + 1.0);
//
//     let reflectance = 0.5 * (r_p + r_s);
//
//     FresnelCoeff {
//         transmittance: 1.0 - reflectance,
//         reflectance,
//     }
// }

// pub fn fresnel_reflectance_between_dielectric(
//     w_i: &Vec3f,
//     n: &Vec3f,
//     eta_i: f32,
//     eta_o: f32,
// ) -> f32 {
//     let mut cos_theta_i = w_i.dot(&n);
//
//     let (eta_i, eta_t) = if cos_theta_i < 0.0 {
//         cos_theta_i = cos_theta_i.abs();
//         (eta_o, eta_i)
//     } else {
//         (eta_i, eta_o)
//     };
//
//     // Compute sin_theta_t using Snell's law.
//     let sin_theta_i = ((1.0 - cos_theta_i * cos_theta_i).max(0.0)).sqrt();
//     let sin_theta_t = eta_i / eta_t * sin_theta_i;
//
//     // Total internal reflection
//     if sin_theta_t >= 1.0 {
//         return 1.0;
//     }
//
//     let cos_theta_t = ((1.0 - sin_theta_t * sin_theta_t).max(0.0)).sqrt();
//
//     // Fresnel reflectance for p-polarized light
//     let r_p =
//         (eta_t * cos_theta_i - eta_i * cos_theta_t) / (eta_t * cos_theta_i +
// eta_i * cos_theta_t);     // Fresnel reflectance for s-polarized light
//     let r_s =
//         (eta_i * cos_theta_i - eta_t * cos_theta_t) / (eta_i * cos_theta_i +
// eta_t * cos_theta_t);
//
//     (r_p * r_p + r_s * r_s) / 2.0
// }
//
// pub fn fresnel_reflectance_dielectric_conductor(
//     w_i: &Vec3f,
//     n: &Vec3f,
//     eta_i: f32,
//     eta_t: f32,
//     k: f32,
// ) -> f32 {
//     let mut cos_i = w_i.dot(&n);
//
//     let (eta_i, eta_t) = if cos_i < 0.0 {
//         cos_i = cos_i.abs();
//         (eta_t, eta_i)
//     } else {
//         (eta_i, eta_t)
//     };
//
//     let cos2 = cos_i * cos_i;
//     let sin2 = 1.0 - cos_i * cos_i;
//     let eta = eta_t / eta_i;
//     let k = k / eta_i;
//     let eta2 = eta * eta;
//     let k2 = k * k;
//
//     // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
//     let t0 = eta2 - k2 - sin2;
//     let a2_plus_b2 = (t0 * t0 + 4.0 * eta2 * k2).sqrt();
//     let t1 = a2_plus_b2 + cos2;
//     let a = (0.5 * (a2_plus_b2 + t0)).sqrt();
//     let t2 = 2.0 * cos_i * a;
//
//     let r_s = (t1 - t2) / (t1 + t2);
//     let t3 = cos2 * a2_plus_b2 + sin2 * sin2;
//     let t4 = t2 * sin2;
//     let r_p = r_s * (t3 - t4) / (t3 + t4);
//
//     0.5 * r_p + r_s
// }

// Computes the unpolarised Fresnel reflection coefficient at a planer
// interface between two dielectrics.
//
// pub fn fresnel_dielectric(cos_i: f32, cos_t: f32, eta: f32) -> f32 {}

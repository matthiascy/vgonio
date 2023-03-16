use crate::{measure::Ray, optics, optics::RefractiveIndex};
use glam::Vec3A;

/// Light scattering result.
#[derive(Debug, Copy, Clone)]
pub struct Scattering {
    /// Reflected ray after scattering.
    pub reflected: Ray,

    /// Transmitted ray after scattering.
    pub refracted: Option<Ray>,

    /// Fresnel reflectance.
    pub reflectance: f32,
}

/// Multiple wavelengths light scattering result.
#[derive(Debug, Clone)]
pub struct ScatteringSpectrum {
    pub reflected: Ray,
    pub refracted: Vec<Option<Ray>>,
    pub reflectance: Vec<f32>,
}

/// Scattering happens between the air and the conductor.
pub fn scatter_air_conductor(ray: Ray, p: Vec3A, n: Vec3A, eta_t: f32, k_t: f32) -> Scattering {
    let reflected_dir = reflect(ray.d.into(), n);
    let refracted_dir = refract(ray.d.into(), n, 1.0, eta_t);
    let reflectance = optics::reflectance_air_conductor(ray.d.dot(n.into()).abs(), eta_t, k_t);

    Scattering {
        reflected: Ray {
            o: p.into(),
            d: reflected_dir.into(),
        },
        refracted: refracted_dir.map(|d| Ray {
            o: p.into(),
            d: d.into(),
        }),
        reflectance,
    }
}

/// Scattering happens between the air and the conductor for rays of different
/// wavelengths.
pub fn scatter_air_conductor_spectrum(
    ray: Ray,
    p: Vec3A,
    n: Vec3A,
    iors: &[RefractiveIndex],
) -> ScatteringSpectrum {
    let reflected_dir = reflect(ray.d.into(), n);
    let refracted = iors
        .iter()
        .map(|ior| {
            refract(ray.d.into(), n, 1.0, ior.eta).map(|d| Ray {
                o: p.into(),
                d: d.into(),
            })
        })
        .collect::<Vec<_>>();

    let reflectance = optics::reflectance_air_conductor_spectrum(ray.d.dot(n.into()).abs(), iors);

    ScatteringSpectrum {
        reflected: Ray {
            o: p.into(),
            d: reflected_dir.into(),
        },
        refracted,
        reflectance,
    }
}

// /// Compute refraction direction using Snell's law.
// ///
// /// # Arguments
// ///
// /// * `w_i` - direction of the ray (normalised).
// /// * `n` - normal of the surface (normalised).
// /// * `eta_t` - index of refraction of the incident medium.
// /// * `eta_t` - index of refraction of the transmitted medium.
// ///
// /// # Notes
// ///
// /// The direction of the incident ray `w_i` is determined from the ray's origin
// /// rather than the point of intersection.
// ///
// /// # Returns
// ///
// /// The refracted direction if the total internal reflection is not occurring,
// /// otherwise None.
// pub fn refract(w_i: Vec3A, n: Vec3A, eta_i: f32, eta_t: f32) -> Option<Vec3A> {
//     let dot = w_i.dot(n);
//
//     let (cos_i, eta_i, eta_t, n) = if dot < 0.0 {
//         // Ray is on the outside of the interface, `cos_i` needs to be positive.
//         (-dot, eta_i, eta_t, n)
//     } else {
//         // Ray is on the inside of the interface, normal and refractive indices need to
//         // be swapped.
//         (dot, eta_t, eta_i, -n)
//     };
//
//     let eta = eta_i / eta_t;
//     let sin_i_sqr = 1.0 - cos_i * cos_i;
//     let t0 = 1.0 - eta * eta * sin_i_sqr;
//
//     if t0 < 0.0 {
//         // Total internal reflection.
//         None
//     } else {
//         Some(w_i * eta + (eta * cos_i - t0.sqrt()) * n)
//     }
// }

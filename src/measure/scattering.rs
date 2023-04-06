use crate::{
    measure::rtc::Ray,
    optics,
    optics::{
        fresnel::{reflect, refract2, RefractionResult},
        ior::RefractiveIndex,
    },
};
use glam::Vec3A;

// TODO: to be removed

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

// TODO: to be removed
/// Scattering happens between the air and the conductor.
pub fn scatter_air_conductor(ray: Ray, p: Vec3A, n: Vec3A, eta_t: f32, k_t: f32) -> Scattering {
    let reflected_dir = reflect(ray.dir.into(), n);
    let refracted = refract2(ray.dir.into(), n, 1.0, eta_t);
    let reflectance = optics::fresnel::reflectance_insulator_conductor(
        ray.dir.dot(n.into()).abs(),
        1.0,
        eta_t,
        k_t,
    );

    Scattering {
        reflected: Ray::new(p.into(), reflected_dir.into()),
        refracted: match refracted {
            RefractionResult::TotalInternalReflection => None,
            RefractionResult::Refraction { dir_t, .. } => Some(Ray::new(p.into(), dir_t.into())),
        },
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
    let reflected_dir = reflect(ray.dir.into(), n);
    let refracted = iors
        .iter()
        .map(|ior| match refract2(ray.dir.into(), n, 1.0, ior.eta) {
            RefractionResult::TotalInternalReflection => None,
            RefractionResult::Refraction { dir_t, .. } => Some(Ray::new(p.into(), dir_t.into())),
        })
        .collect::<Vec<_>>();

    let reflectance = optics::fresnel::reflectance_insulator_conductor_spectrum(
        ray.dir.dot(n.into()).abs(),
        1.0,
        iors,
    );

    ScatteringSpectrum {
        reflected: Ray::new(p.into(), reflected_dir.into()),
        refracted,
        reflectance,
    }
}

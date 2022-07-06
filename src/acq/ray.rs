use std::fmt::Debug;
use crate::acq::fresnel;
use crate::acq::ior::RefractiveIndex;
use glam::{UVec3, Vec3};
use crate::isect::Axis;

/// Representation of a ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// The origin of the ray.
    pub o: Vec3,

    /// The direction of the ray.
    pub d: Vec3,

    /// Energy of the ray.
    pub e: f32,
}

impl Ray {
    /// Create a new ray (direction will be normalised).
    pub fn new(o: Vec3, d: Vec3) -> Self {
        let d = d.normalize();
        // let inv_dir_z = 1.0 / d.z;
        // let kz = Axis::max_axis(d.abs());
        // let kx = kz.next_axis();
        // let ky = kz.next_axis();
        Self {
            o,
            d,
            e: 1.0,
        }
    }

    pub fn from_embree_ray(ray: &embree::Ray, energy: f32) -> Self {
        let o = Vec3::new(ray.org_x, ray.org_y, ray.org_z);
        let d = Vec3::new(ray.dir_x, ray.dir_y, ray.dir_z);
        Self {
            o,
            d,
            e: energy,
        }
    }

    pub fn into_embree_ray(self) -> embree::Ray {
        embree::Ray::new(self.o.into(), self.d.into())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Scattering {
    pub reflected: Ray,
    pub refracted: Ray,
}

pub fn scattering_air_conductor(ray: Ray, p: Vec3, n: Vec3, eta_t: f32, k_t: f32) -> Option<Scattering> {
    if ray.e < 0.0 {
        None
    } else {
        let reflected_dir = reflect(ray.d, n);
        let refracted_dir = refract();
        let reflectance = fresnel::reflectance_dielectric_conductor(ray.d.dot(n).abs(), eta_t, k_t);

        Some(Scattering {
            reflected: Ray {
                o: p,
                d: reflected_dir,
                e: ray.e * reflectance,
            },
            refracted: Ray {
                o: p,
                d: refracted_dir,
                e: ray.e * (1.0 - reflectance),
            },
        })
    }
}

pub fn scattering_air_conductor_spectrum(
    ray: Ray,
    point: Vec3,
    normal: Vec3,
    iors: &[RefractiveIndex],
) -> Vec<Option<Scattering>> {
    if ray.e < 0.0 {
        vec![None; iors.len()]
    } else {
        let reflected_dir = reflect(ray.d, normal);
        let refracted_dir = refract();
        let reflectance = fresnel::reflectance_dielectric_conductor_spectrum(ray.d.dot(normal).abs(), iors);

        reflectance
            .iter()
            .map(|r| {
                Some(Scattering {
                    reflected: Ray {
                        o: point,
                        d: reflected_dir,
                        e: ray.e * r,
                    },
                    refracted: Ray {
                        o: point,
                        d: refracted_dir,
                        e: ray.e * (1.0 - r),
                    },
                })
            })
            .collect()
    }
}

/// Compute reflection direction.
pub fn reflect(w_i: Vec3, n: Vec3) -> Vec3 {
    // if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
    //     #[cfg(target_arch = "x86_64")]
    //     use core::arch::x86_64::*;
    //
    //     #[cfg(target_arch = "x86")]
    //     use core::arch::x86::*;
    //
    //     let mut out = [0.0f32; 4];
    //     unsafe {
    //         let w_i = _mm_setr_ps(w_i.x, w_i.y, w_i.z, 0.0);
    //         let n = _mm_setr_ps(n.x, n.y, n.z, 0.0);
    //         let w_i_dot_n: f32 = {
    //             let mut prod = [0.0f32; 4];
    //             _mm_storeu_ps(prod.as_mut_ptr(), _mm_mul_ps(w_i, n));
    //             prod.iter().sum()
    //         };
    //         _mm_storeu_ps(
    //             out.as_mut_ptr(),
    //             _mm_sub_ps(
    //                 w_i,
    //                 _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2.0),
    // _mm_set1_ps(w_i_dot_n)), n),             ),
    //         )
    //     }
    //     Vec3::from_slice(&out[..3]).normalize()
    // } else {
    (w_i - 2.0 * w_i.dot(n) * n).normalize()
    // }
}

pub fn refract() -> Vec3 {
    // TODO: implement
    Vec3::ZERO
}

#[derive(Debug)]
pub struct RayTraceRecord {
    /// The initial ray being traced.
    pub initial: Ray,

    /// The current ray being traced.
    pub current: Ray,

    /// The number of bounces.
    pub bounces: u32,
}

pub struct RayTraceRecordDbg {
    pub rays: Vec<Ray>,
    pub bounces: u32,
}

use crate::acq::fresnel;
use glam::Vec3;

pub struct Ray {
    /// The origin of the ray.
    pub o: Vec3,

    /// The direction of the ray.
    pub d: Vec3,

    /// Energy of the ray.
    pub e: f32,
}

impl Ray {
    /// Create a new ray.
    pub fn new(o: Vec3, d: Vec3) -> Self {
        Self { o, d, e: 1.0 }
    }
}

pub struct Scattering {
    pub reflected: Ray,
    pub refracted: Ray,
}

pub fn fresnel_scattering_air_conductor(
    ray: &Ray,
    p: Vec3,
    n: Vec3,
    eta_t: f32,
    k_t: f32,
) -> Option<Scattering> {
    if ray.e < 0.0 {
        None
    } else {
        let reflected_dir = reflect(ray.d, n);
        let refracted_dir = refract(ray.d, n, 1.0, eta_t);
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

/// Compute reflection direction.
pub fn reflect(w_i: Vec3, n: Vec3) -> Vec3 {
    if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;

        let mut out = [0.0f32; 4];
        unsafe {
            let w_i = _mm_setr_ps(w_i.x, w_i.y, w_i.z, 0.0);
            let n = _mm_setr_ps(n.x, n.y, n.z, 0.0);
            let w_i_dot_n: f32 = {
                let mut prod = [0.0f32; 4];
                _mm_storeu_ps(prod.as_mut_ptr(), _mm_mul_ps(w_i, n));
                prod.iter().sum()
            };
            _mm_storeu_ps(
                out.as_mut_ptr(),
                _mm_sub_ps(
                    w_i,
                    _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2.0), _mm_set1_ps(w_i_dot_n)), n),
                ),
            )
        }
        Vec3::from_slice(&out[..3]).normalize()
    } else {
        (w_i - 2.0 * w_i.dot(n) * n).normalize()
    }
}

pub fn refract(w_i: Vec3, n: Vec3, eta_i: f32, eta_t: f32) -> Vec3 {
    // TODO: implement
    Vec3::ZERO
}

use crate::{hit::Hit, ray::Ray};
use jabr::Clr3;

pub trait Material: Send + Sync {
    /// Produces a scattered ray and attenuation color.
    fn scatter(&self, wi: &Ray, hit: &Hit) -> Option<(Ray, Clr3)>;
}

#[derive(Clone, Copy, Debug)]
pub struct Lambertian {
    pub albedo: Clr3,
}

impl Material for Lambertian {
    fn scatter(&self, _wi: &Ray, hit: &Hit) -> Option<(Ray, Clr3)> {
        let n = if hit.front_face { hit.n } else { -hit.n };
        let mut scatter_dir = n + crate::random::random_vec3_on_unit_sphere();
        if scatter_dir.near_zero() {
            scatter_dir = n;
        }
        let scattered = Ray::new(hit.p, scatter_dir);
        // No attenuation
        Some((scattered, self.albedo))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Metal {
    pub albedo: Clr3,
    /// Must be in the range [0, 1]. 0 is a perfect mirror, 1 is a glossy
    /// surface.
    pub gloss: f64,
}

impl Metal {
    pub fn new(albedo: Clr3, gloss: f64) -> Self {
        Metal {
            albedo,
            gloss: gloss.clamp(0.0, 1.0),
        }
    }
}

impl Material for Metal {
    fn scatter(&self, wi: &Ray, hit: &Hit) -> Option<(Ray, Clr3)> {
        let n = if hit.front_face { hit.n } else { -hit.n };
        let reflected = wi.dir.reflect(&n);
        let scattered = Ray::new(
            hit.p,
            // Randomize the reflected ray direction by randomly sampling a
            // point on a unit sphere centered at the end of the reflected ray.
            reflected + self.gloss * crate::random::random_vec3_on_unit_sphere(),
        );
        // No attenuation
        Some((scattered, self.albedo))
    }
}

pub struct Dielectric {
    /// The index of refraction of the material.
    pub ior: f64,
}

impl Dielectric {
    pub fn new(ior: f64) -> Self { Dielectric { ior } }
}

impl Material for Dielectric {
    fn scatter(&self, wi: &Ray, hit: &Hit) -> Option<(Ray, Clr3)> { todo!() }
}

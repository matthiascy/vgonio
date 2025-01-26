//! Lambertian BRDF model.
use crate::math::Vec3;

/// Lambertian BRDF model.
pub struct LambertianBrdf {
    /// Reflectance of the surface, i.e. the fraction of light that is
    /// reflected.
    pub reflectance: Vec3,
}

// impl Brdf for LambertianBrdf {}

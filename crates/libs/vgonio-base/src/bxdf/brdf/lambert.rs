use crate::math::Vec3;

pub struct LambertianBrdf {
    // Replace it with spectrum
    /// Reflectance of the surface, i.e. the fraction of light that is
    /// reflected.
    pub reflectance: Vec3,
}

// impl Brdf for LambertianBrdf {}

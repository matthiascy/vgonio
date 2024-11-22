pub mod camera;
pub mod display;
pub mod hit;
pub mod image;
pub mod material;
pub mod random;
pub mod ray;
pub mod sphere;

#[cfg(feature = "f64")]
#[allow(non_camel_case_types)]
pub type float = f64;

#[cfg(not(feature = "f64"))]
#[allow(non_camel_case_types)]
pub type float = f32;

pub struct RenderParams<'a> {
    pub camera: &'a camera::Camera,
    pub world: &'a hit::HittableList,
    pub spp: u32,
    pub max_bounces: u32,
}

trait Float {
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}

#![feature(const_fn_floating_point_arithmetic)]

pub mod camera;
pub mod display;
pub mod hit;
pub mod image;
pub mod material;
pub mod random;
pub mod ray;
pub mod sphere;

pub struct RenderParams<'a> {
    pub camera: &'a camera::Camera,
    pub world: &'a hit::HittableList,
    pub spp: u32,
    pub max_bounces: u32,
}

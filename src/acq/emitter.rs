use crate::acq::collector::Patch;
use crate::acq::desc::EmitterDesc;
use crate::acq::ray::Ray;
use crate::acq::util::SphericalShape;
use glam::Vec3;

pub struct Emitter {
    pub num_rays: u32,
    pub radius: f32,
    pub patches: Vec<Patch>,
}

impl Emitter {
    /// Emit `num_rays` rays from the patch of the emitter.
    /// TODO: implement sampling
    /// TODO: improve signature -> should the patch be the one from the emitter?
    pub fn emit_from_patch(&self, patch: &Patch) -> Vec<Ray> {
        let mut rays = Vec::with_capacity(self.num_rays as usize);
        rays.resize(self.num_rays as usize, Ray::new(Vec3::ZERO, Vec3::ZERO));
        rays
    }
}

impl From<EmitterDesc> for Emitter {
    fn from(desc: EmitterDesc) -> Self {
        Self {
            num_rays: desc.num_rays,
            radius: desc.radius,
            patches: desc.partition.generate_patches(SphericalShape::WholeSphere),
        }
    }
}

use crate::acq::collector::Patch;
use crate::acq::desc::{EmitterDesc, RadiusDesc};
use crate::acq::util::SphericalShape;

pub struct Emitter {
    pub num_rays: u32,
    pub radius: RadiusDesc,
    pub patches: Vec<Patch>,
}

impl From<EmitterDesc> for Emitter {
    fn from(desc: EmitterDesc) -> Self {
        Self {
            num_rays: desc.num_rays,
            radius: desc.radius,
            patches: desc
                .partition
                .generate_patches(SphericalShape::UpperHemisphere),
        }
    }
}

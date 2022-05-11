use crate::acq::collector::{Patch, spherical_partition};
use crate::acq::desc::EmitterDesc;

pub struct Emitter {
    pub radius: f32,
    pub patches: Vec<Patch>,
}

impl From<EmitterDesc> for Emitter {
    fn from(desc: EmitterDesc) -> Self {
        Self {
            radius: desc.radius,
            patches: spherical_partition(Shape),
        }
    }
}

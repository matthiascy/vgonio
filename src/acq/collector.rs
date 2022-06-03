use crate::acq::desc::{CollectorDesc, RadiusDesc};
use crate::acq::ray::Ray;
use crate::acq::util::{SphericalPartition, SphericalShape};
use glam::Vec3;

/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
#[derive(Clone, Debug)]
pub struct Collector {
    pub radius: RadiusDesc,
    pub shape: SphericalShape,
    pub partition: SphericalPartition,
    pub patches: Vec<Patch>,
}

/// Represents a patch on the spherical [`Collector`].
#[derive(Copy, Clone, Debug)]
pub struct Patch {
    /// Polar angle of the center of the patch in radians.
    pub zenith: f32,

    /// Azimuthal angle of the center of the patch in radians.
    pub azimuth: f32,
}

impl Patch {
    pub fn emit_rays(&self, n: u32, radius: f32) -> Vec<Ray> {
        let origin = Vec3::new(
            self.zenith.sin() * radius * self.azimuth.cos(),
            self.zenith.sin() * radius * self.azimuth.sin(),
            self.zenith.cos() * radius,
        );
        let dir = (-origin).normalize();
        let mut rays = Vec::with_capacity(n as usize);
        rays.resize(n as usize, Ray::new(origin, dir));
        rays
    }
}

impl From<CollectorDesc> for Collector {
    fn from(desc: CollectorDesc) -> Self {
        Self {
            radius: desc.radius,
            shape: desc.shape,
            partition: desc.partition,
            patches: desc.partition.generate_patches(desc.shape),
        }
    }
}

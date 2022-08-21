use crate::acq::{
    desc::{CollectorDesc, RadiusDesc},
    util::{SphericalPartition, SphericalShape},
    RtcRecord,
};
use std::fmt::Debug;
use crate::acq::bsdf::BsdfKind;

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

impl Collector {
    pub fn collect<R>(&mut self, records: R, kind: BsdfKind)
    where
        R: IntoIterator<Item = RtcRecord>,
        <<R as IntoIterator>::IntoIter as Iterator>::Item: Debug,
    {
        // for record in records.into_iter() {
        //     // print!("{} ", record.bounces());
        // }
        // println!();
    }

    pub fn save_stats(&self, path: &str) { unimplemented!() }
}

/// Represents a patch on the spherical [`Collector`].
#[derive(Copy, Clone, Debug)]
pub struct Patch {
    /// Polar angle range of the patch (in radians).
    pub zenith: (f32, f32),

    /// Azimuthal angle range of the patch (in radians).
    pub azimuth: (f32, f32),

    /// Solid angle of the patch (in radians).
    pub solid_angle: f32,
}

impl Patch {
    /// Creates a new patch.
    ///
    /// # Arguments
    /// * `zenith` - Polar angle range (start, stop) of the patch (in radians).
    /// * `azimuth` - Azimuthal angle range (start, stop) of the patch (in
    ///   radians).
    pub fn new(zenith: (f32, f32), azimuth: (f32, f32)) -> Self {
        let solid_angle = (zenith.0.cos() - zenith.1.cos()) * (azimuth.1 - azimuth.0);
        Self {
            zenith,
            azimuth,
            solid_angle,
        }
    }
}

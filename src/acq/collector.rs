use crate::acq::{bsdf::BsdfKind, desc::{CollectorDesc, RadiusDesc}, util::{SphericalPartition, SphericalDomain}, RtcRecord, Radians, steradians, SolidAngle};
use std::fmt::Debug;
use crate::acq::desc::{Radius, Range};
use crate::acq::emitter::RegionShape;

/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
#[derive(Clone, Debug)]
pub struct Collector {
    pub radius: Radius,
    pub scheme: CollectorScheme,
}

#[derive(Debug)]
pub enum CollectorScheme {
    /// The patches are subdivided using a spherical partition.
    Partitioned {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Spherical partition of the collector.
        partition: SphericalPartition,
        /// Partitioned patches.
        patches: Vec<Patch>,
    },
    /// The collector is represented by a single shape on the surface of the sphere.
    Individual {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Shape of the collector.
        shape: RegionShape,
        /// Collector's possible positions in spherical coordinates (inclination angle range).
        zenith: Range<Radians>,
        /// Collector's possible positions in spherical coordinates (azimuthal angle range).
        azimuth: Range<Radians>,
    },
}

impl From<CollectorDesc> for Collector {
    fn from(desc: CollectorDesc) -> Self {
        Self {
            radius: desc.radius.into(),
            domain: desc.shape,
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
    pub zenith: (Radians, Radians),

    /// Azimuthal angle range of the patch (in radians).
    pub azimuth: (Radians, Radians),

    /// Solid angle of the patch (in radians).
    pub solid_angle: SolidAngle,
}

impl Patch {
    /// Creates a new patch.
    ///
    /// # Arguments
    /// * `zenith` - Polar angle range (start, stop) of the patch (in radians).
    /// * `azimuth` - Azimuthal angle range (start, stop) of the patch (in
    ///   radians).
    pub fn new(zenith: (Radians, Radians), azimuth: (Radians, Radians)) -> Self {
        Self {
            zenith,
            azimuth,
            solid_angle: SolidAngle::from_angle_ranges(zenith, azimuth),
        }
    }

    /// Updates area of the patch.
    pub fn update(&mut self, zenith: (Radians, Radians), azimuth: (Radians, Radians)) {
        self.zenith = zenith;
        self.azimuth = azimuth;
        self.solid_angle = SolidAngle::from_angle_ranges(zenith, azimuth);
    }
}

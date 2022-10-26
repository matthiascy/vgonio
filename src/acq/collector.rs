use crate::acq::{
    bsdf::BsdfKind,
    emitter::RegionShape,
    measurement::Radius,
    util::{RangeByStepSize, SphericalDomain, SphericalPartition},
    Radians, RtcRecord, SolidAngle,
};

/// Description of a collector.
///
/// A collector could be either a single shape or a set of patches.
///
/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Collector {
    /// Distance from the collector's center to the specimen's center.
    pub radius: Radius,

    /// Strategy for data collection.
    pub scheme: CollectorScheme,

    /// Partitioned patches of the collector.
    #[serde(skip)]
    pub patches: Option<Vec<Patch>>,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CollectorScheme {
    /// The patches are subdivided using a spherical partition.
    Partitioned {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Spherical partition of the collector.
        partition: SphericalPartition,
    },
    /// The collector is represented by a single shape on the surface of the sphere.
    Individual {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Shape of the collector.
        shape: RegionShape,
        /// Collector's possible positions in spherical coordinates (inclination angle range).
        zenith: RangeByStepSize<Radians>,
        /// Collector's possible positions in spherical coordinates (azimuthal angle range).
        azimuth: RangeByStepSize<Radians>,
    },
}

impl CollectorScheme {
    /// Generate the patches of the collector.
    pub fn generate_patches(&self) -> Option<Vec<Patch>> {
        match self {
            Self::Partitioned { domain, partition } => Some(partition.generate_patches(domain)),
            // TODO: implement
            Self::Individual { .. } => None,
        }
    }
}

impl Collector {
    pub fn collect<R>(&mut self, records: R, kind: BsdfKind)
    where
        R: IntoIterator<Item = RtcRecord>,
        <<R as IntoIterator>::IntoIter as Iterator>::Item: core::fmt::Debug,
    {
        todo!("Collector::collect")
    }

    pub fn save_stats(&self, path: &str) { todo!("Collector::save_stats") }

    pub fn generate_patches(&mut self) {
        self.patches = self.scheme.generate_patches();
    }
}

/// Represents a patch on the spherical [`Collector`].
#[derive(Debug, Copy, Clone, PartialEq)]
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

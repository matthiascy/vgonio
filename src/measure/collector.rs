use crate::{
    common::{RangeByStepSize, SphericalDomain, SphericalPartition},
    measure::{bsdf::BsdfKind, emitter::RegionShape, measurement::Radius},
    units::{Radians, SolidAngle},
};

use crate::{
    measure::{bsdf::BsdfStats, rtc::embr::RayStreamStats},
};
use serde::{Deserialize, Serialize};

/// Description of a collector.
///
/// A collector could be either a single shape or a set of patches.
///
/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Collector {
    /// Distance from the collector's center to the specimen's center.
    pub radius: Radius,

    /// Strategy for data collection.
    pub scheme: CollectorScheme,

    /// Partitioned patches of the collector, generated only if the
    /// `scheme` is [`CollectorScheme::Partitioned`].
    #[serde(skip)]
    pub patches: Option<Vec<Patch>>,

    #[serde(skip)]
    /// Initialised flag.
    pub(crate) init: bool,
}

/// Strategy for data collection.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CollectorScheme {
    /// The patches are subdivided using a spherical partition.
    Partitioned {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Spherical partition of the collector.
        partition: SphericalPartition,
    },
    /// The collector is represented by a single shape on the surface of the
    /// sphere.
    SingleRegion {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Shape of the collector.
        shape: RegionShape,
        /// Collector's possible positions in spherical coordinates (inclination
        /// angle range).
        zenith: RangeByStepSize<Radians>,
        /// Collector's possible positions in spherical coordinates (azimuthal
        /// angle range).
        azimuth: RangeByStepSize<Radians>,
    },
}

impl Collector {
    /// Initialise the collector.
    pub fn init(&mut self) {
        log::info!("Initialising collector...");
        // Generate patches of the collector if the scheme is partitioned.
        self.patches = match self.scheme {
            CollectorScheme::Partitioned { domain, partition } => {
                Some(partition.generate_patches_over_domain(&domain))
            }
            CollectorScheme::SingleRegion { .. } => None,
        };
        self.init = true;
    }

    /// Collects the ray-tracing data.
    #[cfg(feature = "embree")]
    pub fn collect_embree_rt(
        &self,
        kind: BsdfKind,
        stats: &[RayStreamStats],
    ) -> BsdfStats<BounceEnergyPerPatch> {
        debug_assert!(self.init, "Collector not initialised");
        // TODO: try to parallelise
        for stats_per_stream in stats {
            log::debug!("Stats per stream: {:?}", stats_per_stream);
            for status in stats_per_stream {
                log::debug!("Status: {:?}", status);
            }
        }
        todo!("Collector::collect")
    }

    // todo: pub fn collect_grid_rt

    pub fn save_stats(&self, _path: &str) {
        debug_assert!(self.init, "Collector not initialised");
        todo!("Collector::save_stats") }
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

/// Bounce and energy of a patch.
pub struct BounceEnergyPerPatch {
    pub bounce: Vec<u32>,
    pub energy: Vec<f32>,
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

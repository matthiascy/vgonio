use std::os::unix::process::parent_id;
use crate::{
    common::{RangeByStepSize, SphericalDomain, SphericalPartition},
    measure::{bsdf::BsdfKind, emitter::RegionShape, measurement::Radius},
    units::{Radians, SolidAngle},
};

use crate::{
    measure::{bsdf::BsdfStats, rtc::embr::RayStreamStats},
};
use serde::{Deserialize, Serialize};
use crate::common::SphericalCoord;
use crate::measure::rtc::embr::{TracingStatus};

/// Description of a collector.
///
/// A collector could be either a single shape or a set of patches.
///
/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Collector {
    /// Distance from the collector's center to the specimen's center.
    pub radius: Radius,

    /// Strategy for data collection.
    pub scheme: CollectorScheme,
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

impl CollectorScheme {
    /// Returns the spherical domain of the collector.
    pub fn domain(&self) -> SphericalDomain {
        match self {
            Self::Partitioned { domain, .. } => *domain,
            Self::SingleRegion { domain, .. } => *domain,
        }
    }
}

impl Collector {
    /// Generates the patches of the collector.
    pub fn generate_patches(&self) -> CollectorPatches {
        match self.scheme {
            CollectorScheme::Partitioned { domain, partition } => {
                CollectorPatches::Partitioned(partition.generate_patches_over_domain(&domain))
            }
            CollectorScheme::SingleRegion { zenith, azimuth, .. } => {
                let n_zenith =
                    ((zenith.stop - zenith.start) / zenith.step_size).ceil() as usize;
                let n_azimuth =
                    ((azimuth.stop - azimuth.start) / azimuth.step_size).ceil() as usize;

                let patches = (0..n_zenith)
                    .flat_map(|i_theta| {
                        (0..n_azimuth).map(move |i_phi| SphericalCoord {
                            zenith: zenith.start + i_theta as f32 * zenith.step_size,
                            azimuth: azimuth.start + i_phi as f32 * azimuth.step_size,
                        })
                    })
                    .collect();
                CollectorPatches::SingleRegion(patches)
            }
        }
    }

    /// Collects the ray-tracing data.
    #[cfg(feature = "embree")]
    pub fn collect_embree_rt(
        &self,
        kind: BsdfKind,
        spectrum_len: usize,
        stats: &[RayStreamStats],
        patches: &CollectorPatches,
    ) -> BsdfStats<BounceEnergyPerPatch> {
        let mut num_missed = vec![0; spectrum_len];
        let mut num_reflected = vec![0; spectrum_len];
        let mut num_absorbed = vec![0; spectrum_len];
        // TODO: try to parallelise
        for stats_per_stream in stats {
            log::debug!("Stats per stream: {:?}", stats_per_stream);
            for ray_status in stats_per_stream {
                for (i, status) in ray_status.tracing_status.iter().enumerate() {
                    match status {
                        TracingStatus::Missed => num_missed[i] += 1,
                        TracingStatus::Reflected(last_ray) => {
                            num_reflected[i] += 1;
                        },
                        TracingStatus::Absorbed(last_ray) => {
                            num_absorbed[i] += 1;
                        },
                    }
                }
                log::debug!("Status: {:?}", ray_status);
            }
        }
        match kind {
            BsdfKind::Brdf => {
                match self.scheme {
                    CollectorScheme::Partitioned {..} => {
                        // collecto into patches

                    }
                    CollectorScheme::SingleRegion { .. } => {
                        todo!("Collector::collect_embree_rt(CollectorScheme::SingleRegion, BsdfKind::Brdf)")
                    }
                }
            }
            BsdfKind::Btdf => {
                todo!("Collector::collect_embree_rt(BsdfKind::Btdf)")
            }
            BsdfKind::Bssdf => {
                todo!("Collector::collect_embree_rt(BsdfKind::Bssrdf)")
            }
            BsdfKind::Bssrdf => {
                todo!("Collector::collect_embree_rt(BsdfKind::Bssrdf)")
            }
            BsdfKind::Bsstdf => {
                todo!("Collector::collect_embree_rt(BsdfKind::Bsstdf)")
            }
        }
    }

    // todo: pub fn collect_grid_rt

    pub fn save_stats(&self, _path: &str) {
        todo!("Collector::save_stats")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectedData {
    /// Partitioned patches of the collector, generated only if the
    /// `scheme` is [`CollectorScheme::Partitioned`].
    #[serde(skip)]
    pub patches: Option<Vec<Patch>>,

    #[serde(skip)]
    /// Initialised flag.
    pub(crate) init: bool,
}

/// Represents patches on the surface of the spherical [`Collector`].
///
/// The domain of the whole collector is defined by the [`Collector`].
pub enum CollectorPatches {
    SingleRegion(Vec<SphericalCoord>),
    Partitioned(Vec<Patch>),
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

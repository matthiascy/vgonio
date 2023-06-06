use crate::{
    math,
    measure::{emitter::RegionShape, measurement::Radius},
    units::{Radians, SolidAngle},
    RangeByStepSizeInclusive, SphericalDomain, SphericalPartition,
};
use glam::{Vec3, Vec3A};
use std::ops::{Deref, DerefMut};

use crate::{
    app::cache::Cache,
    math::{solve_quadratic, sqr, QuadraticSolution},
    measure::{
        bsdf::{BsdfMeasurementDataPoint, BsdfMeasurementStatsPoint, PerWavelength},
        measurement::BsdfMeasurementParams,
        rtc::Trajectory,
    },
    msurf::MicroSurfaceMesh,
    optics::fresnel,
    Handedness, SphericalCoord,
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
#[repr(u8)]
pub enum CollectorScheme {
    /// The patches are subdivided using a spherical partition.
    Partitioned {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Spherical partition of the collector.
        partition: SphericalPartition,
    } = 0x00,
    /// The collector is represented by a single shape on the surface of the
    /// sphere.
    SingleRegion {
        /// Spherical domain of the collector.
        domain: SphericalDomain,
        /// Shape of the collector.
        shape: RegionShape,
        /// Collector's possible positions in spherical coordinates (inclination
        /// angle range).
        zenith: RangeByStepSizeInclusive<Radians>,
        /// Collector's possible positions in spherical coordinates (azimuthal
        /// angle range).
        azimuth: RangeByStepSizeInclusive<Radians>,
    } = 0x01,
}

impl CollectorScheme {
    /// Returns the spherical domain of the collector.
    pub fn domain(&self) -> SphericalDomain {
        match self {
            Self::Partitioned { domain, .. } => *domain,
            Self::SingleRegion { domain, .. } => *domain,
        }
    }

    pub(crate) fn domain_mut(&mut self) -> &mut SphericalDomain {
        match self {
            Self::Partitioned { domain, .. } => domain,
            Self::SingleRegion { domain, .. } => domain,
        }
    }

    /// Returns the shape of the collector only if it is a single region.
    pub fn shape(&self) -> Option<RegionShape> {
        match self {
            Self::Partitioned { .. } => None,
            Self::SingleRegion { shape, .. } => Some(*shape),
        }
    }

    pub(crate) fn shape_mut(&mut self) -> Option<&mut RegionShape> {
        match self {
            Self::Partitioned { .. } => None,
            Self::SingleRegion { shape, .. } => Some(shape),
        }
    }

    pub(crate) fn zenith_mut(&mut self) -> Option<&mut RangeByStepSizeInclusive<Radians>> {
        match self {
            Self::Partitioned { .. } => None,
            Self::SingleRegion { zenith, .. } => Some(zenith),
        }
    }

    pub(crate) fn azimuth_mut(&mut self) -> Option<&mut RangeByStepSizeInclusive<Radians>> {
        match self {
            Self::Partitioned { .. } => None,
            Self::SingleRegion { azimuth, .. } => Some(azimuth),
        }
    }

    pub(crate) fn partition_mut(&mut self) -> Option<&mut SphericalPartition> {
        match self {
            Self::Partitioned { partition, .. } => Some(partition),
            Self::SingleRegion { .. } => None,
        }
    }

    /// Returns true if the collector is partitioned.
    pub fn is_partitioned(&self) -> bool {
        match self {
            Self::Partitioned { .. } => true,
            Self::SingleRegion { .. } => false,
        }
    }

    /// Returns true if the collector is a single region.
    pub fn is_single_region(&self) -> bool {
        match self {
            Self::Partitioned { .. } => false,
            Self::SingleRegion { .. } => true,
        }
    }

    /// Returns the number of samples collected by the collector.
    pub fn total_sample_count(&self) -> usize {
        match self {
            CollectorScheme::Partitioned { partition, .. } => match partition {
                SphericalPartition::EqualAngle { zenith, azimuth } => {
                    zenith.step_count_wrapped() * azimuth.step_count_wrapped()
                }
                SphericalPartition::EqualArea { zenith, azimuth } => {
                    zenith.step_count * azimuth.step_count_wrapped()
                }
                SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                    zenith.step_count * azimuth.step_count_wrapped()
                }
            },
            CollectorScheme::SingleRegion {
                zenith, azimuth, ..
            } => zenith.step_count_wrapped() * azimuth.step_count_wrapped(),
        }
    }

    /// Returns the default value of [`CollectorScheme::Partitioned`].
    pub fn default_partition() -> Self {
        Self::Partitioned {
            domain: SphericalDomain::default(),
            partition: SphericalPartition::default(),
        }
    }

    /// Returns the default value of [`CollectorScheme::SingleRegion`].
    pub fn default_single_region() -> Self {
        Self::SingleRegion {
            domain: SphericalDomain::default(),
            shape: RegionShape::default_spherical_cap(),
            zenith: RangeByStepSizeInclusive::zero_to_half_pi(Radians::from_degrees(5.0)),
            azimuth: RangeByStepSizeInclusive::zero_to_tau(Radians::from_degrees(15.0)),
        }
    }
}

/// Energy after a ray is reflected by the micro-surface.
///
/// Used during the data collection process.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
enum Energy {
    /// The ray of certain wavelength is absorbed by the micro-surface.
    Absorbed,
    /// The ray of a specific wavelength is reflected by the micro-surface.
    Reflected(f32),
}

impl Energy {
    /// Returns the energy of the ray.
    fn energy(&self) -> f32 {
        match self {
            Self::Absorbed => 0.0,
            Self::Reflected(energy) => *energy,
        }
    }
}

impl Collector {
    /// Generates the patches of the collector.
    pub fn generate_patches(&self) -> CollectorPatches {
        log::debug!("Generating patches of the collector.");
        match self.scheme {
            CollectorScheme::Partitioned { domain, partition } => {
                log::trace!("Generating patches of the partitioned collector.");
                CollectorPatches(partition.generate_patches_over_domain(&domain))
            }
            CollectorScheme::SingleRegion {
                zenith, azimuth, ..
            } => {
                // TODO: verification
                log::trace!("Generating patches of the single region collector.");
                let n_zenith = (zenith.span() / zenith.step_size).ceil() as usize;
                let n_azimuth = (azimuth.span() / azimuth.step_size).ceil() as usize;

                let patches = (0..n_zenith)
                    .flat_map(|i_theta| {
                        (0..n_azimuth).map(move |i_phi| {
                            let spherical = SphericalCoord {
                                radius: 1.0,
                                zenith: zenith.start + i_theta as f32 * zenith.step_size,
                                azimuth: azimuth.start + i_phi as f32 * azimuth.step_size,
                            };
                            let cartesian = spherical.to_cartesian(Handedness::RightHandedYUp);
                            let shape = self.scheme.shape().unwrap();
                            let solid_angle = shape.solid_angle();
                            Patch::SingleRegion(PatchSingleRegion {
                                zenith: spherical.zenith,
                                azimuth: spherical.azimuth,
                                shape,
                                unit_vector: cartesian.into(),
                                solid_angle,
                            })
                        })
                    })
                    .collect();
                CollectorPatches(patches)
            }
        }
    }

    /// Collects the ray-tracing data.
    ///
    /// Returns the collected data and the statistics of the BSDF.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the BSDF measurement.
    /// * `mesh` - The micro-surface mesh.
    /// * `position` - The measurement position.
    /// * `trajectories` - The trajectories of the rays.
    /// * `patches` - The patches of the collector.
    /// * `cache` - The cache where all the data are stored.
    pub fn collect(
        &self,
        params: &BsdfMeasurementParams,
        mesh: &MicroSurfaceMesh,
        position: SphericalCoord,
        trajectories: &[Trajectory],
        patches: &CollectorPatches,
        cache: &Cache,
    ) -> BsdfMeasurementDataPoint<BounceAndEnergy> {
        // TODO: use generic type for the data point
        debug_assert!(
            patches.matches_scheme(&self.scheme),
            "Collector patches do not match the collector scheme"
        );

        log::debug!(
            "collecting data for BSDF measurement at position {}",
            position
        );

        let spectrum = params.emitter.spectrum.values().collect::<Vec<_>>();
        let n_wavelengths = spectrum.len();
        log::debug!("spectrum samples: {:?}", spectrum);

        // Get the refractive indices of the incident and transmitted media for each
        // wavelength.
        let iors_i = cache
            .iors
            .ior_of_spectrum(params.incident_medium, &spectrum)
            .expect("incident medium IOR not found");
        log::debug!("incident medium IORs: {:?}", iors_i);
        let iors_t = cache
            .iors
            .ior_of_spectrum(params.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");
        log::debug!("transmitted medium IORs: {:?}", iors_t);

        // Calculate the radius of the collector.
        let radius = self.radius.eval(mesh);
        log::trace!("collector radius: {}", radius);
        let domain = self.scheme.domain();
        let max_bounces = params.emitter.max_bounces as usize;
        let mut stats = BsdfMeasurementStatsPoint::new(n_wavelengths, max_bounces);

        #[derive(Debug, Copy, Clone)]
        struct OutgoingDir {
            idx: usize,
            dir: Vec3A,
            bounce: usize,
        }
        // Convert the last rays of the trajectories into a vector located
        // at the collector's center and pointing to the intersection point
        // of the last ray with the collector's surface. For later use classify
        // the rays according to the patch they intersect.
        //
        // Each element of the vector is a tuple containing the index of the
        // trajectory, the intersection point and the number of bounces.
        let outgoing_dirs = trajectories
            .iter()
            .enumerate()
            .filter_map(|(i, trajectory)| {
                match trajectory.last() {
                    None => None,
                    Some(last) => {
                        stats.n_received += 1;
                        // TODO(yang): take into account the handedness of the coordinate system
                        let is_in_domain = match domain {
                            SphericalDomain::Upper => last.dir.dot(Vec3A::Y) >= 0.0,
                            SphericalDomain::Lower => last.dir.dot(Vec3A::Y) <= 0.0,
                            SphericalDomain::Whole => true,
                        };
                        if !is_in_domain {
                            None
                        } else {
                            // 1. Calculate the intersection point of the last ray with
                            // the collector's surface. Ray-Sphere intersection.
                            // Collect's center is at (0, 0, 0).
                            let a = last.dir.dot(last.dir);
                            let b = 2.0 * last.dir.dot(last.org);
                            let c = last.org.dot(last.org) - sqr(radius);
                            let p = match solve_quadratic(a, b, c) {
                                QuadraticSolution::None | QuadraticSolution::One(_) => {
                                    unreachable!(
                                        "Ray starting inside the collector's surface, it should \
                                         have more than one intersection point."
                                    )
                                }
                                QuadraticSolution::Two(_, t) => last.org + last.dir * t,
                            };
                            // Returns the index of the ray, the unit vector pointing to the
                            // collector's surface, and the number of
                            // bounces.
                            Some(OutgoingDir {
                                idx: i,
                                dir: p.normalize(),
                                bounce: trajectory.len() - 1,
                            })
                        }
                    }
                }
            })
            .collect::<Vec<_>>();

        log::trace!("outgoing_dirs: {:?}", outgoing_dirs);

        // Calculate the energy of the rays (with wavelength) that intersected the
        // collector's surface.
        // Outer index: ray index, inner index: wavelength index.
        let ray_energy_per_wavelength = outgoing_dirs
            .iter()
            .map(|outgoing| {
                let trajectory = &trajectories[outgoing.idx];
                let mut energy = Vec::with_capacity(n_wavelengths);
                energy.resize(n_wavelengths, Energy::Reflected(1.0));
                for node in trajectories[outgoing.idx].iter().take(trajectory.len() - 1) {
                    for i in 0..spectrum.len() {
                        match energy[i] {
                            Energy::Absorbed => continue,
                            Energy::Reflected(ref mut e) => {
                                let cos = node.cos.unwrap_or(1.0);
                                *e *= fresnel::reflectance(cos, iors_i[i], iors_t[i]);
                                if *e <= 0.0 {
                                    energy[i] = Energy::Absorbed;
                                }
                            }
                        }
                    }
                }
                (outgoing.idx, energy)
            })
            .collect::<Vec<_>>();

        log::trace!("ray_energy_per_wavelength: {:?}", ray_energy_per_wavelength);

        // Calculate the energy of the rays (with wavelength) that intersected the
        // collector's surface.
        for (_, energies) in &ray_energy_per_wavelength {
            // per ray
            for (idx_wavelength, energy) in energies.iter().enumerate() {
                // per wavelength
                match energy {
                    Energy::Absorbed => stats.n_absorbed[idx_wavelength] += 1,
                    Energy::Reflected(_) => stats.n_reflected[idx_wavelength] += 1,
                }
            }
        }

        // For each patch, collect the rays that intersect it using the
        // outgoing_dirs vector.
        let outgoing_dirs_per_patch = patches
            .iter()
            .map(|patch| {
                // Retrieve the ray indices (of trajectories) that intersect the patch.
                outgoing_dirs
                    .iter()
                    .filter_map(|outgoing| {
                        log::trace!(
                            "    - outgoing.dir: {:?}, in spherical: {:?}",
                            outgoing.dir,
                            math::cartesian_to_spherical(
                                outgoing.dir.into(),
                                1.0,
                                Handedness::RightHandedYUp
                            )
                        );
                        log::trace!("    - patch: {:?}", patch);
                        log::trace!(
                            "    - patch.contains(outgoing.dir): {:?}",
                            patch.contains(outgoing.dir)
                        );
                        patch
                            .contains(outgoing.dir)
                            .then_some((outgoing.idx, outgoing.bounce))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        log::trace!("outgoing_dirs_per_patch: {:?}", outgoing_dirs_per_patch);

        let data = outgoing_dirs_per_patch
            .iter()
            .map(|dirs| {
                let mut data_per_patch = PerWavelength(vec![
                    BounceAndEnergy::empty(
                        params.emitter.max_bounces as usize
                    );
                    spectrum.len()
                ]);
                for (i, bounce) in dirs.iter() {
                    let ray_energy = &ray_energy_per_wavelength
                        .iter()
                        .find(|(j, energy)| *j == *i)
                        .unwrap()
                        .1;
                    for (lambda_idx, energy) in ray_energy.iter().enumerate() {
                        match energy {
                            Energy::Absorbed => continue,
                            Energy::Reflected(e) => {
                                stats.n_captured[lambda_idx] += 1;
                                data_per_patch[lambda_idx].total_energy += e;
                                data_per_patch[lambda_idx].total_rays += 1;
                                data_per_patch[lambda_idx].energy_per_bounce[*bounce - 1] += e;
                                data_per_patch[lambda_idx].num_rays_per_bounce[*bounce - 1] += 1;
                                stats.num_rays_per_bounce[lambda_idx][*bounce - 1] += 1;
                                stats.energy_per_bounce[lambda_idx][*bounce - 1] += e;
                            }
                        }
                        stats.total_energy_captured[lambda_idx] += energy.energy();
                    }
                }
                data_per_patch
            })
            .collect::<Vec<_>>();
        BsdfMeasurementDataPoint { data, stats }
    }
}

/// Represents patches on the surface of the spherical [`Collector`].
///
/// The domain of the whole collector is defined by the [`Collector`].
#[derive(Debug, Clone)]
pub struct CollectorPatches(Vec<Patch>);

impl CollectorPatches {
    /// Checks if the collector patches match the collector scheme.
    pub fn matches_scheme(&self, scheme: &CollectorScheme) -> bool {
        debug_assert!(self.0.len() > 0, "Collector patches must not be empty");
        let is_self_partitioned = matches!(self.0[0], Patch::Partitioned(_));
        let is_scheme_partitioned = matches!(scheme, CollectorScheme::Partitioned { .. });
        is_self_partitioned == is_scheme_partitioned
    }
}

impl Deref for CollectorPatches {
    type Target = [Patch];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for CollectorPatches {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

/// Represents a patch on the spherical [`Collector`].
///
/// It could be a single region or a partitioned region.
#[derive(Debug, Copy, Clone)]
pub enum Patch {
    /// A patch from spherical partitioning.
    Partitioned(PatchPartitioned),
    /// A patch from a single region.
    SingleRegion(PatchSingleRegion),
}

/// Represents a patch issued from partitioning the spherical [`Collector`].
#[derive(Debug, Copy, Clone)]
pub struct PatchPartitioned {
    /// Polar angle range of the patch (in radians).
    pub zenith: (Radians, Radians),

    /// Azimuthal angle range of the patch (in radians).
    pub azimuth: (Radians, Radians),

    /// Polar angle of the patch (in radians).
    pub zenith_center: Radians,

    /// Azimuthal angle of the patch (in radians).
    pub azimuth_center: Radians,

    /// Unit vector of the patch.
    pub unit_vector: Vec3,

    /// Solid angle of the patch (in radians).
    pub solid_angle: SolidAngle,
}

impl PatchPartitioned {
    /// Checks if a unit vector (ray direction) falls into the patch.
    pub fn contains(&self, unit_vector: Vec3A) -> bool {
        let spherical =
            SphericalCoord::from_cartesian(unit_vector.into(), 1.0, Handedness::RightHandedYUp);
        let (zenith, azimuth) = (spherical.zenith, spherical.azimuth);
        let (zenith_start, zenith_stop) = self.zenith;
        let (azimuth_start, azimuth_stop) = self.azimuth;
        zenith >= zenith_start
            && zenith <= zenith_stop
            && azimuth >= azimuth_start
            && azimuth <= azimuth_stop
    }
}

/// Represents a patch issued from a single region of the spherical.
#[derive(Debug, Copy, Clone)]
pub struct PatchSingleRegion {
    /// Polar angle of the patch (in radians).
    pub zenith: Radians,

    /// Azimuthal angle of the patch (in radians).
    pub azimuth: Radians,

    /// Shape of the patch.
    pub shape: RegionShape,

    /// Unit vector of the patch.
    pub unit_vector: Vec3A,

    /// Solid angle of the patch (in radians).
    pub solid_angle: SolidAngle,
}

impl PatchSingleRegion {
    /// Checks if a unit vector (ray direction) falls into the patch.
    pub fn contains(&self, unit_vector: Vec3A) -> bool {
        // TODO: check if this is correct
        match self.shape {
            RegionShape::SphericalCap { zenith } => {
                unit_vector.dot(self.unit_vector) > zenith.cos()
            }
            RegionShape::SphericalRect { zenith, azimuth } => {
                let spherical = SphericalCoord::from_cartesian(
                    unit_vector.into(),
                    1.0,
                    Handedness::RightHandedYUp,
                );
                let (theta, phi) = (spherical.zenith, spherical.azimuth);
                let (zenith_start, zenith_stop) = zenith;
                let (azimuth_start, azimuth_stop) = azimuth;
                theta >= zenith_start
                    && theta <= zenith_stop
                    && phi >= azimuth_start
                    && phi <= azimuth_stop
            }
        }
    }
}

impl Patch {
    /// Creates a new patch.
    ///
    /// # Arguments
    /// * `zenith` - Polar angle range (start, stop) of the patch (in radians).
    /// * `azimuth` - Azimuthal angle range (start, stop) of the patch (in
    ///   radians).
    pub fn new_partitioned(
        zenith: (Radians, Radians),
        azimuth: (Radians, Radians),
        handedness: Handedness,
    ) -> Self {
        let zenith_center = (zenith.0 + zenith.1) / 2.0;
        let azimuth_center = (azimuth.0 + azimuth.1) / 2.0;
        let unit_vector =
            SphericalCoord::new(1.0, zenith_center, azimuth_center).to_cartesian(handedness);
        Self::Partitioned(PatchPartitioned {
            zenith,
            azimuth,
            zenith_center,
            azimuth_center,
            unit_vector,
            solid_angle: SolidAngle::from_angle_ranges(zenith, azimuth),
        })
    }

    /// Returns the default value of [`Self::SingleRegion`] variant.
    pub fn new_single_region(
        shape: RegionShape,
        zenith: Radians,
        azimuth: Radians,
        handedness: Handedness,
    ) -> Self {
        Self::SingleRegion(PatchSingleRegion {
            zenith,
            azimuth,
            shape,
            unit_vector: SphericalCoord::new(1.0, zenith, azimuth)
                .to_cartesian(handedness)
                .into(),
            solid_angle: shape.solid_angle(),
        })
    }

    /// Checks if a unit vector (ray direction) falls into the patch.
    pub fn contains(&self, unit_vector: Vec3A) -> bool {
        match self {
            Self::Partitioned(p) => p.contains(unit_vector),
            Self::SingleRegion(p) => p.contains(unit_vector),
        }
    }

    /// Returns the patch as a partitioned patch.
    pub fn as_partitioned(&self) -> Option<&PatchPartitioned> {
        match self {
            Self::Partitioned(p) => Some(p),
            _ => None,
        }
    }

    /// Returns the patch as a single region patch.
    pub fn as_single_region(&self) -> Option<&PatchSingleRegion> {
        match self {
            Self::SingleRegion(p) => Some(p),
            _ => None,
        }
    }
}

/// Represents the data that a patch can carry.
pub trait PerPatchData: Sized + Clone + Send + Sync + 'static {}

/// Bounce and energy of a patch.
#[derive(Debug, Clone)]
pub struct BounceAndEnergy {
    /// Number of rays hitting the patch at the given bounce.
    pub num_rays_per_bounce: Vec<u32>,
    /// Total energy of rays hitting the patch at the given bounce.
    pub energy_per_bounce: Vec<f32>,
    /// Total energy of rays that hit the patch.
    pub total_energy: f32,
    /// Total number of rays that hit the patch.
    pub total_rays: u32,
}

impl BounceAndEnergy {
    pub fn empty(bounces: usize) -> Self {
        Self {
            num_rays_per_bounce: vec![0; bounces],
            energy_per_bounce: vec![0.0; bounces],
            total_energy: 0.0,
            total_rays: 0,
        }
    }
}

impl PerPatchData for BounceAndEnergy {}

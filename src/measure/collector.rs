use crate::{
    measure::{emitter::RegionShape, measurement::Radius},
    units::{Radians, SolidAngle},
    RangeByStepSize, SphericalDomain, SphericalPartition,
};
use glam::{Vec3, Vec3A};
use std::ops::{Deref, DerefMut};

use crate::{
    app::cache::Cache,
    math::{solve_quadratic, sqr, QuadraticSolution},
    measure::{
        bsdf::{BsdfMeasurementPoint, BsdfStats, PerWavelength, SpectrumSampler},
        measurement::BsdfMeasurement,
        rtc::Trajectory,
    },
    msurf::MicroSurfaceMesh,
    optics::fresnel,
    units::{um, Nanometres},
    Handedness, Medium, SphericalCoord,
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

    /// Returns the shape of the collector only if it is a single region.
    pub fn shape(&self) -> Option<RegionShape> {
        match self {
            Self::Partitioned { .. } => None,
            Self::SingleRegion { shape, .. } => Some(*shape),
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
        match self.scheme {
            CollectorScheme::Partitioned { domain, partition } => {
                CollectorPatches(partition.generate_patches_over_domain(&domain))
            }
            CollectorScheme::SingleRegion {
                zenith, azimuth, ..
            } => {
                let n_zenith = ((zenith.stop - zenith.start) / zenith.step_size).ceil() as usize;
                let n_azimuth =
                    ((azimuth.stop - azimuth.start) / azimuth.step_size).ceil() as usize;

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
                            (spherical, cartesian);
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
    pub fn collect(
        &self,
        desc: &BsdfMeasurement,
        mesh: &MicroSurfaceMesh,
        trajectories: &[Trajectory],
        patches: &CollectorPatches,
        cache: &Cache,
    ) -> (Vec<BsdfMeasurementPoint<PatchBounceEnergy>>, BsdfStats) {
        debug_assert!(
            patches.matches_scheme(&self.scheme),
            "Collector patches do not match the collector scheme"
        );

        let spectrum = SpectrumSampler::from(desc.emitter.spectrum).samples();
        let n_wavelengths = spectrum.len();
        log::debug!("spectrum samples: {:?}", spectrum);

        // Get the refractive indices of the incident and transmitted media for each
        // wavelength.
        let iors_i = cache
            .iors
            .ior_of_spectrum(desc.incident_medium, &spectrum)
            .expect("incident medium IOR not found");
        let iors_t = cache
            .iors
            .ior_of_spectrum(desc.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");

        // Calculate the radius of the collector.
        let radius = match self.radius {
            // FIXME: max_extent() updated, thus 2.5 is not a good choice
            Radius::Auto(_) => um!(mesh.bounds.max_extent() * 2.5),
            Radius::Fixed(r) => r.in_micrometres(),
        };

        let domain = self.scheme.domain();

        let max_bounces = desc.emitter.max_bounces as usize;

        let mut stats = BsdfStats {
            n_emitted: desc.emitter.num_rays,
            n_received: 0,
            wavelength: spectrum.clone(),
            n_reflected: PerWavelength(vec![0; n_wavelengths]),
            n_absorbed: PerWavelength(vec![0; n_wavelengths]),
            n_captured: PerWavelength(vec![0; n_wavelengths]),
            total_energy_emitted: desc.emitter.num_rays as f32,
            total_energy_captured: PerWavelength(vec![0.0; n_wavelengths]),
            num_rays_per_bounce: PerWavelength(vec![vec![0; max_bounces]; n_wavelengths]),
            energy_per_bounce: PerWavelength(vec![vec![0.0; max_bounces]; n_wavelengths]),
        };

        // Convert the last rays of the trajectories into a vector located
        // at the collector's center and pointing to the intersection point
        // of the last ray with the collector's surface. For later use classify
        // the rays according to the patch they intersect.
        //
        // Each element of the vector is a tuple containing the index of the
        // trajectory, the intersection point and the number of bounces.
        let mut unit_dirs = trajectories
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
                            let c = last.org.dot(last.org) - sqr(radius.as_f32());
                            let p = match solve_quadratic(a, b, c) {
                                QuadraticSolution::None | QuadraticSolution::One(_) => {
                                    unreachable!(
                                        "Ray starting inside the collector's surface, it should \
                                         have more than one intersection point."
                                    )
                                }
                                QuadraticSolution::Two(t, _) => last.org + last.dir * t,
                            };
                            Some((i, p.normalize(), trajectory.len() - 1))
                        }
                    }
                }
            })
            .collect::<Vec<_>>();

        // Calculate the energy of the rays (with wavelength) that intersected the
        // collector's surface.
        // Outer index: ray index, inner index: wavelength index.
        let ray_energy_per_wavelength = unit_dirs
            .iter()
            .map(|(idx, vector, _)| {
                let trajectory = &trajectories[*idx];
                let mut energy = Vec::with_capacity(n_wavelengths);
                energy.resize(n_wavelengths, Energy::Reflected(1.0));
                for node in trajectories[*idx].iter().take(trajectory.len() - 1) {
                    for (i, lambda) in spectrum.iter().enumerate() {
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
                (idx, energy)
            })
            .collect::<Vec<_>>();

        // per ray
        for ray_energy in &ray_energy_per_wavelength {
            // per wavelength
            for (idx_wavelength, energy) in ray_energy.1.iter().enumerate() {
                match energy {
                    Energy::Absorbed => stats.n_absorbed[idx_wavelength] += 1,
                    Energy::Reflected(_) => stats.n_reflected[idx_wavelength] += 1,
                }
            }
        }

        let mut measurement_points = patches
            .iter()
            .map(|patch| BsdfMeasurementPoint {
                patch: *patch,
                data: PerWavelength(vec![]),
            })
            .collect::<Vec<_>>();

        let dirs_per_patch = patches.iter().map(|patch| {
            // Retrieve the ray indices (of trajectories) that intersect the patch.
            unit_dirs
                .iter()
                .filter_map(|(i, v, bounce)| patch.contains(*v).then(|| (*i, *bounce)))
                .collect::<Vec<_>>()
        });

        measurement_points
            .iter_mut()
            .zip(dirs_per_patch)
            .for_each(|(measurement_point, dirs)| {
                let mut data = PerWavelength(vec![
                    PatchBounceEnergy::empty(
                        desc.emitter.max_bounces as usize
                    );
                    spectrum.len()
                ]);
                for (i, bounce) in dirs.iter() {
                    let ray_energy = &ray_energy_per_wavelength
                        .iter()
                        .find(|(j, energy)| **j == *i)
                        .unwrap()
                        .1;
                    for (wavelength_idx, energy) in ray_energy.iter().enumerate() {
                        match energy {
                            Energy::Absorbed => continue,
                            Energy::Reflected(e) => {
                                stats.n_captured[wavelength_idx] += 1;
                                data[wavelength_idx].total_energy += e;
                                data[wavelength_idx].total_rays += 1;
                                data[wavelength_idx].energy_per_bounce[*bounce - 1] += e;
                                data[wavelength_idx].num_rays_per_bounce[*bounce - 1] += 1;
                                stats.num_rays_per_bounce[wavelength_idx][*bounce - 1] += 1;
                                stats.energy_per_bounce[wavelength_idx][*bounce - 1] += e;
                            }
                        }
                        stats.total_energy_captured[wavelength_idx] += energy.energy();
                    }
                }
                measurement_point.data = data;
            });

        (measurement_points, stats)
    }

    // todo: pub fn collect_grid_rt

    pub fn save_stats(&self, _path: &str) { todo!("Collector::save_stats") }
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
    Partitioned(PatchPartitioned),
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

    pub fn contains(&self, unit_vector: Vec3A) -> bool {
        match self {
            Self::Partitioned(p) => p.contains(unit_vector),
            Self::SingleRegion(p) => p.contains(unit_vector),
        }
    }

    pub fn as_partitioned(&self) -> Option<&PatchPartitioned> {
        match self {
            Self::Partitioned(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_single_region(&self) -> Option<&PatchSingleRegion> {
        match self {
            Self::SingleRegion(p) => Some(p),
            _ => None,
        }
    }
}

/// Bounce and energy of a patch.
#[derive(Debug, Clone)]
pub struct PatchBounceEnergy {
    /// Number of rays hitting the patch at the given bounce.
    pub num_rays_per_bounce: Vec<u32>,
    /// Total energy of rays hitting the patch at the given bounce.
    pub energy_per_bounce: Vec<f32>,
    /// Total energy of rays that hit the patch.
    pub total_energy: f32,
    /// Total number of rays that hit the patch.
    pub total_rays: u32,
}

impl PatchBounceEnergy {
    pub fn empty(bounces: usize) -> Self {
        Self {
            num_rays_per_bounce: vec![0; bounces],
            energy_per_bounce: vec![0.0; bounces],
            total_energy: 0.0,
            total_rays: 0,
        }
    }
}

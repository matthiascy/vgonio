use crate::{
    app::cache::Cache,
    measure::{
        bsdf::{
            rtc::RayTrajectory, BsdfMeasurementDataPoint, BsdfMeasurementStatsPoint, PerWavelength,
        },
        params::BsdfMeasurementParams,
    },
    optics::fresnel,
    SphericalDomain,
};
use serde::{Deserialize, Serialize};
use vgcore::{
    math,
    math::{Sph2, Vec3, Vec3A},
    units::{rad, Radians},
};
use vgsurf::MicroSurfaceMesh;

/// Description of a detector collecting the data.
///
/// The virtual goniophotometer's sensors are represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
///
/// A detector is defined by its domain, the precision of the
/// measurements and the partitioning scheme.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorParams {
    /// Domain of the collector.
    pub domain: SphericalDomain,
    /// Zenith angle step size (in radians).
    pub precision: Radians,
    /// Partitioning scheme of the collector.
    pub scheme: DetectorScheme,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectorScheme {
    /// Partition scheme based on "A general rule for disk and hemisphere
    /// partition into equal-area cells" by Benoit Beckers et Pierre Beckers.
    Beckers = 0x00,
    /// Partition scheme based on "Subdivision of the sky hemisphere for
    /// luminance measurements" by P.R. Tregenza.
    Tregenza = 0x01,
}

/// Partitioned patches of the collector.
#[derive(Debug, Clone)]
pub enum DetectorPatches {
    Beckers {
        /// The annuli of the collector.
        rings: Vec<Ring>,
        /// The patches of the collector.
        patches: Vec<Patch>,
    },
    // TODO: implement Tregenza
    Tregenza,
}

#[derive(Debug, Copy, Clone)]
pub struct Patch {
    /// Minimum zenith (theta) and azimuth (phi) angles of the patch.
    pub min: Sph2,
    /// Maximum zenith (theta) and azimuth (phi) angles of the patch.
    pub max: Sph2,
}

impl Patch {
    pub fn new(theta_min: Radians, theta_max: Radians, phi_min: Radians, phi_max: Radians) -> Self {
        Self {
            min: Sph2::new(theta_min, phi_min),
            max: Sph2::new(theta_max, phi_max),
        }
    }

    /// Returns the center of the patch.
    pub fn center(&self) -> Sph2 {
        Sph2::new(
            (self.min.theta + self.max.theta) / 2.0,
            (self.min.phi + self.max.phi) / 2.0,
        )
    }
}

/// A segment in form of an annulus of the collector.
#[derive(Debug, Copy, Clone)]
struct Ring {
    /// Minimum theta angle of the annulus.
    pub theta_inner: f32,
    /// Maximum theta angle of the annulus.
    pub theta_outer: f32,
    /// Step size of the phi angle inside the annulus.
    pub phi_step: f32,
    /// Number of patches in the annulus: 2 * pi / phi_step == patch_count.
    pub patch_count: usize,
    /// Base index of the annulus in the patches buffer.
    pub base_index: usize,
}

mod beckers {
    use vgcore::math::sqr;

    /// Computes the number of cells inside the external circle of the ring.
    pub fn compute_ks(k0: u32, num_rings: u32) -> Vec<u32> {
        let mut ks = vec![0; num_rings as usize];
        ks[0] = k0;
        let sqrt_pi = std::f32::consts::PI.sqrt();
        for i in 1..num_rings as usize {
            ks[i] = sqr(f32::sqrt(ks[i - 1] as f32) + sqrt_pi).round() as u32;
        }
        ks
    }

    /// Computes the radius of the rings.
    pub fn compute_rs(ks: &[u32], num_rings: u32, radius: f32) -> Vec<f32> {
        let mut rs = vec![0.0; num_rings as usize];
        rs[0] = radius * f32::sqrt(ks[0] as f32 / ks[num_rings as usize - 1] as f32);
        for i in 0..num_rings as usize {
            rs[i] = (ks[i] as f32 / ks[i - 1] as f32).sqrt() * rs[i - 1]
        }
        rs
    }

    /// Computes the zenith angle of the rings on the hemisphere.
    pub fn compute_ts(rs: &[f32]) -> Vec<f32> {
        rs.iter().map(|r| 2.0 * (r / 2.0).asin()).collect()
    }
}

impl DetectorParams {
    /// Generates the patches of the collector.
    ///
    /// The patches are generated based on the scheme of the collector. They are
    /// used to collect the data. The patches are generated in the order of
    /// the azimuth angle first, then the zenith angle.
    pub fn generate_patches(&self) -> DetectorPatches {
        match self.scheme {
            DetectorScheme::Beckers => {
                let num_rings = (Radians::HALF_PI / self.precision).round() as u32;
                let ks = beckers::compute_ks(1, num_rings);
                let rs = beckers::compute_rs(&ks, num_rings, f32::sqrt(2.0));
                let ts = beckers::compute_ts(&rs);
                log::debug!("ks: {:?}", ks);
                log::debug!("rs: {:?}", rs);
                log::debug!("ts: {:?}", ts);
                let mut patches = Vec::with_capacity(ks[num_rings as usize - 1] as usize);
                let mut rings = Vec::with_capacity(num_rings as usize);
                for (i, (t, k)) in ts.iter().zip(ks.iter()).enumerate() {
                    log::trace!("Ring {}: t = {}, k = {}", i, t.to_degrees(), k);
                    let k_prev = if i == 0 { 0 } else { ks[i - 1] };
                    let n = k - k_prev;
                    let t_prev = if i == 0 { 0.0 } else { ts[i - 1] };
                    let phi_step = Radians::TWO_PI / n as f32;
                    rings.push(Ring {
                        theta_inner: t_prev,
                        theta_outer: *t,
                        phi_step: phi_step.as_f32(),
                        patch_count: n as usize,
                        base_index: patches.len(),
                    });
                    for j in 0..n {
                        let phi_min = phi_step * j as f32;
                        let phi_max = phi_step * (j + 1) as f32;
                        patches.push(Patch::new(
                            t_prev.into(),
                            rad!(*t),
                            phi_min.into(),
                            phi_max.into(),
                        ));
                    }
                }
                DetectorPatches::Beckers { rings, patches }
            }
            DetectorScheme::Tregenza => {
                todo!("Tregenza partitioning scheme is not implemented yet")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Detector {
    params: DetectorParams,
    patches: DetectorPatches,
}

impl Detector {
    pub fn new(params: &DetectorParams) -> Self {
        Self {
            params: *params,
            patches: params.generate_patches(),
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
        pos: Sph2,
        trajectories: &[RayTrajectory],
        cache: &Cache,
    ) -> BsdfMeasurementDataPoint<BounceAndEnergy> {
        // TODO: use generic type for the data point
        log::debug!(
            "[Collector] collecting data for BSDF measurement at position {}",
            pos
        );

        let spectrum = params.emitter.spectrum.values().collect::<Vec<_>>();
        let n_wavelengths = spectrum.len();
        log::debug!("[Collector] spectrum samples: {:?}", spectrum);

        // Get the refractive indices of the incident and transmitted media for each
        // wavelength.
        let iors_i = cache
            .iors
            .ior_of_spectrum(params.incident_medium, &spectrum)
            .expect("incident medium IOR not found");
        log::debug!("[Collector] incident medium IORs: {:?}", iors_i);
        let iors_t = cache
            .iors
            .ior_of_spectrum(params.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");
        log::debug!("[Collector] transmitted medium IORs: {:?}", iors_t);

        // Calculate the radius of the collector.
        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        let disc_radius = crate::measure::estimate_disc_radius(mesh);
        let max_bounces = params.emitter.max_bounces as usize;
        let mut stats = BsdfMeasurementStatsPoint::new(n_wavelengths, max_bounces);
        log::trace!(
            "[Collector] Estimated orbit radius: {}, shape radius: {:?}",
            orbit_radius,
            disc_radius
        );

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
                        // 1. Calculate the intersection point of the last ray with
                        // the collector's surface. Ray-Sphere intersection.
                        // Collect's center is at (0, 0, 0).
                        let a = last.dir.dot(last.dir);
                        let b = 2.0 * last.dir.dot(last.org);
                        let c = last.org.dot(last.org) - math::sqr(orbit_radius);
                        let p = match math::solve_quadratic(a, b, c) {
                            math::QuadraticSolution::None | math::QuadraticSolution::One(_) => {
                                unreachable!(
                                    "Ray starting inside the collector's surface, it should have \
                                     more than one intersection point."
                                )
                            }
                            math::QuadraticSolution::Two(_, t) => last.org + last.dir * t,
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
            })
            .collect::<Vec<_>>();

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

        // log::trace!("ray_energy_per_wavelength: {:?}", ray_energy_per_wavelength);

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
        let outgoing_dirs_per_patch = self
            .patches
            .iter()
            .map(|patch| {
                // Retrieve the ray indices (of trajectories) that intersect the patch.
                outgoing_dirs
                    .iter()
                    .filter_map(|outgoing| {
                        match patch {
                            Patch::Partitioned(p) => p.contains(outgoing.dir),
                        }
                        .then_some((outgoing.idx, outgoing.bounce))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // log::trace!("outgoing_dirs_per_patch: {:?}", outgoing_dirs_per_patch);

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
                        .find(|(j, _)| *j == *i)
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
                        stats.e_captured[lambda_idx] += energy.energy();
                    }
                }
                data_per_patch
            })
            .collect::<Vec<_>>();

        // Compute the vertex positions of the outgoing rays.
        #[cfg(debug_assertions)]
        let outgoing_vertex_positions: Vec<Vec3> = outgoing_dirs
            .iter()
            .map(|outgoing| (outgoing.dir * orbit_radius).into())
            .collect::<Vec<_>>();

        BsdfMeasurementDataPoint {
            data,
            stats,
            #[cfg(debug_assertions)]
            trajectories: trajectories.to_vec(),
            #[cfg(debug_assertions)]
            hit_points: outgoing_vertex_positions,
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

/// Represents the data that a patch can carry.
pub trait PerPatchData: Sized + Clone + Send + Sync + 'static {}

/// Bounce and energy of a patch.
///
/// Length of `num_rays_per_bounce` and `energy_per_bounce` is equal to the
/// maximum number of bounces.
#[derive(Debug, Clone)]
pub struct BounceAndEnergy {
    /// Total number of rays that hit the patch.
    pub total_rays: u32,
    /// Total energy of rays that hit the patch.
    pub total_energy: f32,
    /// Number of rays hitting the patch at the given bounce.
    pub num_rays_per_bounce: Vec<u32>,
    /// Total energy of rays hitting the patch at the given bounce.
    pub energy_per_bounce: Vec<f32>,
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

    pub const fn calc_size_in_bytes(bounces: usize) -> usize { 8 * bounces + 8 }
}

impl PartialEq for BounceAndEnergy {
    fn eq(&self, other: &Self) -> bool {
        self.total_rays == other.total_rays
            && self.total_energy == other.total_energy
            && self.num_rays_per_bounce == other.num_rays_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}

impl PerPatchData for BounceAndEnergy {}

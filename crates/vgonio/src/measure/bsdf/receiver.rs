//! Sensor of the virtual gonio-reflectometer.

use crate::{
    app::cache::{Handle, RawCache},
    measure::{
        bsdf::{
            BsdfMeasurementStatsPoint, BsdfSnapshot, BsdfSnapshotRaw, SimulationResultPoint,
            SpectralSamples,
        },
        params::BsdfMeasurementParams,
    },
    partition::{PartitionScheme, SphericalPartition},
    RangeByStepSizeInclusive, SphericalDomain,
};
use base::{
    math::{rcp_f32, Sph2, Vec3, Vec3A},
    optics::{fresnel, ior::RefractiveIndex},
    units::{Nanometres, Radians},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic;
use surf::MicroSurface;

/// Data collected by the receiver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Hash, Default)]
pub enum DataRetrieval {
    /// The full data are collected.
    #[serde(rename = "full-data")]
    FullData = 0x00,
    /// Only the BSDF values are collected.
    #[serde(rename = "bsdf-only")]
    #[default]
    BsdfOnly = 0x01,
}

impl From<u8> for DataRetrieval {
    fn from(v: u8) -> Self {
        match v {
            0x00 => Self::FullData,
            0x01 => Self::BsdfOnly,
            _ => panic!("invalid data retrieval mode"),
        }
    }
}

/// Description of a receiver collecting the data.
///
/// The virtual goniophotometer's sensors are represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
///
/// A receiver is defined by its domain, the precision of the
/// measurements and the partitioning scheme.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceiverParams {
    /// Domain of the collector.
    pub domain: SphericalDomain,
    /// Step size of the zenith and azimuth angles.
    /// Azimuth angle precision is only used for the EqualAngle partitioning
    /// scheme.
    pub precision: Sph2,
    /// Partitioning scheme of the globe.
    pub scheme: PartitionScheme,
    /// Type of data to retrieve.
    pub retrieval: DataRetrieval,
}

impl ReceiverParams {
    /// Returns the number of patches of the collector.
    pub fn num_patches(&self) -> usize {
        let num_patches_hemi = match self.scheme {
            PartitionScheme::Beckers => {
                let num_rings = (Radians::HALF_PI / self.precision.theta).round() as u32;
                let ks = crate::partition::beckers::compute_ks(1, num_rings);
                ks[num_rings as usize - 1] as usize
            }
            PartitionScheme::EqualAngle => {
                let num_rings = (Radians::HALF_PI / self.precision.theta).round() as usize + 1;
                let num_patches_per_ring = RangeByStepSizeInclusive::new(
                    Radians::ZERO,
                    Radians::TWO_PI,
                    self.precision.phi,
                )
                .step_count_wrapped();
                num_rings * num_patches_per_ring
            }
        };
        if self.domain == SphericalDomain::Whole {
            return num_patches_hemi * 2;
        }
        num_patches_hemi
    }

    // TODO: constify this
    /// Returns the number rings of the collector.
    pub fn num_rings(&self) -> usize {
        let num_rings_hemi =
            (self.domain.zenith_angle_diff() / self.precision.theta).round() as usize;
        if self.domain == SphericalDomain::Whole {
            return num_rings_hemi * 2;
        }
        num_rings_hemi
    }

    /// Partition the receiver into patches.
    ///
    /// The patches are generated based on the scheme of the collector. They are
    /// used to collect the data. The patches are generated in the order of
    /// the azimuth angle first, then the zenith angle.
    pub fn partitioning(&self) -> SphericalPartition {
        SphericalPartition::new(self.scheme, self.domain, self.precision)
    }
}

/// Sensor of the virtual gonio-reflectometer.
#[derive(Debug, Clone)]
pub struct Receiver {
    /// The parameters of the receiver.
    params: ReceiverParams,
    /// Wavelengths of the measurement.
    pub spectrum: Vec<Nanometres>,
    /// Incident medium's refractive indices.
    pub iors_i: Box<[RefractiveIndex]>,
    /// Transmitted medium's refractive indices.
    pub iors_t: Box<[RefractiveIndex]>,
    /// The partitioned patches of the receiver.
    pub patches: SphericalPartition,
}

/// Outgoing ray of a trajectory [`RayTrajectory`].
///
/// Used during the data collection process.
#[derive(Debug, Clone)]
struct OutgoingRay {
    /// Index of the ray in the trajectory.
    pub ray_idx: u32,
    /// The final direction of the ray.
    pub ray_dir: Vec3A,
    /// The number of bounces of the ray.
    pub bounce: u32,
    /// The energy of the ray per wavelength.
    pub energy: Vec<Energy>,
    /// Index of the patch where the ray is collected.
    pub patch_idx: usize,
}

impl Receiver {
    /// Creates a new receiver.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the receiver.
    pub fn new(
        receiver_params: &ReceiverParams,
        meas_params: &BsdfMeasurementParams,
        cache: &RawCache,
    ) -> Self {
        let spectrum = meas_params.emitter.spectrum.values().collect::<Vec<_>>();
        // Retrieve the incident medium's refractive indices for each wavelength.
        let iors_i = cache
            .iors
            .ior_of_spectrum(meas_params.incident_medium, &spectrum)
            .expect("incident medium IOR not found");
        // Retrieve the transmitted medium's refractive indices for each wavelength.
        let iors_t = cache
            .iors
            .ior_of_spectrum(meas_params.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");
        Self {
            params: *receiver_params,
            spectrum,
            iors_i,
            iors_t,
            patches: receiver_params.partitioning(),
        }
    }

    /// Collects the ray-tracing data of one measurement point.
    ///
    /// Returns the collected data and the statistics of the BSDF.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the BSDF measurement.
    /// * `sim_res` - The simulation results.
    /// * `cache` - The cache where all the data are stored.
    ///
    /// # Returns
    ///
    /// The collected data for each simulation result which is a vector of
    /// [`BsdfSnapshotRaw`].
    pub fn collect(
        &self,
        result: &SimulationResultPoint,
        collected: &mut CollectedData<'_>,
        orbit_radius: f32,
    ) {
        const CHUNK_SIZE: usize = 1024;
        // TODO: deal with the domain of the receiver
        let spectrum_len = self.spectrum.len();

        #[cfg(feature = "bench")]
        let start = std::time::Instant::now();
        let n_escaped = atomic::AtomicU32::new(0);
        let n_bounce = atomic::AtomicU32::new(0);
        let n_received = atomic::AtomicU32::new(0);
        // Convert the last rays of the trajectories into a vector located
        // at the center of the collector.
        let dirs: Box<[OutgoingRay]> = result
            .trajectories
            .par_chunks(CHUNK_SIZE)
            .enumerate()
            .flat_map(|(chunk_idx, trajectories)| {
                trajectories
                    .iter()
                    .enumerate()
                    .filter_map(|(i, trajectory)| {
                        let ray_idx = chunk_idx * CHUNK_SIZE + i;
                        match trajectory.last() {
                            None => None,
                            Some(last) => {
                                // 1. Get the outgoing ray direction it's the last ray of the
                                //    trajectory.
                                let ray_dir = last.dir.normalize();
                                // 2. Calculate the energy of the ray per wavelength.
                                let mut energy = vec![Energy::Reflected(1.0); spectrum_len];
                                for node in trajectory.iter().take(trajectory.len() - 1) {
                                    for i in 0..spectrum_len {
                                        match energy[i] {
                                            Energy::Absorbed => continue,
                                            Energy::Reflected(ref mut e) => {
                                                *e *= fresnel::reflectance(
                                                    node.cos.unwrap_or(1.0),
                                                    self.iors_i[i],
                                                    self.iors_t[i],
                                                );
                                                if *e <= 0.0 {
                                                    energy[i] = Energy::Absorbed;
                                                }
                                            }
                                        }
                                    }
                                }

                                // 3. Compute the index of the patch where the ray is collected.
                                let patch_idx = match self
                                    .patches
                                    .contains(Sph2::from_cartesian(ray_dir.into()))
                                {
                                    Some(idx) => idx,
                                    None => {
                                        n_escaped.fetch_add(1, atomic::Ordering::Relaxed);
                                        return None;
                                    }
                                };

                                // 4. Update the maximum number of bounces.
                                let bounce = (trajectory.len() - 1) as u32;
                                n_bounce.fetch_max(bounce, atomic::Ordering::Relaxed);

                                // 5. Update the number of received rays.
                                n_received.fetch_add(1, atomic::Ordering::Relaxed);

                                // Returns the index of the ray, the unit vector pointing to
                                // the collector's surface, and the number of bounces.
                                Some(OutgoingRay {
                                    ray_idx: ray_idx as u32,
                                    ray_dir,
                                    bounce,
                                    energy,
                                    patch_idx,
                                })
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        log::debug!("n_escaped: {}", n_escaped.load(atomic::Ordering::Relaxed));

        let max_bounces = n_bounce.load(atomic::Ordering::Relaxed) as usize;

        let mut stats = BsdfMeasurementStatsPoint::new(spectrum_len, max_bounces);
        stats.n_received = n_received.load(atomic::Ordering::Relaxed);

        // #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
        // log::debug!("process dirs: {:?}", dirs);

        #[cfg(feature = "bench")]
        let dirs_proc_time = start.elapsed().as_millis();
        #[cfg(feature = "bench")]
        log::info!("Collector::collect: dirs: {} ms", dirs_proc_time);

        for dir in dirs.iter() {
            for i in 0..spectrum_len {
                match dir.energy[i] {
                    Energy::Absorbed => stats.n_absorbed[i] += 1,
                    Energy::Reflected(_) => stats.n_reflected[i] += 1,
                }
            }
        }

        log::debug!("stats.n_absorbed: {:?}", stats.n_absorbed);
        log::debug!("stats.n_reflected: {:?}", stats.n_reflected);
        log::debug!("stats.n_received: {:?}", stats.n_received);

        let mut data =
            vec![
                SpectralSamples::splat(BounceAndEnergy::empty(max_bounces), spectrum_len);
                self.patches.num_patches()
            ];

        // Compute the vertex positions of the outgoing rays.
        #[cfg(any(feature = "visu-dbg", debug_assertions))]
        let mut outgoing_intersection_points = vec![Vec3::ZERO; dirs.len()];

        for (j, dir) in dirs.iter().enumerate() {
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            {
                outgoing_intersection_points[j] = (dir.ray_dir * orbit_radius).into();
            }
            let patch = &mut data[dir.patch_idx];
            let bounce_idx = dir.bounce as usize - 1;
            for (i, energy) in dir.energy.iter().enumerate() {
                stats.e_captured[i] += energy.energy();
                match energy {
                    Energy::Absorbed => continue,
                    Energy::Reflected(e) => {
                        stats.n_captured[i] += 1;
                        patch[i].total_energy += e;
                        patch[i].total_rays += 1;
                        patch[i].energy_per_bounce[bounce_idx] += e;
                        patch[i].num_rays_per_bounce[bounce_idx] += 1;
                        stats.num_rays_per_bounce[i][bounce_idx] += 1;
                        stats.energy_per_bounce[i][bounce_idx] += e;
                    }
                }
            }
        }
        #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
        log::debug!("data per patch: {:?}", data);
        #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
        log::debug!("stats: {:?}", stats);

        #[cfg(feature = "bench")]
        let data_time = start.elapsed().as_millis();
        #[cfg(feature = "bench")]
        log::info!(
            "Collector::collect: data: {} ms",
            data_time - dirs_proc_time
        );

        collected.snapshots.push(BsdfSnapshotRaw {
            w_i: result.w_i,
            records: data.into_boxed_slice(),
            stats,
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            trajectories: result.trajectories.to_vec(),
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            hit_points: outgoing_intersection_points,
        });
    }
}

/// Collected data by the receiver for a specific micro-surface.
///
/// The data are collected for patches of the receiver and for each
/// wavelength of the incident spectrum.
#[derive(Debug, Clone)]
pub struct CollectedData<'a> {
    /// The micro-surface where the data were collected.
    pub surface: Handle<MicroSurface>,
    /// The partitioned patches of the receiver.
    pub partition: &'a SphericalPartition,
    /// The collected data.
    pub snapshots: Vec<BsdfSnapshotRaw<BounceAndEnergy>>,
}

impl<'a> CollectedData<'a> {
    /// Creates an empty collected data.
    pub fn empty(surf: Handle<MicroSurface>, partition: &'a SphericalPartition) -> Self {
        Self {
            surface: surf,
            partition,
            snapshots: vec![],
        }
    }

    /// Computes the BSDF according to the collected data.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the BSDF measurement.
    ///
    /// # Returns
    ///
    /// The BSDF computed from the collected data for each wavelength of the
    /// incident spectrum. The output vector is of size equal to the number of
    /// measurement points of the emitter times the number of patches of the
    /// receiver.
    pub fn compute_bsdf(&self, params: &BsdfMeasurementParams) -> Box<[BsdfSnapshot]> {
        // For each snapshot (w_i), compute the BSDF.
        log::info!(
            "Computing BSDF... with {} patches",
            params.receiver.num_patches()
        );
        self.snapshots
            .par_iter()
            .map(|snapshot| {
                let mut samples = {
                    let mut samples = Box::new_uninit_slice(snapshot.records.len());
                    for spectral_samples in samples.iter_mut() {
                        spectral_samples.write(SpectralSamples::splat(
                            0.0,
                            params.emitter.spectrum.values().len(),
                        ));
                    }
                    unsafe { samples.assume_init() }
                };

                let cos_i = snapshot.w_i.theta.cos();
                let l_i = snapshot.stats.n_received as f32 * cos_i;
                for (i, patch_data) in snapshot.records.iter().enumerate() {
                    // Per wavelength
                    for (j, stats) in patch_data.iter().enumerate() {
                        let patch = self.partition.patches.get(i).unwrap();
                        let cos_o = patch.center().theta.cos();
                        if cos_o == 0.0 {
                            samples[i][j] = 0.0;
                        } else {
                            let l_o = stats.total_energy * rcp_f32(cos_o);
                            samples[i][j] = l_o * rcp_f32(l_i);
                            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                            log::debug!(
                                "energy of patch {i}: {}, λ[{j}] --  L_i: {}, L_o[{i}]: {} -- \
                                 brdf: {}",
                                stats.total_energy,
                                l_i,
                                l_o,
                                samples[i][j],
                            );
                        }
                    }
                }
                // #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                // log::debug!("snapshot.samples, w_i = {:?}: {:?}", snapshot.w_i, samples);
                BsdfSnapshot {
                    w_i: snapshot.w_i,
                    samples,
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    trajectories: snapshot.trajectories.clone(),
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    hit_points: snapshot.hit_points.clone(),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
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
#[derive(Debug, Clone, Default)]
pub struct BounceAndEnergy {
    /// Maximum number of bounces of rays hitting the patch.
    pub n_bounces: u32,
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
    /// Creates a new bounce and energy.
    pub fn empty(bounces: usize) -> Self {
        Self {
            n_bounces: bounces as u32,
            num_rays_per_bounce: vec![0; bounces],
            energy_per_bounce: vec![0.0; bounces],
            total_energy: 0.0,
            total_rays: 0,
        }
    }
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

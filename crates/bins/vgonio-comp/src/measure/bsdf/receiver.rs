//! Sensor of the virtual gonio-reflectometer.
#[cfg(feature = "vdbg")]
use crate::measure::bsdf::rtc::RayTrajectory;
#[cfg(feature = "vdbg")]
use vgonio_core::math::{Vec3, Vec3A};

use crate::{
    app::cache::RawCache,
    measure::{
        bsdf::{SingleBsdfMeasurementStats, SingleSimResult},
        params::BsdfMeasurementParams,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::{atomic, atomic::AtomicU64};
use vgonio_core::{
    math::Sph2,
    optics::ior::Ior,
    units::{Nanometres, Radians},
    utils::{
        partition::{PartitionScheme, SphericalDomain, SphericalPartition},
        range::StepRangeIncl,
    },
};

/// Description of a receiver collecting the data.
///
/// The virtual goniophotometer's sensors are represented by the patches
/// of a sphere or a hemisphere positioned around the specimen.
///
/// A receiver is defined by its domain, the precision of the
/// measurements and the partitioning scheme.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceiverParams {
    /// Domain of the collector.
    pub domain: SphericalDomain,
    /// Angular resolution of the collector.
    pub precision: Sph2,
    /// Partitioning schema of the globe.
    pub scheme: PartitionScheme,
}

impl ReceiverParams {
    /// Returns the number of patches of the collector.
    pub fn num_patches(&self) -> usize {
        let num_patches_hemi = match self.scheme {
            PartitionScheme::Beckers => {
                let num_rings = (Radians::HALF_PI / self.precision.theta).round() as u32;
                let ks = vgonio_core::utils::partition::beckers::compute_ks(1, num_rings);
                ks[num_rings as usize - 1] as usize
            },
            PartitionScheme::EqualAngle => {
                let num_rings = (Radians::HALF_PI / self.precision.theta).round() as usize + 1;
                let num_patches_per_ring =
                    StepRangeIncl::new(Radians::ZERO, Radians::TWO_PI, self.precision.phi)
                        .step_count_wrapped();
                num_rings * num_patches_per_ring
            },
        };
        match self.domain {
            SphericalDomain::Whole => num_patches_hemi * 2,
            _ => num_patches_hemi,
        }
    }

    // TODO: constify this
    /// Returns the number rings of the collector.
    pub fn num_rings(&self) -> usize {
        let num_rings_hemi =
            (self.domain.zenith_angle_diff() / self.precision.theta).round() as usize;
        match self.domain {
            SphericalDomain::Whole => num_rings_hemi * 2,
            _ => num_rings_hemi,
        }
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
    pub params: ReceiverParams,
    /// Wavelengths of the measurement.
    pub spectrum: Box<[Nanometres]>,
    /// Incident medium's refractive indices.
    pub iors_i: Box<[Ior]>,
    /// Transmitted medium's refractive indices.
    pub iors_t: Box<[Ior]>,
    /// The partitioned patches of the receiver.
    pub patches: SphericalPartition,
}

/// Outgoing ray of a trajectory [`RayTrajectory`].
///
/// Used during the data collection process.
#[cfg(feature = "vdbg")]
#[derive(Debug, Clone)]
struct OutgoingRay {
    #[allow(unused)]
    /// Index of the ray in the trajectory.
    pub ray_idx: u32,
    /// The final direction of the ray.
    pub ray_dir: Vec3A,
    /// The number of bounces of the ray.
    pub bounce: u32,
    /// The energy of the ray per wavelength.
    pub energy_per_wavelength: Vec<Energy>,
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
        let spectrum = meas_params.emitter.spectrum.values().collect::<Box<_>>();
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

    /// Returns the number of outgoing directions of the receiver.
    pub const fn n_wo(&self) -> usize { self.patches.n_patches() }

    /// Collects the ray-tracing data of one measurement point.
    ///
    /// Returns the collected data and the statistics of the BSDF.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the BSDF measurement.
    /// * `sim_res` - The simulation results.
    /// * `cache` - The cache where all the data are stored.
    pub fn collect(
        &self,
        result: &SingleSimResult,
        out_stats: &mut Option<SingleBsdfMeasurementStats>,
        records: &mut [Option<BounceAndEnergy>],
        #[cfg(feature = "vdbg")] orbit_radius: f32,
        #[cfg(feature = "vdbg")] fresnel: bool,
        #[cfg(feature = "vdbg")] out_trajs: &mut Vec<RayTrajectory>,
        #[cfg(feature = "vdbg")] out_hpnts: &mut Vec<Vec3>,
    ) {
        debug_assert!(
            records.len() == self.patches.n_patches() * self.spectrum.len(),
            "records length mismatch"
        );
        const CHUNK_SIZE: usize = 4096;
        // TODO: deal with the domain of the receiver
        let n_spectrum = self.spectrum.len();

        #[cfg(feature = "bench")]
        let start = std::time::Instant::now();

        #[cfg(feature = "vdbg")]
        {
            use rayon::prelude::*;
            use std::sync::atomic::AtomicU32;
            use vgonio_core::optics::fresnel;

            log::debug!(
                "SingleSimResult at {}, {} rays",
                result.wi,
                result.trajectories.len(),
            );
            let (n_bounce, mut stats, dirs) = {
                let n_bounce = AtomicU32::new(0);
                let n_received = AtomicU64::new(0);
                let n_missed = AtomicU64::new(0);
                let mut n_reflected = Vec::with_capacity(n_spectrum);
                let mut n_escaped = Vec::with_capacity(n_spectrum);
                for _ in 0..n_spectrum {
                    n_reflected.push(AtomicU64::new(0));
                    n_escaped.push(AtomicU64::new(0));
                }
                n_reflected.shrink_to_fit();
                n_escaped.shrink_to_fit();

                log::debug!("Initial trajectories count: {}", result.trajectories.len());

                // Convert the last rays of the trajectories into a vector located
                // at the centre of the collector.
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
                                    None => {
                                        n_missed.fetch_add(1, atomic::Ordering::Relaxed);
                                        None
                                    },
                                    Some(last) => {
                                        // 1. Get the outgoing ray direction it's the last ray of
                                        //    the trajectory.
                                        let ray_dir = last.dir.normalize();
                                        // 2. Calculate the energy of the ray per wavelength, the
                                        //    energy is attenuated by the Fresnel reflectance only
                                        //    if the flag is set to true. Initially, all types of
                                        //    energy are reflected and the energy is set to 1.0.
                                        //    Then, the energy is attenuated by the Fresnel
                                        //    reflectance at each node of the trajectory, and if the
                                        //    energy is less than or equal to 0.0, the energy will
                                        //    be set as been absorbed.
                                        let mut energy_per_wavelength =
                                            vec![Energy::Reflected(1.0); n_spectrum];
                                        for node in trajectory.iter().take(trajectory.len() - 1) {
                                            if node.cos.is_none() {
                                                continue;
                                            }
                                            let cos_i_abs = node.cos.unwrap().abs();
                                            energy_per_wavelength.iter_mut().enumerate().for_each(
                                                |(i, energy)| match energy {
                                                    Energy::Reflected(e) => {
                                                        if fresnel {
                                                            *e *= fresnel::reflectance(
                                                                cos_i_abs,
                                                                &self.iors_i[i],
                                                                &self.iors_t[i],
                                                            ) * cos_i_abs;
                                                        } else {
                                                            *e *= cos_i_abs;
                                                        }
                                                        if *e <= 0.0 {
                                                            *energy = Energy::Absorbed;
                                                        }
                                                    },
                                                    _ => unreachable!(
                                                        "energy initially is not reflected"
                                                    ),
                                                },
                                            );
                                        }

                                        // 3. Update the number of received rays.
                                        n_received.fetch_add(1, atomic::Ordering::Relaxed);

                                        // 4. Update the maximum number of bounces.
                                        let bounce = (trajectory.len() - 1) as u32;
                                        n_bounce.fetch_max(bounce, atomic::Ordering::Relaxed);

                                        // 5. Compute the index of the patch where the ray is
                                        //    collected.
                                        let patch_idx = self
                                            .patches
                                            .contains(Sph2::from_cartesian(ray_dir.into()));

                                        // 6. Update the number of rays reflected by the surface.
                                        for (i, energy) in energy_per_wavelength.iter().enumerate()
                                        {
                                            match energy {
                                                Energy::Absorbed => continue,
                                                Energy::Reflected(_) => {
                                                    n_reflected[i]
                                                        .fetch_add(1, atomic::Ordering::Relaxed);
                                                    if patch_idx.is_none() {
                                                        n_escaped[i].fetch_add(
                                                            1,
                                                            atomic::Ordering::Relaxed,
                                                        );
                                                    }
                                                },
                                            }
                                        }

                                        // Returns the index of the ray, the unit vector pointing to
                                        // the collector's
                                        // surface, and the number of bounces.
                                        patch_idx.and_then(|patch_idx| {
                                            Some(OutgoingRay {
                                                ray_idx: ray_idx as u32,
                                                ray_dir,
                                                bounce,
                                                energy_per_wavelength,
                                                patch_idx,
                                            })
                                        })
                                    },
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Box<_>>();
                let n_bounce = n_bounce.load(atomic::Ordering::Relaxed) as usize;
                let n_received = n_received.load(atomic::Ordering::Relaxed);

                // Create the statistics of the BSDF measurement for the current incident point.
                let mut stats = SingleBsdfMeasurementStats::new(n_spectrum, n_bounce);
                stats.n_received = n_received;

                // Update the number of rays statistics including the number of rays absorbed
                // per wavelength, the number of rays reflected per wavelength.
                // The number of rays captured per bounce will be updated later.
                for i in 0..n_spectrum {
                    let reflected = n_reflected[i].load(atomic::Ordering::Relaxed);
                    let absorbed = n_received - reflected;
                    let escaped = n_escaped[i].load(atomic::Ordering::Relaxed);
                    stats.n_absorbed_mut()[i] = absorbed;
                    stats.n_reflected_mut()[i] = reflected;
                    stats.n_escaped_mut()[i] = escaped;
                }

                (n_bounce, stats, dirs)
            };

            #[cfg(feature = "bench")]
            let dirs_proc_time = start.elapsed().as_millis();
            #[cfg(feature = "bench")]
            log::info!("Collector::collect: dirs: {} ms", dirs_proc_time);

            // Compute the vertex positions of the outgoing rays.
            let mut hit_points = vec![Vec3::ZERO; dirs.len()];

            for (i, dir) in dirs.iter().enumerate() {
                hit_points[i] = (dir.ray_dir * orbit_radius).into();
                let samples_offset = dir.patch_idx * n_spectrum;
                let patch_samples = &mut records[samples_offset..samples_offset + n_spectrum];
                let bounce_idx = dir.bounce as usize - 1;
                dir.energy_per_wavelength
                    .iter()
                    .zip(patch_samples.iter_mut())
                    .enumerate()
                    .for_each(|(j, (energy, patch))| match energy {
                        Energy::Absorbed => {},
                        Energy::Reflected(e) => {
                            let mut_patch = {
                                if patch.is_none() {
                                    *patch = Some(BounceAndEnergy::empty(n_bounce));
                                }
                                patch.as_mut().unwrap()
                            };
                            let energy = *e as f64;
                            mut_patch.energy_per_bounce[0] += energy;
                            mut_patch.n_ray_per_bounce[0] += 1;
                            mut_patch.energy_per_bounce[bounce_idx + 1] += energy;
                            mut_patch.n_ray_per_bounce[bounce_idx + 1] += 1;
                            stats.n_ray_per_bounce[j * n_bounce + bounce_idx] += 1;
                            stats.n_captured_mut()[j] += 1;
                            stats.energy_per_bounce[j * n_bounce + bounce_idx] += energy;
                            stats.e_captured[j] += energy;
                        },
                    });
            }

            debug_assert!(stats.is_valid(), "stats is invalid");
            #[cfg(debug_assertions)]
            log::debug!("{:?}", stats);

            #[cfg(feature = "bench")]
            let data_time = start.elapsed().as_millis();
            #[cfg(feature = "bench")]
            log::info!(
                "Collector::collect: data: {} ms",
                data_time - dirs_proc_time
            );

            match out_stats {
                None => {
                    *out_stats = Some(stats);
                },
                Some(existing) => {
                    existing.merge(stats);
                },
            }

            out_trajs.extend_from_slice(&result.trajectories);
            out_hpnts.extend_from_slice(&hit_points);
        }

        #[cfg(not(feature = "vdbg"))]
        {
            use std::sync::RwLock;

            debug_assert_eq!(
                result.bounces.len(),
                result.dirs.len(),
                "bounces and dirs mismatch"
            );
            debug_assert_eq!(
                result.bounces.len(),
                result.energy.len() / n_spectrum,
                "bounces and energy mismatch"
            );

            log::debug!(
                "SingleSimulationResult at {}, {} rays",
                result.wi,
                result.dirs.len()
            );

            let n_bounce = result.bounces.iter().max().copied().unwrap();
            assert!(n_bounce > 0, "no bounces");
            // TODO: deal with the grazing angle 90 degrees, especially while storing and
            // reading. General suggestion: avoid 90 degrees
            if n_bounce == 0 {
                log::warn!("no bounces at {}", result.wi);
                *out_stats = Some(SingleBsdfMeasurementStats::empty());
                return;
            }
            log::debug!("collecting -- n_bounce: {}", n_bounce);

            let n_received = AtomicU64::new(0);
            let n_missed = AtomicU64::new(0);
            let mut n_reflected = Vec::with_capacity(n_spectrum);
            let mut n_escaped = Vec::with_capacity(n_spectrum);
            for _ in 0..n_spectrum {
                n_reflected.push(AtomicU64::new(0));
                n_escaped.push(AtomicU64::new(0));
            }
            use rayon::iter::{ParallelBridge, ParallelIterator};
            // Create the statistics of the BSDF measurement for the current incident point.
            let stats_rw_lock = RwLock::new(SingleBsdfMeasurementStats::new(
                n_spectrum,
                n_bounce as usize,
            ));
            let records_rw_lock = RwLock::new(records);
            result
                .iter_ray_chunks(CHUNK_SIZE)
                .par_bridge()
                .for_each(|chunk| {
                    let mut n_missed_local = 0;
                    let mut n_received_local = 0;
                    let mut n_reflected_local = vec![0; n_spectrum].into_boxed_slice();
                    let mut n_escaped_local = vec![0; n_spectrum].into_boxed_slice();
                    for ray in chunk {
                        if ray.bounce == &0 {
                            n_missed_local += 1;
                            continue;
                        }

                        n_received_local += 1;

                        // Compute the index of the patch where the ray is
                        // collected.
                        let patch_idx = self.patches.contains(Sph2::from_cartesian(*ray.dir));

                        // Update the number of rays reflected by the surface.
                        for (i, energy) in ray.energy.iter().enumerate() {
                            if energy <= &0.0 {
                                continue;
                            }

                            n_reflected_local[i] += 1;
                            if patch_idx.is_none() {
                                n_escaped_local[i] += 1;
                            }
                        }

                        // Update the energy of the ray per wavelength.
                        if patch_idx.is_none() {
                            continue;
                        }

                        let patch_idx = patch_idx.unwrap();
                        let samples_offset = patch_idx * n_spectrum;
                        let mut records = records_rw_lock.write().unwrap();

                        let patch_samples =
                            &mut records[samples_offset..samples_offset + n_spectrum];
                        let bounce_idx = *ray.bounce as usize - 1;
                        let mut mut_stats = stats_rw_lock.write().unwrap();
                        ray.energy
                            .iter()
                            .enumerate()
                            .zip(patch_samples.iter_mut())
                            .for_each(|((k, e), patch)| {
                                let energy = *e as f64;
                                if energy <= 0.0 {
                                    return;
                                }
                                let mut_patch = {
                                    if patch.is_none() {
                                        *patch = Some(BounceAndEnergy::empty(n_bounce as usize));
                                    }
                                    patch.as_mut().unwrap()
                                };
                                // Reallocate the memory in case the number of bounces is greater
                                mut_patch.reallocate(n_bounce as usize);
                                mut_patch.energy_per_bounce[0] += energy;
                                mut_patch.n_ray_per_bounce[0] += 1;
                                mut_patch.energy_per_bounce[bounce_idx + 1] += energy;
                                mut_patch.n_ray_per_bounce[bounce_idx + 1] += 1;
                                mut_stats.n_ray_per_bounce[k * n_bounce as usize + bounce_idx] += 1;
                                mut_stats.n_captured_mut()[k] += 1;
                                mut_stats.energy_per_bounce[k * n_bounce as usize + bounce_idx] +=
                                    energy;
                                mut_stats.e_captured[k] += energy;
                            })
                    }

                    n_received.fetch_add(n_received_local, atomic::Ordering::Relaxed);
                    n_missed.fetch_add(n_missed_local, atomic::Ordering::Relaxed);
                    for i in 0..n_spectrum {
                        n_reflected[i].fetch_add(n_reflected_local[i], atomic::Ordering::Relaxed);
                        n_escaped[i].fetch_add(n_escaped_local[i], atomic::Ordering::Relaxed);
                    }
                });

            let mut stats = stats_rw_lock.into_inner().unwrap();

            stats.n_received = n_received.load(atomic::Ordering::Relaxed);
            // Update the number of rays statistics including the number of rays absorbed
            // per wavelength, the number of rays reflected per wavelength.
            // The number of rays captured per bounce will be updated later.
            for i in 0..n_spectrum {
                let reflected = n_reflected[i].load(atomic::Ordering::Relaxed);
                let absorbed = stats.n_received - reflected;
                let escaped = n_escaped[i].load(atomic::Ordering::Relaxed);
                stats.n_absorbed_mut()[i] = absorbed;
                stats.n_reflected_mut()[i] = reflected;
                stats.n_escaped_mut()[i] = escaped;
            }

            #[cfg(debug_assertions)]
            log::debug!("{:?}", stats);

            match out_stats {
                None => {
                    *out_stats = Some(stats);
                },
                Some(existing) => {
                    existing.merge(stats);
                },
            }
        }
    }
}

#[cfg(feature = "vdbg")]
/// Energy after a ray is reflected by the micro-surface.
///
/// Used during the data collection process.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
enum Energy {
    /// The ray of a certain wavelength is absorbed by the micro-surface.
    Absorbed,
    /// The ray of a specific wavelength is reflected by the micro-surface.
    Reflected(f32),
}

/// Bounce and energy of a patch for each bounce.
///
/// The index of the array corresponds to the bounce number starting from 1.
/// At index 0, the data is the sum of all the bounces.
#[derive(Debug, Clone, Default)]
pub struct BounceAndEnergy {
    /// Maximum number of bounces of rays hitting the patch.
    pub n_bounce: u32,
    /// Number of rays hitting the patch for each bounce.
    pub n_ray_per_bounce: Box<[u64]>,
    /// Total energy of rays hitting the patch for each bounce.
    pub energy_per_bounce: Box<[f64]>,
}

impl BounceAndEnergy {
    /// Creates a new bounce and energy.
    pub fn empty(bounces: usize) -> Self {
        Self {
            n_bounce: bounces as u32,
            n_ray_per_bounce: vec![0; bounces + 1].into_boxed_slice(),
            energy_per_bounce: vec![0.0; bounces + 1].into_boxed_slice(),
        }
    }

    /// Reallocates the memory in case the number of bounces is greater
    /// than the current number of bounces. The function creates a new
    /// array of the number of rays and energy per bounce and then
    /// copies the data from the old arrays to the new arrays.
    pub fn reallocate(&mut self, bounces: usize) {
        if bounces as u32 <= self.n_bounce {
            return;
        }
        let mut n_ray_per_bounce = vec![0; bounces + 1];
        let mut energy_per_bounce = vec![0.0; bounces + 1];
        n_ray_per_bounce[..=self.n_bounce as usize].copy_from_slice(&self.n_ray_per_bounce);
        energy_per_bounce[..=self.n_bounce as usize].copy_from_slice(&self.energy_per_bounce);
        self.n_ray_per_bounce = n_ray_per_bounce.into_boxed_slice();
        self.energy_per_bounce = energy_per_bounce.into_boxed_slice();
        self.n_bounce = bounces as u32;
    }

    /// Returns the total number of rays.
    pub fn total_rays(&self) -> u64 { self.n_ray_per_bounce[0] }

    /// Returns the total energy of rays.
    pub fn total_energy(&self) -> f64 { self.energy_per_bounce[0] }
}

impl PartialEq for BounceAndEnergy {
    fn eq(&self, other: &Self) -> bool {
        self.n_bounce == other.n_bounce
            && self.n_ray_per_bounce == other.n_ray_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}

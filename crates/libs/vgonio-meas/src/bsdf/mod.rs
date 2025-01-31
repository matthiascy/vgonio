use crate::{
    bsdf::trace::BounceAndEnergy, params::BsdfMeasurementParams, DataCarriedOnHemisphereSampler,
};
use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    path::Path,
};
use vgonio_bxdf::brdf::measured::{
    ClausenBrdf, ClausenBrdfParametrisation, VgonioBrdf, VgonioBrdfParameterisation,
};
use vgonio_core::{
    bxdf::{MeasuredBrdfKind, Origin},
    error::VgonioError,
    math::{rcp_f64, Sph2, Vec3},
    units::{Degs, Nanometres, Radians, Rads},
    utils::{medium::Medium, partition::SphericalPartition},
    AnyMeasured, AnyMeasuredBrdf, BrdfLevel, MeasurementKind,
};
use vgonio_jabr::array::DyArr;

pub mod emitter;
pub mod params;
pub mod trace;

/// Type of the BSDF to be measured.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BsdfKind {
    /// Bidirectional reflectance distribution function.
    Brdf = 0x00,

    /// Bidirectional transmittance distribution function.
    Btdf = 0x01,

    /// Bidirectional scattering-surface distribution function.
    Bssdf = 0x02,

    /// Bidirectional scattering-surface reflectance distribution function.
    Bssrdf = 0x03,

    /// Bidirectional scattering-surface transmittance distribution function.
    Bsstdf = 0x04,
}

impl Display for BsdfKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BsdfKind::Brdf => {
                write!(f, "brdf")
            },
            BsdfKind::Btdf => {
                write!(f, "btdf")
            },
            BsdfKind::Bssdf => {
                write!(f, "bssdf")
            },
            BsdfKind::Bssrdf => {
                write!(f, "bssrdf")
            },
            BsdfKind::Bsstdf => {
                write!(f, "bsstdf")
            },
        }
    }
}

impl From<u8> for BsdfKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => BsdfKind::Brdf,
            0x01 => BsdfKind::Btdf,
            0x02 => BsdfKind::Bssdf,
            0x03 => BsdfKind::Bssrdf,
            0x04 => BsdfKind::Bsstdf,
            _ => panic!("Invalid BSDF kind: {}", value),
        }
    }
}

// TODO: save multiple Receiver data into one file
/// BSDF measurement data.
///
/// The Number of emitted rays, wavelengths, and bounces are invariant over
/// emitter's position.
///
/// At each emitter's position, each emitted ray carries an initial energy
/// equals to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct BsdfMeasurement {
    /// Parameters of the measurement.
    pub params: BsdfMeasurementParams,
    /// Raw data of the measurement.
    pub raw: RawBsdfMeasurement,
    /// Collected BSDF data.
    pub bsdfs: HashMap<BrdfLevel, VgonioBrdf>,
}

impl AnyMeasured for BsdfMeasurement {
    fn kind(&self) -> MeasurementKind { MeasurementKind::Bsdf }

    fn has_multiple_levels(&self) -> bool { true }

    fn as_any(&self) -> &dyn std::any::Any { self }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn as_any_brdf(&self, level: BrdfLevel) -> Option<&dyn AnyMeasuredBrdf> {
        assert!(level.is_valid(), "Invalid BRDF level!");
        self.bsdfs
            .get(&level)
            .map(|bsdf| bsdf as &dyn AnyMeasuredBrdf)
    }
}

impl BsdfMeasurement {
    /// Returns the number of wavelengths.
    #[inline]
    pub fn n_spectrum(&self) -> usize { self.raw.n_spectrum() }

    /// Returns the brdf at the given level.
    pub fn brdf_at(&self, level: BrdfLevel) -> Option<&VgonioBrdf> { self.bsdfs.get(&level) }

    /// Writes the BSDF data to images in exr format.
    ///
    /// Only BSDF data is written to the images. The full measurement data is
    /// not written.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The path to the directory where the images will be saved.
    /// * `timestamp` - The timestamp of the measurement.
    /// * `resolution` - The resolution of the images.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &DateTime<Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        for (level, bsdf) in &self.bsdfs {
            let filename = format!(
                "{}_{}.exr",
                filepath.file_stem().unwrap().to_str().unwrap(),
                level
            );
            bsdf.write_as_exr(&filepath.with_file_name(filename), timestamp, resolution)?;
        }
        Ok(())
    }

    /// Resamples the BSDF data to match the `ClausenBrdfParametrisation`.
    pub fn resample(
        &self,
        params: &ClausenBrdfParametrisation,
        level: BrdfLevel,
        dense: bool,
        phi_offset: Radians,
    ) -> ClausenBrdf {
        let spectrum = self.params.emitter.spectrum.values().collect::<Vec<_>>();
        let n_spectrum = spectrum.len();
        let n_wi = params.incoming.len();
        let n_wo = params.n_wo;
        let n_wo_dense = if dense { n_wo * 4 } else { n_wo };

        // row-major [wi, wo, spectrum]
        let mut samples = vec![0.0; n_wi * n_wo_dense * n_spectrum];
        let wo_step = Degs::new(2.5).to_radians();

        let new_params = if dense {
            let mut outgoing = DyArr::<Sph2, 2>::zeros([n_wi, n_wo_dense]);
            for (i, wi) in params.incoming.as_slice().iter().enumerate() {
                let mut new_j = 0;
                for j in 0..n_wo {
                    let wo = params.outgoing[[i, j]];
                    if wo.approx_eq(&Sph2::zero()) {
                        outgoing[[i, new_j]] = Sph2::zero();
                        new_j += 1;
                    } else if (wo.theta - wi.theta).abs().as_f32() < 1e-6 {
                        // The incident and outgoing directions are the same.
                        if wo.phi.as_f32().abs() < 1e-6 {
                            // The azimuth of the incident direction is almost zero.
                            for k in 0..4 {
                                outgoing[[i, new_j]] = Sph2::new(
                                    wo.theta - wo_step * (3 - k) as f32,
                                    wo.phi + phi_offset,
                                );
                                new_j += 1;
                            }
                            for k in 0..3 {
                                outgoing[[i, new_j]] = Sph2::new(
                                    wo.theta - wo_step * (3 - k) as f32,
                                    Rads::PI + phi_offset,
                                );
                                new_j += 1;
                            }
                        } else {
                            for k in 0..3 {
                                outgoing[[i, new_j]] = Sph2::new(
                                    wo.theta - wo_step * (3 - k) as f32,
                                    Rads::ZERO + phi_offset,
                                );
                                new_j += 1;
                            }
                            for k in 0..4 {
                                outgoing[[i, new_j]] = Sph2::new(
                                    wo.theta - wo_step * (3 - k) as f32,
                                    wo.phi + phi_offset,
                                );
                            }
                        }
                    } else {
                        for k in 0..4 {
                            outgoing[[i, new_j]] =
                                Sph2::new(wo.theta - wo_step * (3 - k) as f32, wo.phi + phi_offset);
                            new_j += 1;
                        }
                    }
                }
            }
            ClausenBrdfParametrisation {
                incoming: params.incoming.clone(),
                outgoing,
                n_wo: n_wo_dense,
                i_thetas: params.i_thetas.clone(),
                o_thetas: params.o_thetas.clone(),
                phis: params.phis.clone(),
            }
        } else {
            params.clone()
        };

        let brdf = self.bsdfs.get(&level).unwrap();
        let sampler = DataCarriedOnHemisphereSampler::new(brdf).unwrap();

        // Get the interpolated samples for wi, wo, and spectrum.
        new_params.wi_wos_iter().for_each(|(i, (wi, wos))| {
            let samples_offset_wi = i * n_wo_dense * n_spectrum;
            for (j, wo) in wos.iter().enumerate() {
                let samples_offset = samples_offset_wi + j * n_spectrum;
                sampler.sample_point_at(
                    *wi,
                    *wo,
                    &mut samples[samples_offset..samples_offset + n_spectrum],
                );
            }
        });

        ClausenBrdf {
            kind: MeasuredBrdfKind::Clausen,
            origin: Origin::RealWorld,
            incident_medium: self.params.incident_medium,
            transmitted_medium: self.params.transmitted_medium,
            params: Box::new(new_params),
            spectrum: DyArr::from_vec_1d(spectrum),
            samples: DyArr::<f32, 3>::from_vec([n_wi, n_wo_dense, n_spectrum], samples),
        }
    }
}

/// Raw data of the BSDF measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct RawBsdfMeasurement {
    /// The number of incident directions along the zenith.
    pub n_zenith_in: usize,
    /// The spectrum of the emitter.
    pub spectrum: DyArr<Nanometres>,
    /// The incident directions of the emitter.
    pub incoming: DyArr<Sph2>,
    /// The outgoing directions of the receiver.
    pub outgoing: SphericalPartition,
    /// Collected data of the receiver per incident direction per
    /// outgoing (patch) direction and per wavelength: `ωi, ωo, λ`.
    pub records: DyArr<Option<BounceAndEnergy>, 3>,
    /// The statistics of the measurement per incident direction.
    pub stats: DyArr<SingleBsdfMeasurementStats>,
    #[cfg(feature = "vdbg")]
    /// Extra ray trajectory data per incident direction.
    pub trajectories: Box<[Box<[RayTrajectory]>]>,
    #[cfg(feature = "vdbg")]
    /// Hit points on the receiver per incident direction.
    pub hit_points: Box<[Vec<Vec3>]>,
}

impl RawBsdfMeasurement {
    /// Returns the iterator over the snapshots of the raw measured BSDF data.
    pub fn snapshots(&self) -> RawBsdfSnapshotIterator {
        RawBsdfSnapshotIterator { data: self, idx: 0 }
    }

    /// Returns the number of incident directions.
    #[inline]
    pub fn n_wi(&self) -> usize { self.incoming.len() }

    /// Returns the number of outgoing directions.
    #[inline]
    pub fn n_wo(&self) -> usize { self.outgoing.n_patches() }

    /// Returns the number of wavelengths.
    #[inline]
    pub fn n_spectrum(&self) -> usize { self.spectrum.len() }

    /// Returns maximum number of bounces.
    pub fn max_bounces(&self) -> usize {
        self.stats
            .iter()
            .map(|stats| stats.n_bounce as usize)
            .max()
            .unwrap()
    }

    /// Computes the BSDF data from the raw data.
    pub fn compute_bsdfs(
        &self,
        medium_i: Medium,
        medium_t: Medium,
    ) -> HashMap<BrdfLevel, VgonioBrdf> {
        let n_wi = self.n_wi();
        let n_wo = self.n_wo();
        let n_spectrum = self.n_spectrum();
        let max_bounces = self.max_bounces();
        log::debug!(
            "Computing BSDF... with {} patches, {} bounces, and {} snapshots",
            n_wo,
            max_bounces,
            self.snapshots().len()
        );
        let mut bsdfs = HashMap::new();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        let mut samples_l1_plus = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        for b in 0..max_bounces {
            #[cfg(all(debug_assertions, feature = "vvdbg"))]
            log::debug!("Computing BSDF at bounce #{}", b);
            let mut any_energy = false;
            for (wi_idx, snapshot) in self.snapshots().enumerate() {
                if snapshot.stats.n_bounce <= b as u32 {
                    continue;
                }
                let snapshot_samples = &mut samples.as_mut_slice()
                    [wi_idx * n_wo * n_spectrum..(wi_idx + 1) * n_wo * n_spectrum];
                let snapshot_l1_plus = &mut samples_l1_plus.as_mut_slice()
                    [wi_idx * n_wo * n_spectrum..(wi_idx + 1) * n_wo * n_spectrum];
                let cos_i = snapshot.wi.theta.cos() as f64;
                let rcp_e_i = rcp_f64(snapshot.stats.n_received as f64 * cos_i);
                for (wo_idx, patch_data_per_wavelength) in
                    snapshot.records.chunks(n_spectrum).enumerate()
                {
                    #[cfg(all(debug_assertions, feature = "vvdbg"))]
                    log::debug!("- snapshot #{wi_idx} and patch #{wo_idx}",);
                    for (k, patch_energy) in patch_data_per_wavelength.iter().enumerate() {
                        if patch_energy.is_none() {
                            continue;
                        }
                        let patch = &self.outgoing.patches[wo_idx];
                        let cos_o = patch.center().theta.cos() as f64;
                        let solid_angle = patch.solid_angle().as_f64();
                        let e_o = patch_energy
                            .as_ref()
                            .unwrap()
                            .energy_per_bounce
                            .get(b)
                            .unwrap_or(&0.0);
                        if e_o == &0.0 {
                            continue;
                        }
                        if cos_o != 0.0 {
                            let l_o = e_o * rcp_f64(cos_o) * rcp_f64(solid_angle);
                            let val = l_o * rcp_e_i;
                            snapshot_samples[wo_idx * n_spectrum + k] = val as f32;
                            if b > 1 {
                                snapshot_l1_plus[wo_idx * n_spectrum + k] += val as f32;
                            }
                            any_energy = true;

                            #[cfg(all(debug_assertions, feature = "vvdbg"))]
                            log::debug!(
                                "    - energy of patch {wo_idx}: {:>12.4}, λ[{k}] --  e_i: \
                                 {:>12.4}, L_o[{k}]: {:>12.4} -- brdf: {:>14.8}",
                                e_o,
                                rcp_f64(rcp_e_i),
                                l_o,
                                snapshot_samples[wo_idx * n_spectrum + k],
                            );
                        }
                    }
                }
            }
            if any_energy {
                bsdfs.insert(
                    BrdfLevel::from(b),
                    VgonioBrdf::new(
                        Origin::Simulated,
                        medium_i,
                        medium_t,
                        VgonioBrdfParameterisation {
                            n_zenith_i: self.n_zenith_in,
                            incoming: self.incoming.clone(),
                            outgoing: self.outgoing.clone(),
                        },
                        self.spectrum.clone(),
                        samples.clone(),
                    ),
                );
            }
        }
        bsdfs.insert(
            BrdfLevel::L1Plus,
            VgonioBrdf::new(
                Origin::Simulated,
                medium_i,
                medium_t,
                VgonioBrdfParameterisation {
                    n_zenith_i: self.n_zenith_in,
                    incoming: self.incoming.clone(),
                    outgoing: self.outgoing.clone(),
                },
                self.spectrum.clone(),
                samples_l1_plus,
            ),
        );
        bsdfs
    }
}

/// A snapshot of the raw measured BSDF data at a given incident direction of
/// the emitter.
pub struct RawBsdfSnapshotIterator<'a> {
    data: &'a RawBsdfMeasurement,
    idx: usize,
}

/// A snapshot of the raw measured BSDF data at a given incident direction of
/// the emitter.
pub struct RawBsdfSnapshot<'a> {
    /// The incident direction of the snapshot.
    pub wi: Sph2,
    /// The collected data of the receiver per outgoing direction and per
    /// wavelength. The data is stored as `ωo, λ`.
    pub records: &'a [Option<BounceAndEnergy>],
    /// The statistics of the snapshot.
    pub stats: &'a SingleBsdfMeasurementStats,
    /// Extra ray trajectory data.
    #[cfg(feature = "vdbg")]
    pub trajectories: &'a [RayTrajectory],
    /// Hit points on the receiver.
    #[cfg(feature = "vdbg")]
    pub hit_points: &'a [Vec3],
}

impl<'a> Iterator for RawBsdfSnapshotIterator<'a> {
    type Item = RawBsdfSnapshot<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.incoming.len();
        if self.idx < self.data.incoming.len() {
            let n_wo = self.data.outgoing.n_patches();
            let n_spectrum = self.data.spectrum.len();
            let idx = self.idx * n_wo * n_spectrum;
            let snapshot = RawBsdfSnapshot {
                wi: self.data.incoming[self.idx],
                records: &self.data.records.as_slice()[idx..idx + n_wo * n_spectrum],
                stats: &self.data.stats[self.idx],
                #[cfg(feature = "vdbg")]
                trajectories: &self.data.trajectories[self.idx],
                #[cfg(feature = "vdbg")]
                hit_points: &self.data.hit_points[self.idx],
            };
            self.idx += 1;
            Some(snapshot)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for RawBsdfSnapshotIterator<'_> {
    fn len(&self) -> usize { self.data.incoming.len() - self.idx }
}

/// BSDF measurement statistics for a single emitter's position.
///
/// N_emitted = N_missed + N_received
/// N_received = N_absorbed + N_reflected
/// N_reflected = N_captured + N_escaped
#[derive(Clone)]
pub struct SingleBsdfMeasurementStats {
    /// Number of bounces (actual bounce, not always equals to the maximum
    /// bounce limit).
    pub n_bounce: u32,
    /// Number of emitted rays that hit the surface; invariant over wavelength.
    pub n_received: u64,
    /// Number of emitted rays that missed the surface; invariant over
    /// wavelength.
    pub n_missed: u64,
    /// Number of wavelengths, which is the number of samples in the spectrum.
    pub n_spectrum: usize,
    /// Statistics of the number of emitted rays that either
    /// 1. hit the surface and were absorbed, or
    /// 2. hit the surface and were reflected, or
    /// 3. hit the surface and were captured by the collector or
    /// 4. hit the surface and escaped.
    ///
    /// The statistics are variant over wavelength. The data is stored as a
    /// flat array in row-major order with the dimensions (statistics,
    /// wavelength). The index of the statistics is defined by the constants
    /// `ABSORBED_IDX`, `REFLECTED_IDX`, and `CAPTURED_IDX`, `ESCAPED_IDX`.
    /// The total number of elements in the array is `4 * n_wavelength`.
    pub n_ray_stats: Box<[u64]>,
    /// Energy captured by the collector; variant over wavelength.
    pub e_captured: Box<[f64]>,
    /// Histogram of reflected rays by number of bounces, variant over
    /// wavelength. The data is stored as a flat array in row-major order with
    /// the dimensions (wavelegnth, bounce). The total number of elements in
    /// the array is `n_wavelength * n_bounces`.
    pub n_ray_per_bounce: Box<[u64]>,
    /// Histogram of energy of reflected rays by number of bounces. The energy
    /// is the sum of the energy of the rays that were reflected at the
    /// corresponding bounce. The data is stored as a flat array in row-major
    /// order with the dimensions (wavelegnth, bounce). The total number of
    /// elements in the array is `n_wavelength * n_bounces`.
    pub energy_per_bounce: Box<[f64]>,
}

static_assertions::assert_eq_size!(
    SingleBsdfMeasurementStats,
    Option<SingleBsdfMeasurementStats>
);

impl PartialEq for SingleBsdfMeasurementStats {
    fn eq(&self, other: &Self) -> bool {
        self.n_bounce == other.n_bounce
            && self.n_spectrum == other.n_spectrum
            && self.n_missed == other.n_missed
            && self.n_received == other.n_received
            && self.n_ray_stats == other.n_ray_stats
            && self.e_captured == other.e_captured
            && self.n_ray_per_bounce == other.n_ray_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}

impl SingleBsdfMeasurementStats {
    /// Index of the number of absorbed rays statistics in the `n_ray_stats`
    /// array.
    pub const ABSORBED_IDX: usize = 0;
    /// Index of the number of reflected rays statistics in the `n_ray_stats`
    /// array.
    pub const REFLECTED_IDX: usize = 1;
    /// Index of the number of captured rays statistics in the `n_ray_stats`
    /// array.
    pub const CAPTURED_IDX: usize = 2;
    /// Index of the number of escaped rays statistics in the `n_ray_stats`
    /// array.
    pub const ESCAPED_IDX: usize = 3;
    /// Number of statistics in the `n_ray_stats` array.
    pub const N_STATS: usize = 4;

    /// Creates an empty `BsdfMeasurementStatsPoint`.
    ///
    /// # Arguments
    /// * `n_wavelength`: Number of wavelengths.
    /// * `max_bounce`: Maximum number of bounces. This is used to pre-allocate
    ///   the memory for the histograms.
    pub fn new(n_spectrum: usize, max_bounce: usize) -> Self {
        Self {
            n_bounce: max_bounce as u32,
            n_received: 0,
            n_missed: 0,
            n_spectrum,
            n_ray_stats: vec![0; Self::N_STATS * n_spectrum].into_boxed_slice(),
            e_captured: vec![0.0; n_spectrum].into_boxed_slice(),
            n_ray_per_bounce: vec![0; n_spectrum * max_bounce].into_boxed_slice(),
            energy_per_bounce: vec![0.0; n_spectrum * max_bounce].into_boxed_slice(),
        }
    }

    /// Creates an empty `SingleBsdfMeasurementStats`.
    pub fn empty() -> Self {
        Self {
            n_bounce: 0,
            n_received: 0,
            n_missed: 0,
            n_spectrum: 0,
            n_ray_stats: Box::new([]),
            e_captured: Box::new([]),
            n_ray_per_bounce: Box::new([]),
            energy_per_bounce: Box::new([]),
        }
    }

    /// Merges the statistics of another `SingleBsdfMeasurementStats` into this
    /// one.
    pub fn merge(&mut self, mut other: SingleBsdfMeasurementStats) {
        debug_assert!(self.n_spectrum == other.n_spectrum, "Mismatched n_spectrum");

        #[cfg(feature = "vvdbg")]
        log::debug!("Merging stats:\n{:#?}\n+\n{:#?}", self, &other);

        self.n_received += other.n_received;
        self.n_missed += other.n_missed;

        self.n_ray_stats
            .iter_mut()
            .zip(other.n_ray_stats.iter())
            .for_each(|(a, b)| *a += *b);

        self.e_captured
            .iter_mut()
            .zip(other.e_captured.iter())
            .for_each(|(a, b)| *a += *b);

        // Replace the histograms of current stats with the other one.
        if self.n_bounce < other.n_bounce {
            #[cfg(feature = "vvdbg")]
            log::debug!("Swapping stats");
            std::mem::swap(&mut self.n_bounce, &mut other.n_bounce);
            std::mem::swap(&mut self.n_ray_per_bounce, &mut other.n_ray_per_bounce);
            std::mem::swap(&mut self.energy_per_bounce, &mut other.energy_per_bounce);
        }

        #[cfg(feature = "vvdbg")]
        log::debug!("After swap:\n{:#?}\n+\n{:#?}", self, &other);

        // Add the histograms of the other stats to this one.
        for i in 0..other.n_spectrum {
            for j in 0..other.n_bounce as usize {
                self.n_ray_per_bounce[i * self.n_bounce as usize + j] +=
                    other.n_ray_per_bounce[i * other.n_bounce as usize + j];
                self.energy_per_bounce[i * self.n_bounce as usize + j] +=
                    other.energy_per_bounce[i * other.n_bounce as usize + j];
            }
        }

        #[cfg(feature = "vvdbg")]
        log::debug!("After merge:\n{:#?}", self);
    }

    /// Returns the number of absorbed rays statistics per wavelength.
    pub fn n_absorbed(&self) -> &[u64] {
        let offset = Self::ABSORBED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of absorbed rays statistics per wavelength.
    pub fn n_absorbed_mut(&mut self) -> &mut [u64] {
        let offset = Self::ABSORBED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of reflected rays statistics per wavelength.
    pub fn n_reflected(&self) -> &[u64] {
        let offset = Self::REFLECTED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of reflected rays statistics per wavelength.
    pub fn n_reflected_mut(&mut self) -> &mut [u64] {
        let offset = Self::REFLECTED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of captured rays statistics per wavelength.
    pub fn n_captured(&self) -> &[u64] {
        let offset = Self::CAPTURED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of captured rays statistics per wavelength.
    pub fn n_captured_mut(&mut self) -> &mut [u64] {
        let offset = Self::CAPTURED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of escaped rays statistics per wavelength.
    pub fn n_escaped(&self) -> &[u64] {
        let offset = Self::ESCAPED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of escaped rays statistics per wavelength.
    pub fn n_escaped_mut(&mut self) -> &mut [u64] {
        let offset = Self::ESCAPED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the energy per wavelength which is the sum of the energy of the
    /// per bounce for the given wavelength.
    pub fn energy_per_wavelength(&self) -> Box<[f64]> {
        let mut energy = vec![0.0; self.n_spectrum].into_boxed_slice();
        for i in 0..self.n_spectrum {
            for j in 0..self.n_bounce as usize {
                energy[i] += self.energy_per_bounce[i * self.n_bounce as usize + j];
            }
        }
        energy
    }

    /// Tests if the statistics are valid.
    pub fn is_valid(&self) -> bool {
        if self.n_ray_stats.len() != Self::N_STATS * self.n_spectrum {
            eprintln!("Invalid n_ray_stats length: {}", self.n_ray_stats.len());
            return false;
        }
        if self.n_ray_per_bounce.len() != self.n_spectrum * self.n_bounce as usize {
            eprintln!(
                "Invalid n_ray_per_bounce length: {}",
                self.n_ray_per_bounce.len()
            );
            return false;
        }
        if self.energy_per_bounce.len() != self.n_spectrum * self.n_bounce as usize {
            eprintln!(
                "Invalid energy_per_bounce length: {}",
                self.energy_per_bounce.len()
            );
            return false;
        }
        if self.e_captured.len() != self.n_spectrum {
            eprintln!("Invalid e_captured length: {}", self.e_captured.len());
            return false;
        }
        // N_emitted = N_missed + N_received
        for i in 0..self.n_spectrum {
            // N_received = N_absorbed + N_reflected
            if self.n_reflected()[i] + self.n_absorbed()[i] != self.n_received {
                eprintln!(
                    "Invalid N_received: {} = Nr {} + Na {}",
                    self.n_received,
                    self.n_reflected()[i],
                    self.n_absorbed()[i]
                );
                return false;
            }
            // N_reflected = N_captured + N_escaped
            if self.n_captured()[i] + self.n_escaped()[i] != self.n_reflected()[i] {
                eprintln!(
                    "Invalid N_reflected: {} = {} + {}",
                    self.n_reflected()[i],
                    self.n_captured()[i],
                    self.n_escaped()[i]
                );
                return false;
            }
        }
        true
    }
}

impl Default for SingleBsdfMeasurementStats {
    fn default() -> Self { Self::new(0, 0) }
}

impl Debug for SingleBsdfMeasurementStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"BsdfMeasurementPointStats:
    - n_bounces:   {},
    - n_received:  {},
    - n_missed:    {},
    - n_absorbed:  {:?},
    - n_reflected: {:?},
    - n_escaped:   {:?},
    - n_captured:  {:?},
    - total_energy_captured: {:?},
    - num_rays_per_bounce:   {:?}"#,
            self.n_bounce,
            self.n_received,
            self.n_missed,
            self.n_absorbed(),
            self.n_reflected(),
            self.n_escaped(),
            self.n_captured(),
            self.e_captured,
            self.n_ray_per_bounce,
        )?;
        for i in 0..self.n_spectrum {
            let offset = i * self.n_bounce as usize;
            write!(
                f,
                "\n    - energy per bounce λ[{}]: {:?}",
                i,
                &self.energy_per_bounce[offset..offset + self.n_bounce as usize],
            )?;
        }
        Ok(())
    }
}

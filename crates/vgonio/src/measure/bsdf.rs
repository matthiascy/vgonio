//! Measurement of the BSDF (Bidirectional Scattering Distribution Function) of
//! micro-surfaces.

use super::params::BsdfMeasurementParams;
#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::embr;
use crate::{
    app::{
        cache::{Handle, RawCache},
        cli::ansi,
    },
    measure::{
        bsdf::{
            emitter::Emitter,
            receiver::{BounceAndEnergy, Receiver},
            rtc::{RayTrajectory, RtcMethod},
        },
        params::SimulationKind,
        MeasuredData, Measurement, MeasurementSource,
    },
};
#[cfg(any(feature = "visu-dbg", debug_assertions))]
use base::math::Vec3;
use base::{
    error::VgonioError,
    impl_measured_data_trait,
    math::{circular_angle_dist, projected_barycentric_coords, rcp_f32, Sph2},
    medium::Medium,
    partition::{PartitionScheme, SphericalPartition},
    units::{Degs, Nanometres, Radians, Rads},
    MeasurementKind,
};
use bxdf::brdf::measured::{
    ClausenBrdf, ClausenBrdfParameterisation, Origin, VgonioBrdf, VgonioBrdfParameterisation,
};
use chrono::{DateTime, Local};
use jabr::array::DyArr;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{write, Debug, Display, Formatter},
    mem::MaybeUninit,
    path::Path,
    str::FromStr,
};
use surf::{MicroSurface, MicroSurfaceMesh};

pub mod emitter;
pub(crate) mod params;
pub mod receiver;
pub mod rtc;

// TODO: data retrieval and processing

/// Raw data of the BSDF measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct RawMeasuredBsdfData {
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
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    /// Extra ray trajectory data per incident direction.
    pub trajectories: Box<[Vec<RayTrajectory>]>,
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    /// Hit points on the receiver per incident direction.
    pub hit_points: Box<[Vec<Vec3>]>,
}

/// A snapshot of the raw measured BSDF data at a given incident direction of
/// the emitter.
pub struct RawBsdfSnapshotIterator<'a> {
    data: &'a RawMeasuredBsdfData,
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
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    pub trajectories: &'a [RayTrajectory],
    /// Hit points on the receiver.
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
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
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                trajectories: &self.data.trajectories[self.idx],
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
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

impl RawMeasuredBsdfData {
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
    ) -> HashMap<MeasuredBrdfLevel, VgonioBrdf> {
        let n_wi = self.n_wi();
        let n_wo = self.n_wo();
        let n_spectrum = self.n_spectrum();
        log::info!("Computing BSDF... with {} patches", n_wo);
        let max_bounces = self.max_bounces();
        let mut bsdfs = HashMap::new();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        let mut samples_l1_plus = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        for b in 0..max_bounces {
            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
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
                let cos_i = snapshot.wi.theta.cos();
                let rcp_e_i = rcp_f32(snapshot.stats.n_received as f32 * cos_i);
                for (wo_idx, patch_data_per_wavelength) in
                    snapshot.records.chunks(n_spectrum).enumerate()
                {
                    #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                    log::debug!("- snapshot #{wi_idx} and patch #{wo_idx}",);
                    for (k, patch_energy) in patch_data_per_wavelength.iter().enumerate() {
                        if patch_energy.is_none() {
                            continue;
                        }
                        let patch = &self.outgoing.patches[wo_idx];
                        let cos_o = patch.center().theta.cos();
                        let solid_angle = patch.solid_angle().as_f32();
                        let e_o = patch_energy.as_ref().unwrap().energy_per_bounce[b];
                        if cos_o != 0.0 {
                            let l_o = e_o * rcp_f32(cos_o) * rcp_f32(solid_angle);
                            let val = l_o * rcp_e_i;
                            snapshot_samples[wo_idx * n_spectrum + k] = val;
                            if b > 1 {
                                snapshot_l1_plus[wo_idx * n_spectrum + k] += val;
                            }
                            any_energy = true;

                            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                            log::debug!(
                                "    - energy of patch {wo_idx}: {:>12.4}, λ[{k}] --  e_i: \
                                 {:>12.4}, L_o[{k}]: {:>12.4} -- brdf: {:>14.8}",
                                e_o,
                                rcp_f32(rcp_e_i),
                                l_o,
                                snapshot_samples[wo_idx * n_spectrum + k],
                            );
                        }
                    }
                }
            }
            if any_energy {
                bsdfs.insert(
                    MeasuredBrdfLevel::from(b),
                    VgonioBrdf::new(
                        Origin::Simulated,
                        medium_i,
                        medium_t,
                        VgonioBrdfParameterisation {
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
            MeasuredBrdfLevel::L1PLUS,
            VgonioBrdf::new(
                Origin::Simulated,
                medium_i,
                medium_t,
                VgonioBrdfParameterisation {
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

/// The level of the measured BRDF.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MeasuredBrdfLevel(u32);

impl MeasuredBrdfLevel {
    /// The level of the measured BRDF that includes the energy of rays at
    /// bounces greater than 1.
    pub const L1PLUS: MeasuredBrdfLevel = MeasuredBrdfLevel(u32::MAX);

    /// The level of the measured BRDF that includes the energy of rays at
    /// all bounces.
    pub const L0: MeasuredBrdfLevel = MeasuredBrdfLevel(0);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the first bounce.
    pub const L1: MeasuredBrdfLevel = MeasuredBrdfLevel(1);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the second bounce.
    pub const L2: MeasuredBrdfLevel = MeasuredBrdfLevel(2);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the third bounce.
    pub const L3: MeasuredBrdfLevel = MeasuredBrdfLevel(3);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the fourth bounce.
    pub const L4: MeasuredBrdfLevel = MeasuredBrdfLevel(4);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the fifth bounce.
    pub const L5: MeasuredBrdfLevel = MeasuredBrdfLevel(5);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the sixth bounce.
    pub const L6: MeasuredBrdfLevel = MeasuredBrdfLevel(6);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the seventh bounce.
    pub const L7: MeasuredBrdfLevel = MeasuredBrdfLevel(7);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the eighth bounce.
    pub const L8: MeasuredBrdfLevel = MeasuredBrdfLevel(8);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the ninth bounce.
    pub const L9: MeasuredBrdfLevel = MeasuredBrdfLevel(9);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the tenth bounce.
    pub const L10: MeasuredBrdfLevel = MeasuredBrdfLevel(10);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the eleventh bounce.
    pub const L11: MeasuredBrdfLevel = MeasuredBrdfLevel(11);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the twelfth bounce.
    pub const L12: MeasuredBrdfLevel = MeasuredBrdfLevel(12);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the thirteenth bounce.
    pub const L13: MeasuredBrdfLevel = MeasuredBrdfLevel(13);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the fourteenth bounce.
    pub const L14: MeasuredBrdfLevel = MeasuredBrdfLevel(14);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the fifteenth bounce.
    pub const L15: MeasuredBrdfLevel = MeasuredBrdfLevel(15);

    /// The level of the measured BRDF that includes the energy of rays at
    /// the sixteenth bounce.
    pub const L16: MeasuredBrdfLevel = MeasuredBrdfLevel(16);

    /// Returns the u32 representation of the level.
    pub const fn as_u32(&self) -> u32 { self.0 }
}

impl From<usize> for MeasuredBrdfLevel {
    fn from(n: usize) -> Self { MeasuredBrdfLevel(n as u32) }
}

impl From<u32> for MeasuredBrdfLevel {
    fn from(n: u32) -> Self { MeasuredBrdfLevel(n) }
}

impl FromStr for MeasuredBrdfLevel {
    type Err = VgonioError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let val = s.to_lowercase();
        let val = val.trim();
        if val.len() > 3 {
            return Err(VgonioError::new(
                "Invalid BRDF level. The level must be between l0 and l16 or l1+.",
                None,
            ));
        }
        match val {
            "l1+" => Ok(MeasuredBrdfLevel::L1PLUS),
            _ => {
                let level = s.strip_prefix('l').ok_or_else(|| {
                    VgonioError::new(
                        "Invalid BRDF level. The level must be between l0 and l16",
                        None,
                    )
                })?;
                Ok(MeasuredBrdfLevel::from(level.parse::<u32>().map_err(
                    |err| {
                        VgonioError::new(
                            format!("Invalid BRDF level. {}", err),
                            Some(Box::new(err)),
                        )
                    },
                )?))
            }
        }
    }
}

impl Display for MeasuredBrdfLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            u32::MAX => f.write_str("l1+"),
            _ => write!(f, "l{}", self.0),
        }
    }
}

/// BSDF measurement data.
///
/// The Number of emitted rays, wavelengths, and bounces are invariant over
/// emitter's position.
///
/// At each emitter's position, each emitted ray carries an initial energy
/// equals to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredBsdfData {
    /// Parameters of the measurement.
    pub params: BsdfMeasurementParams,
    /// Raw data of the measurement.
    pub raw: RawMeasuredBsdfData,
    /// Collected BSDF data.
    pub bsdfs: HashMap<MeasuredBrdfLevel, VgonioBrdf>,
}

impl_measured_data_trait!(MeasuredBsdfData, Bsdf);

impl MeasuredBsdfData {
    /// Returns the number of wavelengths.
    #[inline]
    pub fn n_spectrum(&self) -> usize { self.raw.n_spectrum() }

    /// Returns the brdf at the given level.
    pub fn brdf_at(&self, level: MeasuredBrdfLevel) -> Option<&VgonioBrdf> {
        self.bsdfs.get(&level)
    }

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

    /// Resamples the BSDF data to match the `ClausenBrdfParameterisation`.
    pub fn resample(
        &self,
        params: &ClausenBrdfParameterisation,
        level: MeasuredBrdfLevel,
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
            ClausenBrdfParameterisation {
                incoming: params.incoming.clone(),
                outgoing,
                n_wo: n_wo_dense,
            }
        } else {
            params.clone()
        };

        // Get the interpolated samples for wi, wo, and spectrum.
        new_params.wi_wos_iter().for_each(|(i, (wi, wos))| {
            let samples_offset_wi = i * n_wo_dense * n_spectrum;
            for (j, wo) in wos.iter().enumerate() {
                let samples_offset = samples_offset_wi + j * n_spectrum;
                self.sample_at(
                    level.0,
                    *wi,
                    *wo,
                    &mut samples[samples_offset..samples_offset + n_spectrum],
                );
            }
        });

        ClausenBrdf {
            origin: Origin::RealWorld,
            incident_medium: self.params.incident_medium,
            transmitted_medium: self.params.transmitted_medium,
            params: Box::new(new_params),
            spectrum: DyArr::from_vec_1d(spectrum),
            samples: DyArr::<f32, 3>::from_vec([n_wi, n_wo_dense, n_spectrum], samples),
        }
    }

    // TODO: merge this method to [`MeasuredDataSampler`]
    /// Retrieve the BSDF sample data (full wavelength) at the given position.
    ///
    /// The position is given in the unit spherical coordinates. The returned
    /// data is the BSDF values for each snapshot and each wavelength at the
    /// given position.
    ///
    /// # Arguments
    ///
    /// * `wi` - The incident direction.
    /// * `wo` - The outgoing direction.
    /// * `interpolated` - The interpolated BSDF values at the given position.
    ///
    /// # Panics
    ///
    /// Panic if the number of wavelengths in the interpolated data does not
    /// match the number of wavelengths in the BSDF data.
    pub fn sample_at(&self, level: u32, wi: Sph2, wo: Sph2, interpolated: &mut [f32]) {
        log::trace!(
            "Sampling at wi: ({}, {}), wo: ({} {})",
            wi.theta.to_degrees().prettified(),
            wi.phi.to_degrees().prettified(),
            wo.theta.to_degrees().prettified(),
            wo.phi.to_degrees().prettified()
        );
        assert_eq!(
            interpolated.len(),
            self.params.emitter.spectrum.step_count(),
            "Mismatch in the number of wavelengths."
        );
        let snapshot_idx = self
            .raw
            .incoming
            .as_slice()
            .iter()
            .position(|snap| wi.approx_eq(snap))
            .expect(
                "The incident direction is not found in the BSDF snapshots. The incident \
                 direction must be one of the directions of the emitter.",
            );
        log::trace!("  - Found snapshot at wi: {}", wi);
        let n_spectrum = self.raw.n_spectrum();
        let n_wo = self.raw.n_wo();
        let brdf = self.bsdfs.get(&MeasuredBrdfLevel(level)).unwrap();
        let snapshot_samples = &brdf.samples.as_slice()
            [snapshot_idx * n_wo * n_spectrum..(snapshot_idx + 1) * n_wo * n_spectrum];
        match self.params.receiver.scheme {
            PartitionScheme::Beckers => {
                let partition = SphericalPartition::new(
                    self.params.receiver.scheme,
                    self.params.receiver.domain,
                    self.params.receiver.precision,
                );
                // 1. Find the upper and lower ring where the position is located.
                // The Upper ring is the ring with the smallest zenith angle.
                let (upper_ring_idx, lower_ring_idx) = partition.find_rings(wo);
                log::trace!(
                    "  - Upper ring: {}, Lower ring: {}",
                    upper_ring_idx,
                    lower_ring_idx
                );

                // 2. Find the patch where the position is located inside the ring.
                if lower_ring_idx == 0 || lower_ring_idx == 1 {
                    // Interpolate inside a triangle.
                    let lower_ring = partition.rings[1];
                    let patch_idx = {
                        let patch_idx = lower_ring.find_patch_indices(wo.phi);
                        (
                            0,
                            lower_ring.base_index + patch_idx.0,
                            lower_ring.base_index + patch_idx.1,
                        )
                    };
                    let center = (
                        partition.patches[patch_idx.0].center(),
                        partition.patches[patch_idx.1].center(),
                        partition.patches[patch_idx.2].center(),
                    );
                    let (u, v, w) = projected_barycentric_coords(
                        wo.to_cartesian(),
                        center.0.to_cartesian(),
                        center.1.to_cartesian(),
                        center.2.to_cartesian(),
                    );
                    let patch0_samples =
                        &snapshot_samples[patch_idx.0 * n_spectrum..(patch_idx.0 + 1) * n_spectrum];
                    let patch1_samples =
                        &snapshot_samples[patch_idx.1 * n_spectrum..(patch_idx.1 + 1) * n_spectrum];
                    let patch2_samples =
                        &snapshot_samples[patch_idx.2 * n_spectrum..(patch_idx.2 + 1) * n_spectrum];
                    log::trace!(
                        "  - Interpolating inside a triangle between patches #{} ({}, {} | {:?}), \
                         #{} ({}, {} | {:?}), and #{} ({}, {} | {:?})",
                        patch_idx.0,
                        center.0,
                        center.0.to_cartesian(),
                        patch0_samples,
                        patch_idx.1,
                        center.1,
                        center.1.to_cartesian(),
                        patch1_samples,
                        patch_idx.2,
                        center.2,
                        center.2.to_cartesian(),
                        patch2_samples
                    );
                    log::trace!("  - Barycentric coordinates: ({}, {}, {})", u, v, w);
                    interpolated.iter_mut().enumerate().for_each(|(i, spl)| {
                        *spl =
                            u * patch0_samples[i] + v * patch1_samples[i] + w * patch2_samples[i];
                    });
                } else if upper_ring_idx == lower_ring_idx
                    && upper_ring_idx == partition.n_rings() - 1
                {
                    // This should be the last ring.
                    // Interpolate between two patches.
                    let ring = partition.rings[upper_ring_idx];
                    let patch_idx = ring.find_patch_indices(wo.phi);
                    let patch0_idx = ring.base_index + patch_idx.0;
                    let patch1_idx = ring.base_index + patch_idx.1;
                    let patch0 = partition.patches[patch0_idx];
                    let patch1 = partition.patches[patch1_idx];
                    let center = (patch0.center(), patch1.center());
                    let patch0_samples =
                        &snapshot_samples[patch0_idx * n_spectrum..(patch0_idx + 1) * n_spectrum];
                    let patch1_samples =
                        &snapshot_samples[patch1_idx * n_spectrum..(patch1_idx + 1) * n_spectrum];
                    log::trace!(
                        "  - Interpolating between two patches: #{} ({}, {} | {:?}) and #{} ({}, \
                         {} | {:?}) at ring #{}",
                        patch0_idx,
                        center.0,
                        center.0.to_cartesian(),
                        patch0_samples,
                        patch1_idx,
                        center.1,
                        center.1.to_cartesian(),
                        patch1_samples,
                        upper_ring_idx
                    );
                    let t = (circular_angle_dist(wo.phi, center.0.phi)
                        / circular_angle_dist(center.1.phi, center.0.phi))
                    .clamp(0.0, 1.0);
                    interpolated.iter_mut().enumerate().for_each(|(i, spl)| {
                        *spl = (1.0 - t) * patch0_samples[i] + t * patch1_samples[i];
                    });
                } else {
                    // Interpolate inside a quadrilateral.
                    let (upper_t, upper_patch_center, upper_patch_idx) = {
                        let upper_ring = partition.rings[upper_ring_idx];
                        let upper_patch_idx = {
                            let patches = upper_ring.find_patch_indices(wo.phi);
                            (
                                upper_ring.base_index + patches.0,
                                upper_ring.base_index + patches.1,
                            )
                        };
                        let upper_patch_center = (
                            partition.patches[upper_patch_idx.0].center(),
                            partition.patches[upper_patch_idx.1].center(),
                        );
                        log::trace!(
                            "        - upper_#{} center: {}",
                            upper_patch_idx.0,
                            upper_patch_center.0
                        );
                        log::trace!(
                            "        - upper_#{} center: {}",
                            upper_patch_idx.1,
                            upper_patch_center.1
                        );
                        let upper_t = (circular_angle_dist(wo.phi, upper_patch_center.0.phi)
                            / circular_angle_dist(
                                upper_patch_center.1.phi,
                                upper_patch_center.0.phi,
                            ))
                        .clamp(0.0, 1.0);
                        log::trace!("          - upper_t: {}", upper_t);
                        (upper_t, upper_patch_center, upper_patch_idx)
                    };

                    let (lower_t, lower_patch_center, lower_patch_idx) = {
                        let lower_ring = partition.rings[lower_ring_idx];
                        let lower_patch_idx = {
                            let patches = lower_ring.find_patch_indices(wo.phi);
                            (
                                lower_ring.base_index + patches.0,
                                lower_ring.base_index + patches.1,
                            )
                        };
                        let lower_patch_center = (
                            partition.patches[lower_patch_idx.0].center(),
                            partition.patches[lower_patch_idx.1].center(),
                        );
                        log::trace!(
                            "        - lower_#{} center: {}",
                            lower_patch_idx.0,
                            lower_patch_center.0,
                        );
                        log::trace!(
                            "        - lower_#{} center: {}",
                            lower_patch_idx.1,
                            lower_patch_center.1
                        );
                        let lower_t = (circular_angle_dist(wo.phi, lower_patch_center.0.phi)
                            / circular_angle_dist(
                                lower_patch_center.1.phi,
                                lower_patch_center.0.phi,
                            ))
                        .clamp(0.0, 1.0);
                        log::trace!("          - lower_t: {}", lower_t);
                        (lower_t, lower_patch_center, lower_patch_idx)
                    };
                    let s = (circular_angle_dist(wo.theta, upper_patch_center.0.theta)
                        / circular_angle_dist(
                            lower_patch_center.0.theta,
                            upper_patch_center.0.theta,
                        ))
                    .clamp(0.0, 1.0);
                    // Bilateral interpolation.
                    let upper_patch0_samples = &snapshot_samples
                        [upper_patch_idx.0 * n_spectrum..(upper_patch_idx.0 + 1) * n_spectrum];
                    let upper_patch1_samples = &snapshot_samples
                        [upper_patch_idx.1 * n_spectrum..(upper_patch_idx.1 + 1) * n_spectrum];
                    let lower_patch0_samples = &snapshot_samples
                        [lower_patch_idx.0 * n_spectrum..(lower_patch_idx.0 + 1) * n_spectrum];
                    let lower_patch1_samples = &snapshot_samples
                        [lower_patch_idx.1 * n_spectrum..(lower_patch_idx.1 + 1) * n_spectrum];
                    log::trace!(
                        "  - Interpolating inside a quadrilateral between rings #{} (#{} vals \
                         {:?}, #{} vals {:?} | t = {}), and #{} (#{} vals {:?}, #{} vals {:?} | t \
                         = {}), v = {}",
                        upper_ring_idx,
                        upper_patch_idx.0,
                        upper_patch0_samples,
                        upper_patch_idx.1,
                        upper_patch1_samples,
                        upper_t,
                        lower_ring_idx,
                        lower_patch_idx.0,
                        lower_patch0_samples,
                        lower_patch_idx.1,
                        lower_patch1_samples,
                        lower_t,
                        s
                    );
                    interpolated.iter_mut().enumerate().for_each(|(i, spl)| {
                        let upper_interp = (1.0 - upper_t) * upper_patch0_samples[i]
                            + upper_t * upper_patch1_samples[i];
                        let lower_interp = (1.0 - lower_t) * lower_patch0_samples[i]
                            + lower_t * lower_patch1_samples[i];
                        *spl = (1.0 - s) * upper_interp + s * lower_interp;
                    });
                }
                log::trace!("  - Sampled: {:?}", &interpolated);
            }
            PartitionScheme::EqualAngle => {
                unimplemented!()
            }
        }
    }
}

// pub(crate) fn compute_bsdf_snapshots_max_values(
//     snapshots: &[BsdfSnapshot],
//     n_wavelength: usize,
// ) -> Box<[f32]> {
//     let n_wi = snapshots.len();
//     let mut max_values = vec![0.0; n_wi * n_wavelength].into_boxed_slice();
//     snapshots.iter().enumerate().for_each(|(i, snapshot)| {
//         let offset = i * n_wavelength;
//         snapshot
//             .samples
//             .as_slice()
//             .chunks(n_wavelength)
//             .for_each(|patch_samples| {
//                 patch_samples.iter().enumerate().for_each(|(j, val)| {
//                     max_values[offset + j] = f32::max(max_values[offset + j],
// *val);                 });
//             });
//     });
//     max_values
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::bsdf::{
        emitter::EmitterParams, receiver::ReceiverParams, rtc::RtcMethod::Grid,
    };
    use base::{
        medium::Medium, partition::SphericalDomain, range::RangeByStepSizeInclusive, units::nm,
    };

    #[test]
    fn test_measured_bsdf_sample() {
        let precision = Sph2 {
            theta: Rads::from_degrees(30.0),
            phi: Rads::from_degrees(2.0),
        };
        let partition =
            SphericalPartition::new(PartitionScheme::Beckers, SphericalDomain::Upper, precision);
        let n_wo = partition.n_patches();
        let spectrum_range = RangeByStepSizeInclusive::new(nm!(100.0), nm!(400.0), nm!(100.0));
        let spectrum = DyArr::from_iterator([-1], spectrum_range.values());
        let n_spectrum = spectrum.len();
        let mut bsdfs = HashMap::new();
        bsdfs.insert(
            MeasuredBrdfLevel(0),
            VgonioBrdf {
                origin: Origin::RealWorld,
                incident_medium: Medium::Vacuum,
                transmitted_medium: Medium::Aluminium,
                params: Box::new(VgonioBrdfParameterisation {
                    incoming: DyArr::zeros([1]),
                    outgoing: partition.clone(),
                }),
                spectrum: spectrum.clone(),
                samples: DyArr::ones([1, n_wo, 4]),
            },
        );
        let measured = MeasuredBsdfData {
            params: BsdfMeasurementParams {
                emitter: EmitterParams {
                    num_rays: 0,
                    max_bounces: 0,
                    zenith: RangeByStepSizeInclusive::new(Rads::ZERO, Rads::ZERO, Rads::ZERO),
                    azimuth: RangeByStepSizeInclusive::new(Rads::ZERO, Rads::ZERO, Rads::ZERO),
                    spectrum: spectrum_range,
                },
                receiver: ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision,
                    scheme: PartitionScheme::Beckers,
                },
                kind: BsdfKind::Brdf,
                sim_kind: SimulationKind::GeomOptics(Grid),
                incident_medium: Medium::Vacuum,
                fresnel: Default::default(),
                transmitted_medium: Medium::Vacuum,
            },
            raw: RawMeasuredBsdfData {
                spectrum,
                incoming: DyArr::zeros([1]),
                outgoing: partition,
                records: DyArr::splat(Some(BounceAndEnergy::empty(2)), [1, n_wo, n_spectrum]),
                stats: DyArr::splat(
                    SingleBsdfMeasurementStats {
                        n_bounce: 0,
                        n_received: 0,
                        n_missed: 0,
                        n_spectrum,
                        n_ray_stats: Box::new([0, 0, 0]),
                        e_captured: Box::new([0.0]),
                        n_ray_per_bounce: Box::new([]),
                        energy_per_bounce: Box::new([]),
                    },
                    [1],
                ),
                trajectories: Box::new([]),
                hit_points: Box::new([]),
            },
            bsdfs,
        };

        let mut interpolated = vec![0.0, 0.0, 0.0, 0.0];
        let wi = Sph2 {
            theta: Rads::ZERO,
            phi: Rads::ZERO,
        };
        measured.sample_at(
            0,
            wi,
            Sph2 {
                theta: Rads::ZERO,
                phi: Rads::ZERO,
            },
            &mut interpolated,
        );
        assert_eq!(interpolated, vec![1.0, 1.0, 1.0, 1.0]);

        interpolated.iter_mut().for_each(|spl| *spl = 0.0);
        measured.sample_at(
            0,
            wi,
            Sph2 {
                theta: Rads::from_degrees(80.0),
                phi: Rads::from_degrees(30.0),
            },
            &mut interpolated,
        );
        assert_eq!(interpolated, vec![1.0, 1.0, 1.0, 1.0]);

        measured.sample_at(
            0,
            wi,
            Sph2 {
                theta: Rads::from_degrees(40.0),
                phi: Rads::from_degrees(30.0),
            },
            &mut interpolated,
        );
        assert_eq!(interpolated, vec![1.0, 1.0, 1.0, 1.0]);

        measured.sample_at(
            0,
            wi,
            Sph2 {
                theta: Rads::from_degrees(55.0),
                phi: Rads::from_degrees(30.0),
            },
            &mut interpolated,
        );
        assert_eq!(interpolated, vec![1.0, 1.0, 1.0, 1.0]);
    }
}

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
            }
            BsdfKind::Btdf => {
                write!(f, "btdf")
            }
            BsdfKind::Bssdf => {
                write!(f, "bssdf")
            }
            BsdfKind::Bssrdf => {
                write!(f, "bssrdf")
            }
            BsdfKind::Bsstdf => {
                write!(f, "bsstdf")
            }
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
    pub n_received: u32,
    /// Number of emitted rays that missed the surface; invariant over
    /// wavelength.
    pub n_missed: u32,
    /// Number of wavelengths, which is the number of samples in the spectrum.
    pub n_spectrum: usize,
    /// Statistics of the number of emitted rays that either
    /// 1. hit the surface and were absorbed, or
    /// 2. hit the surface and were reflected, or
    /// 3. hit the surface and were captured by the collector or
    /// 4. hit the surface and escaped.
    /// The statistics are variant over wavelength. The data is stored as a
    /// flat array in row-major order with the dimensions (statistics,
    /// wavelength). The index of the statistics is defined by the constants
    /// `ABSORBED_IDX`, `REFLECTED_IDX`, and `CAPTURED_IDX`, `ESCAPED_IDX`.
    /// The total number of elements in the array is `4 * n_wavelength`.
    pub n_ray_stats: Box<[u32]>,
    // TODO: verify if this is the correct way to store the energy captured by
    // the collector. Check if the energy is the sum of the energy of the rays
    // that were captured in each patch. After that, this could be removed.
    /// Energy captured by the collector; variant over wavelength.
    pub e_captured: Box<[f32]>,
    /// Histogram of reflected rays by number of bounces, variant over
    /// wavelength. The data is stored as a flat array in row-major order with
    /// the dimensions (wavelegnth, bounce). The total number of elements in
    /// the array is `n_wavelength * n_bounces`.
    pub n_ray_per_bounce: Box<[u32]>,
    /// Histogram of energy of reflected rays by number of bounces. The energy
    /// is the sum of the energy of the rays that were reflected at the
    /// corresponding bounce. The data is stored as a flat array in row-major
    /// order with the dimensions (wavelegnth, bounce). The total number of
    /// elements in the array is `n_wavelength * n_bounces`.
    pub energy_per_bounce: Box<[f32]>,
}

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

    /// Returns the number of absorbed rays statistics per wavelength.
    pub fn n_absorbed(&self) -> &[u32] {
        let offset = Self::ABSORBED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of absorbed rays statistics per wavelength.
    pub fn n_absorbed_mut(&mut self) -> &mut [u32] {
        let offset = Self::ABSORBED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of reflected rays statistics per wavelength.
    pub fn n_reflected(&self) -> &[u32] {
        let offset = Self::REFLECTED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of reflected rays statistics per wavelength.
    pub fn n_reflected_mut(&mut self) -> &mut [u32] {
        let offset = Self::REFLECTED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of captured rays statistics per wavelength.
    pub fn n_captured(&self) -> &[u32] {
        let offset = Self::CAPTURED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of captured rays statistics per wavelength.
    pub fn n_captured_mut(&mut self) -> &mut [u32] {
        let offset = Self::CAPTURED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of escaped rays statistics per wavelength.
    pub fn n_escaped(&self) -> &[u32] {
        let offset = Self::ESCAPED_IDX * self.n_spectrum;
        &self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the number of escaped rays statistics per wavelength.
    pub fn n_escaped_mut(&mut self) -> &mut [u32] {
        let offset = Self::ESCAPED_IDX * self.n_spectrum;
        &mut self.n_ray_stats[offset..offset + self.n_spectrum]
    }

    /// Returns the energy per wavelength which is the sum of the energy of the
    /// per bounce for the given wavelength.
    pub fn energy_per_wavelength(&self) -> Box<[f32]> {
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

/// Ray tracing simulation result for a single incident direction of a surface.
pub struct SigleSimulationResult {
    /// Incident direction in the unit spherical coordinates.
    pub wi: Sph2,
    /// Trajectories of the rays.
    pub trajectories: Vec<RayTrajectory>,
}

/// Measures the BSDF of a surface using geometric ray tracing methods.
pub fn measure_bsdf_rt(
    params: BsdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    sim_kind: SimulationKind,
    cache: &RawCache,
) -> Box<[Measurement]> {
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = cache.get_micro_surfaces(handles);
    let emitter = Emitter::new(&params.emitter);
    let receiver = Receiver::new(&params.receiver, &params, cache);
    let n_wi = emitter.measpts.len();
    let n_wo = receiver.patches.n_patches();

    log::debug!(
        "Measuring BSDF of {} surfaces from {} measurement points with {} rays",
        surfaces.len(),
        emitter.measpts.len(),
        emitter.params.num_rays,
    );

    let mut measurements = Vec::with_capacity(surfaces.len());
    for (surf, mesh) in surfaces.iter().zip(meshes) {
        if surf.is_none() || mesh.is_none() {
            log::debug!("Skipping surface {:?} and its mesh {:?}", surf, mesh);
            continue;
        }

        let surf = surf.unwrap();
        let mesh = mesh.unwrap();

        log::info!(
            "Measuring surface {}",
            surf.path.as_ref().unwrap().display()
        );

        let sim_results = match sim_kind {
            SimulationKind::GeomOptics(method) => {
                println!(
                    "    {} Measuring {} with geometric optics...",
                    ansi::YELLOW_GT,
                    params.kind
                );
                match method {
                    #[cfg(feature = "embree")]
                    RtcMethod::Embree => embr::simulate_bsdf_measurement(&emitter, mesh),
                    #[cfg(feature = "optix")]
                    RtcMethod::Optix => rtc_simulation_optix(&params, mesh, &emitter, cache),
                    RtcMethod::Grid => rtc_simulation_grid(&params, surf, mesh, &emitter, cache),
                }
            }
            SimulationKind::WaveOptics => {
                println!(
                    "    {} Measuring {} with wave optics...",
                    ansi::YELLOW_GT,
                    params.kind
                );
                todo!("Wave optics simulation is not yet implemented")
            }
        };

        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        log::trace!("Estimated orbit radius: {}", orbit_radius);

        let incoming = DyArr::<Sph2>::from_slice([n_wi], &emitter.measpts);
        let n_spectrum = params.emitter.spectrum.step_count();

        #[cfg(any(feature = "visu-dbg", debug_assertions))]
        let mut trajectories: Box<[MaybeUninit<Vec<RayTrajectory>>]> = Box::new_uninit_slice(n_wi);
        #[cfg(any(feature = "visu-dbg", debug_assertions))]
        let mut hit_points: Box<[MaybeUninit<Vec<Vec3>>]> = Box::new_uninit_slice(n_wi);

        let mut records = DyArr::splat(Option::<BounceAndEnergy>::None, [n_wi, n_wo, n_spectrum]);
        let mut stats: Box<[MaybeUninit<SingleBsdfMeasurementStats>]> = Box::new_uninit_slice(n_wi);

        for (i, sim) in sim_results.enumerate() {
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            let trjs = trajectories[i].as_mut_ptr();
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            let hpts = hit_points[i].as_mut_ptr();
            let recs =
                &mut records.as_mut_slice()[i * n_wo * n_spectrum..(i + 1) * n_wo * n_spectrum];

            #[cfg(feature = "bench")]
            let t = std::time::Instant::now();

            println!(
                "        {} Collecting BSDF snapshot {}{}/{}{}...",
                ansi::YELLOW_GT,
                ansi::BRIGHT_CYAN,
                i + 1,
                n_wi,
                ansi::RESET
            );

            // Collect the tracing data into raw bsdf snapshots.
            receiver.collect(
                sim,
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                trjs,
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                hpts,
                stats[i].as_mut_ptr(),
                recs,
                orbit_radius,
                params.fresnel,
            );

            #[cfg(feature = "bench")]
            {
                let elapsed = t.elapsed();
                log::debug!(
                    "bsdf measurement data collection (one snapshot) took {} secs.",
                    elapsed.as_secs_f64()
                );
            }
        }

        let raw = RawMeasuredBsdfData {
            spectrum: DyArr::from_iterator([-1], params.emitter.spectrum.values()),
            incoming,
            outgoing: receiver.patches.clone(),
            records,
            stats: DyArr::from_boxed_slice([n_wi], unsafe { stats.assume_init() }),
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            trajectories: unsafe { trajectories.assume_init() },
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            hit_points: unsafe { hit_points.assume_init() },
        };

        let bsdfs = raw.compute_bsdfs(params.incident_medium, params.transmitted_medium);

        measurements.push(Measurement {
            name: surf.file_stem().unwrap().to_owned(),
            source: MeasurementSource::Measured(Handle::with_id(surf.uuid)),
            timestamp: chrono::Local::now(),
            measured: Box::new(MeasuredBsdfData { params, raw, bsdfs }),
        });
    }

    measurements.into_boxed_slice()
}

/// Brdf measurement of a microfacet surface using the grid ray tracing.
fn rtc_simulation_grid(
    _params: &BsdfMeasurementParams,
    _surf: &MicroSurface,
    _mesh: &MicroSurfaceMesh,
    _emitter: &Emitter,
    _cache: &RawCache,
) -> Box<dyn Iterator<Item = SigleSimulationResult>> {
    // for (surf, mesh) in surfaces.iter().zip(meshes.iter()) {
    //     if surf.is_none() || mesh.is_none() {
    //         log::debug!("Skipping surface {:?} and its mesh {:?}", surf,
    // mesh);         continue;
    //     }
    //     let surf = surf.unwrap();
    //     let _mesh = mesh.unwrap();
    //     println!(
    //         "      {BRIGHT_YELLOW}>{RESET} Measure surface {}",
    //         surf.path.as_ref().unwrap().display()
    //     );
    //     // let t = std::time::Instant::now();
    //     // crate::measure::bsdf::rtc::grid::measure_bsdf(
    //     //     &params, surf, mesh, &emitter, cache,
    //     // );
    //     // println!(
    //     //     "        {BRIGHT_CYAN}✓{RESET} Done in {:?} s",
    //     //     t.elapsed().as_secs_f32()
    //     // );
    // }
    todo!("Grid ray tracing is not yet implemented");
}

/// Brdf measurement of a microfacet surface using the OptiX ray tracing.
#[cfg(feature = "optix")]
fn rtc_simulation_optix(
    _params: &BsdfMeasurementParams,
    _surf: &MicroSurfaceMesh,
    _emitter: &Emitter,
    _cache: &RawCache,
) -> Box<dyn Iterator<Item = SigleSimulationResult>> {
    todo!()
}

// pub fn measure_in_plane_brdf_grid(
//     desc: &MeasurementDesc,
//     ior_db: &RefractiveIndexDatabase,
//     surfaces: &[Heightfield],
// ) { let collector: Collector = desc.collector.into(); let emitter: Emitter =
//   desc.emitter.into(); log::debug!("Emitter generated {} patches.",
//   emitter.patches.len());
//
//     let mut embree_rt = EmbreeRayTracing::new(Config::default());
//
//     for surface in surfaces {
//         let scene_id = embree_rt.create_scene();
//         let triangulated = surface.triangulate(TriangulationMethod::Regular);
//         let radius = triangulated.extent.max_edge() * 2.5;
//         let surface_mesh = embree_rt.create_triangle_mesh(&triangulated);
//         let surface_id = embree_rt.attach_geometry(scene_id, surface_mesh);
//         let spectrum_samples =
// SpectrumSampler::from(desc.emitter.spectrum).samples();         let grid_rt =
// GridRayTracing::new(surface, &triangulated);         log::debug!(
//             "Grid - min: {}, max: {} | origin: {:?}",
//             grid_rt.min,
//             grid_rt.max,
//             grid_rt.origin
//         );
//         // let ior_i = ior_db
//         //     .ior_of_spectrum(desc.incident_medium, &spectrum_samples)
//         //     .unwrap();
//         // let ior_t = ior_db
//         //     .ior_of_spectrum(desc.transmitted_medium, &spectrum_samples)
//         //     .unwrap();
//
//         for wavelength in spectrum_samples {
//             println!("Capturing with wavelength = {}", wavelength);
//             let ior_t = ior_db
//                 .refractive_index_of(desc.transmitted_medium, wavelength)
//                 .unwrap();
//
//             // For all incident angles; generate samples on each patch
//             for (i, patch) in emitter.patches.iter().enumerate() {
//                 // Emit rays from the patch of the emitter. Uniform sampling
// over the patch.                 let rays =
// patch.emit_rays(desc.emitter.num_rays, radius);                 log::debug!(
//                     "Emitted {} rays from patch {} - {:?}: {:?}",
//                     rays.len(),
//                     i,
//                     patch,
//                     rays
//                 );
//
//                 // Populate Embree ray stream with generated rays.
//                 let mut ray_stream = embree::RayN::new(rays.len());
//                 for (i, mut ray) in ray_stream.iter_mut().enumerate() {
//                     ray.set_origin(rays[i].o.into());
//                     ray.set_dir(rays[i].d.into());
//                 }
//
//                 // Trace primary rays with coherent context.
//                 let mut coherent_ctx = embree::IntersectContext::coherent();
//                 let ray_hit =
//                     embree_rt.intersect_stream_soa(scene_id, ray_stream, &mut
// coherent_ctx);
//
//                 // Filter out primary rays that hit the surface.
//                 let filtered = ray_hit
//                     .iter()
//                     .enumerate()
//                     .filter_map(|(i, (_, h))| h.hit().then(|| i));
//
//                 let records = filtered
//                     .into_iter()
//                     .map(|i| {
//                         let ray = Ray {
//                             o: ray_hit.ray.org(i).into(),
//                             d: ray_hit.ray.dir(i).into(),
//                             e: 1.0,
//                         };
//                         trace_one_ray_grid_tracing(ray, &grid_rt, ior_t,
// None)                     })
//                     .collect::<Vec<_>>();
//                 println!("{:?}", records);
//             }
//         }
//     }
// }

// Approach 1: sort filtered rays to continue take advantage of
// coherent tracing
// Approach 2: trace each filtered ray with incoherent context
// Approach 3: using heightfield tracing method to trace rays

// fn trace_one_ray_grid_tracing(
//     ray: Ray,
//     rt_grid: &GridRayTracing,
//     ior_t: RefractiveIndex,
//     record: Option<RayTraceRecord>,
// ) -> Option<RayTraceRecord> { if let Some(isect) = rt_grid.trace_ray(ray) {
//   if let Some(Scattering { reflected, .. }) = scattering_air_conductor(ray,
//   isect.hit_point, isect.normal,
// ior_t.eta, ior_t.k)         {
//             if reflected.e >= 0.0 {
//                 let curr_record = RayTraceRecord {
//                     initial: record.as_ref().unwrap().initial,
//                     current: ray,
//                     bounces: record.as_ref().unwrap().bounces + 1,
//                 };
//                 trace_one_ray_grid_tracing(reflected, rt_grid, ior_t,
// Some(curr_record))             } else {
//                 record
//             }
//         } else {
//             record
//         }
//     } else {
//         record
//     }
// }

//! Measurement of the BSDF (Bidirectional Scattering Distribution Function) of
//! micro-surfaces.

use super::{params::BsdfMeasurementParams, DataCarriedOnHemisphereSampler};
#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::embr;
#[cfg(feature = "vdbg")]
use crate::measure::bsdf::rtc::RayTrajectory;
use crate::{
    app::{cache::RawCache, cli::ansi},
    measure::{
        bsdf::{
            emitter::Emitter,
            receiver::{BounceAndEnergy, Receiver},
            rtc::RtcMethod,
        },
        params::SimulationKind,
        AnyMeasured, Measurement, MeasurementSource,
    },
};
use base::{
    bxdf::brdf::measured::{
        ClausenBrdf, ClausenBrdfParameterisation, MeasuredBrdfKind, Origin, VgonioBrdf,
        VgonioBrdfParameterisation,
    },
    error::VgonioError,
    math::{rcp_f64, Sph2, Vec3},
    units::{Degs, Nanometres, Radians, Rads},
    utils::{handle::Handle, medium::Medium, partition::SphericalPartition},
    AnyMeasuredBrdf, BrdfLevel, MeasurementKind,
};
use chrono::{DateTime, Local};
use jabr::array::DyArr;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    path::Path,
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
    pub raw: RawMeasuredBsdfData,
    /// Collected BSDF data.
    pub bsdfs: HashMap<BrdfLevel, VgonioBrdf>,
}

impl AnyMeasured for BsdfMeasurement {
    fn has_multiple_levels(&self) -> bool { true }

    fn kind(&self) -> MeasurementKind { MeasurementKind::Bsdf }

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

    /// Resamples the BSDF data to match the `ClausenBrdfParameterisation`.
    pub fn resample(
        &self,
        params: &ClausenBrdfParameterisation,
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
            ClausenBrdfParameterisation {
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
    use base::{
        units::nm,
        utils::{
            partition::{PartitionScheme, SphericalDomain},
            range::StepRangeIncl,
        },
        BrdfLevel,
    };

    use super::*;
    use crate::measure::bsdf::{
        emitter::EmitterParams, receiver::ReceiverParams, rtc::RtcMethod::Grid,
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
        let spectrum_range = StepRangeIncl::new(nm!(100.0), nm!(400.0), nm!(100.0));
        let spectrum = DyArr::from_iterator([-1], spectrum_range.values());
        let n_spectrum = spectrum.len();
        let mut bsdfs = HashMap::new();
        bsdfs.insert(
            BrdfLevel(0),
            VgonioBrdf {
                origin: Origin::RealWorld,
                incident_medium: Medium::Vacuum,
                transmitted_medium: Medium::Aluminium,
                params: Box::new(VgonioBrdfParameterisation {
                    n_zenith_i: 1,
                    incoming: DyArr::zeros([1]),
                    outgoing: partition.clone(),
                }),
                spectrum: spectrum.clone(),
                samples: DyArr::ones([1, n_wo, 4]),
                kind: MeasuredBrdfKind::Vgonio,
            },
        );
        let measured = BsdfMeasurement {
            params: BsdfMeasurementParams {
                emitter: EmitterParams {
                    num_rays: 0,
                    num_sectors: 1,
                    max_bounces: 0,
                    zenith: StepRangeIncl::new(Rads::ZERO, Rads::ZERO, Rads::ZERO),
                    azimuth: StepRangeIncl::new(Rads::ZERO, Rads::ZERO, Rads::ZERO),
                    spectrum: spectrum_range,
                },
                receivers: vec![ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision,
                    scheme: PartitionScheme::Beckers,
                }],
                kind: BsdfKind::Brdf,
                sim_kind: SimulationKind::GeomOptics(Grid),
                incident_medium: Medium::Vacuum,
                fresnel: Default::default(),
                transmitted_medium: Medium::Vacuum,
            },
            raw: RawMeasuredBsdfData {
                n_zenith_in: 1,
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

/// Ray tracing simulation result for a single incident direction of a surface.
pub struct SingleSimResult {
    /// Incident direction in the unit spherical coordinates.
    pub wi: Sph2,
    /// Trajectories of the rays.
    #[cfg(feature = "vdbg")]
    pub trajectories: Box<[RayTrajectory]>,
    /// Number of bounces of the rays.
    #[cfg(not(feature = "vdbg"))]
    pub bounces: Box<[u32]>,
    /// Final directions of the rays.
    #[cfg(not(feature = "vdbg"))]
    pub dirs: Box<[Vec3]>,
    /// Energy of the rays per wavelength.
    #[cfg(not(feature = "vdbg"))]
    pub energy: DyArr<f32, 2>,
}

/// Iterator over the rays in the simulation result.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRays<'a> {
    idx: usize,
    result: &'a SingleSimResult,
}

/// Single ray information in the simulation result.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRay<'a> {
    /// Bounce of the ray.
    pub bounce: &'a u32,
    /// Direction of the ray.
    pub dir: &'a Vec3,
    /// Energy of the ray per wavelength.
    pub energy: &'a [f32],
}

#[cfg(not(feature = "vdbg"))]
impl SingleSimResult {
    /// Returns an iterator over the rays in the simulation result.
    pub fn iter_rays(&self) -> SingleSimResultRays {
        debug_assert_eq!(self.bounces.len(), self.dirs.len(), "Length mismatch");
        debug_assert_eq!(
            self.bounces.len() * self.energy.shape()[1],
            self.energy.len(),
            "Length mismatch"
        );

        SingleSimResultRays {
            idx: 0,
            result: self,
        }
    }

    /// Returns an iterator over the rays in the simulation result in chunks.
    pub fn iter_ray_chunks(&self, chunk_size: usize) -> SingleSimResultRayChunks {
        debug_assert_eq!(self.bounces.len(), self.dirs.len(), "Length mismatch");
        debug_assert_eq!(
            self.bounces.len() * self.energy.shape()[1],
            self.energy.len(),
            "Length mismatch"
        );

        SingleSimResultRayChunks {
            chunk_idx: 0,
            chunk_size,
            chunk_count: (self.bounces.len() + chunk_size - 1) / chunk_size,
            result: self,
        }
    }
}

#[cfg(not(feature = "vdbg"))]
impl<'a> Iterator for SingleSimResultRays<'a> {
    type Item = SingleSimResultRay<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.result.bounces.len() {
            let bounce = &self.result.bounces[self.idx];
            let dir = &self.result.dirs[self.idx];
            let n_spectrum = self.result.energy.shape()[1];
            let energy =
                &self.result.energy.as_slice()[self.idx * n_spectrum..(self.idx + 1) * n_spectrum];
            self.idx += 1;
            Some(SingleSimResultRay {
                bounce,
                dir,
                energy,
            })
        } else {
            None
        }
    }
}

#[cfg(not(feature = "vdbg"))]
impl<'a> ExactSizeIterator for SingleSimResultRays<'a> {
    fn len(&self) -> usize { self.result.bounces.len() }
}

/// Chunks of rays in the simulation result.
///
/// This is useful for processing the rays in parallel.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRayChunks<'a> {
    chunk_idx: usize,
    chunk_size: usize,
    chunk_count: usize,
    result: &'a SingleSimResult,
}

#[cfg(not(feature = "vdbg"))]
impl<'a> Iterator for SingleSimResultRayChunks<'a> {
    type Item = SingleSimResultRayChunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_rays = self.result.bounces.len();
        if self.chunk_idx < self.chunk_count {
            let start = self.chunk_idx * self.chunk_size;
            let end = usize::min((self.chunk_idx + 1) * self.chunk_size, n_rays);
            let size = self.result.bounces.len().min(end) - start;
            let bounces = &self.result.bounces[start..start + size];
            let dirs = &self.result.dirs[start..start + size];
            let n_spectrum = self.result.energy.shape()[1];
            let energy = &self.result.energy.as_slice()[start * n_spectrum..end * n_spectrum];
            self.chunk_idx += 1;
            Some(SingleSimResultRayChunk {
                size,
                n_spectrum,
                bounces,
                dirs,
                energy,
                curr: 0,
            })
        } else {
            None
        }
    }
}

/// A chunk of rays in the simulation result.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRayChunk<'a> {
    /// Number of rays in the chunk.
    pub size: usize,
    /// Number of wavelengths.
    pub n_spectrum: usize,
    /// Bounces of the rays.
    pub bounces: &'a [u32],
    /// Directions of the rays.
    pub dirs: &'a [Vec3],
    /// Energy of the rays per wavelength.
    pub energy: &'a [f32],
    /// Current index of the iterator.
    pub curr: usize,
}

#[cfg(not(feature = "vdbg"))]
impl<'a> Iterator for SingleSimResultRayChunk<'a> {
    type Item = SingleSimResultRay<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.size {
            let idx = self.curr;
            let bounce = &self.bounces[idx];
            let dir = &self.dirs[idx];
            let energy = &self.energy[idx * self.n_spectrum..(idx + 1) * self.n_spectrum];
            self.curr += 1;
            Some(SingleSimResultRay {
                bounce,
                dir,
                energy,
            })
        } else {
            None
        }
    }
}

#[cfg(not(feature = "vdbg"))]
impl<'a> ExactSizeIterator for SingleSimResultRayChunk<'a> {
    fn len(&self) -> usize { self.size }
}

// /// Measures the BSDF of a surface using geometric ray tracing methods.
// pub fn measure_bsdf_rt(
//     params: BsdfMeasurementParams,
//     handles: &[Handle<MicroSurface>],
//     cache: &RawCache,
// ) -> Box<[Measurement]> {
//     let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
//     let surfaces = cache.get_micro_surfaces(handles);
//     let emitter = Emitter::new(&params.emitter);
//     let receiver = Receiver::new(&params.receivers, &params, cache);
//     let n_wi = emitter.measpts.len();
//     let n_wo = receiver.patches.n_patches();
//     let n_spectrum = params.emitter.spectrum.step_count();
//     let spectrum = DyArr::from_iterator([-1],
// params.emitter.spectrum.values());     #[cfg(not(feature = "visu-dbg"))]
//     let iors_i = cache
//         .iors
//         .ior_of_spectrum(params.incident_medium, spectrum.as_ref())
//         .unwrap();
//     #[cfg(not(feature = "visu-dbg"))]
//     let iors_t = cache
//         .iors
//         .ior_of_spectrum(params.transmitted_medium, spectrum.as_ref())
//         .unwrap();
//
//     log::debug!(
//         "Measuring BSDF of {} surfaces from {} measurement points with {}
// rays",         surfaces.len(),
//         emitter.measpts.len(),
//         emitter.params.num_rays,
//     );
//
//     let mut measurements = Vec::with_capacity(surfaces.len());
//     for (surf, mesh) in surfaces.iter().zip(meshes) {
//         if surf.is_none() || mesh.is_none() {
//             log::debug!("Skipping surface {:?} and its mesh {:?}", surf,
// mesh);             continue;
//         }
//
//         let surf = surf.unwrap();
//         let mesh = mesh.unwrap();
//
//         log::info!(
//             "Measuring surface {}",
//             surf.path.as_ref().unwrap().display()
//         );
//
//         let sim_results = match &params.sim_kind {
//             SimulationKind::GeomOptics(method) => {
//                 println!(
//                     "    {} Measuring {} with geometric optics...",
//                     ansi::YELLOW_GT,
//                     params.kind
//                 );
//                 match method {
//                     #[cfg(feature = "embree")]
//                     RtcMethod::Embree => embr::simulate_bsdf_measurement(
//                         #[cfg(not(feature = "visu-dbg"))]
//                         &params,
//                         &emitter,
//                         mesh,
//                         #[cfg(not(feature = "visu-dbg"))]
//                         &iors_i,
//                         #[cfg(not(feature = "visu-dbg"))]
//                         &iors_t,
//                     ),
//                     #[cfg(feature = "optix")]
//                     RtcMethod::Optix => rtc_simulation_optix(&params, mesh,
// &emitter, cache),                     RtcMethod::Grid =>
// rtc_simulation_grid(&params, surf, mesh, &emitter, cache),                 }
//             }
//             SimulationKind::WaveOptics => {
//                 println!(
//                     "    {} Measuring {} with wave optics...",
//                     ansi::YELLOW_GT,
//                     params.kind
//                 );
//                 todo!("Wave optics simulation is not yet implemented")
//             }
//         };
//
//         let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
//         log::trace!("Estimated orbit radius: {}", orbit_radius);
//
//         let incoming = DyArr::<Sph2>::from_slice([n_wi], &emitter.measpts);
//
//         #[cfg(feature = "visu-dbg")]
//         let mut trajectories: Box<[MaybeUninit<Box<[RayTrajectory]>>]> =
//             Box::new_uninit_slice(n_wi);
//         #[cfg(feature = "visu-dbg")]
//         let mut hit_points: Box<[MaybeUninit<Vec<Vec3>>]> =
// Box::new_uninit_slice(n_wi);
//
//         let mut records = DyArr::splat(Option::<BounceAndEnergy>::None,
// [n_wi, n_wo, n_spectrum]);         let mut stats:
// Box<[MaybeUninit<SingleBsdfMeasurementStats>]> = Box::new_uninit_slice(n_wi);
//
//         for (i, sim) in sim_results.into_iter().enumerate() {
//             #[cfg(feature = "visu-dbg")]
//             let trjs = trajectories[i].as_mut_ptr();
//             #[cfg(feature = "visu-dbg")]
//             let hpts = hit_points[i].as_mut_ptr();
//             let recs =
//                 &mut records.as_mut_slice()[i * n_wo * n_spectrum..(i + 1) *
// n_wo * n_spectrum];
//
//             #[cfg(feature = "bench")]
//             let t = std::time::Instant::now();
//
//             println!(
//                 "        {} Collecting BSDF snapshot {}{}/{}{}...",
//                 ansi::YELLOW_GT,
//                 ansi::BRIGHT_CYAN,
//                 i + 1,
//                 n_wi,
//                 ansi::RESET
//             );
//
//             // Collect the tracing data into raw bsdf snapshots.
//             receiver.collect(
//                 sim,
//                 stats[i].as_mut_ptr(),
//                 recs,
//                 #[cfg(feature = "visu-dbg")]
//                 orbit_radius,
//                 #[cfg(feature = "visu-dbg")]
//                 params.fresnel,
//                 #[cfg(feature = "visu-dbg")]
//                 trjs,
//                 #[cfg(feature = "visu-dbg")]
//                 hpts,
//             );
//
//             #[cfg(feature = "bench")]
//             {
//                 let elapsed = t.elapsed();
//                 log::debug!(
//                     "bsdf measurement data collection (one snapshot) took {}
// secs.",                     elapsed.as_secs_f64()
//                 );
//             }
//         }
//
//         let raw = RawMeasuredBsdfData {
//             n_zenith_in: emitter.params.zenith.step_count_wrapped(),
//             spectrum: spectrum.clone(),
//             incoming,
//             outgoing: receiver.patches.clone(),
//             records,
//             stats: DyArr::from_boxed_slice([n_wi], unsafe {
// stats.assume_init() }),             #[cfg(feature = "visu-dbg")]
//             trajectories: unsafe { trajectories.assume_init() },
//             #[cfg(feature = "visu-dbg")]
//             hit_points: unsafe { hit_points.assume_init() },
//         };
//
//         let bsdfs = raw.compute_bsdfs(params.incident_medium,
// params.transmitted_medium);
//
//         measurements.push(Measurement {
//             name: surf.file_stem().unwrap().to_owned(),
//             source: MeasurementSource::Measured(Handle::with_id(surf.uuid)),
//             timestamp: chrono::Local::now(),
//             measured: Box::new(MeasuredBsdfData { params, raw, bsdfs }),
//         });
//     }
//
//     measurements.into_boxed_slice()
// }

/// Measures the BSDF of a surface using geometric ray tracing methods.
pub fn measure_bsdf_rt(
    params: BsdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    cache: &RawCache,
) -> Box<[Measurement]> {
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = cache.get_micro_surfaces(handles);
    let emitter = Emitter::new(&params.emitter);
    let n_wi = emitter.measpts.len();
    let n_spectrum = params.emitter.spectrum.step_count();
    let spectrum = DyArr::from_iterator([-1], params.emitter.spectrum.values());
    #[cfg(not(feature = "vdbg"))]
    let iors_i = cache
        .iors
        .ior_of_spectrum(params.incident_medium, spectrum.as_ref())
        .unwrap();
    #[cfg(not(feature = "vdbg"))]
    let iors_t = cache
        .iors
        .ior_of_spectrum(params.transmitted_medium, spectrum.as_ref())
        .unwrap();

    log::debug!(
        "Measuring BSDF of {} surfaces from {} measurement points with {} rays",
        surfaces.len(),
        emitter.measpts.len(),
        emitter.params.num_rays,
    );

    let incoming = DyArr::<Sph2>::from_slice([n_wi], &emitter.measpts);
    let mut measurements = Vec::with_capacity(surfaces.len() * params.receivers.len());

    for (surf, mesh) in surfaces.iter().zip(meshes) {
        #[cfg(feature = "vdbg")]
        let mut trajectories: Vec<Vec<RayTrajectory>> =
            vec![Vec::with_capacity(emitter.params.num_rays as usize); n_wi];
        #[cfg(feature = "vdbg")]
        let mut hit_points: Vec<Vec<Vec3>> = vec![Vec::new(); n_wi];

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

        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        log::trace!("Estimated orbit radius: {}", orbit_radius);

        // Receiver with its records & stats
        let mut receivers = params
            .receivers
            .iter()
            .map(|rparams| {
                let r = Receiver::new(rparams, &params, cache);
                let records = DyArr::splat(
                    Option::<BounceAndEnergy>::None,
                    [n_wi, r.n_wo(), n_spectrum],
                );
                let stats: Box<[Option<SingleBsdfMeasurementStats>]> =
                    vec![None; n_wi].into_boxed_slice();
                (r, records, stats, *rparams)
            })
            .collect::<Box<_>>();

        match &params.sim_kind {
            SimulationKind::GeomOptics(method) => {
                #[cfg(feature = "embree")]
                let (_, scene, geometry) = embr::create_resources(mesh);
                for sector in emitter.circular_sectors() {
                    for (i, wi) in sector.measpts.iter().enumerate() {
                        #[cfg(feature = "bench")]
                        let t = std::time::Instant::now();
                        let single_result = match method {
                            #[cfg(feature = "embree")]
                            RtcMethod::Embree => embr::simulate_bsdf_measurement_single_point(
                                *wi,
                                &sector,
                                mesh,
                                geometry.clone(),
                                &scene,
                                #[cfg(not(feature = "vdbg"))]
                                params.fresnel,
                                #[cfg(not(feature = "vdbg"))]
                                &iors_i,
                                #[cfg(not(feature = "vdbg"))]
                                &iors_t,
                            ),
                            _ => unimplemented!("Temporarily deactivated"),
                        };
                        #[cfg(feature = "bench")]
                        {
                            let elapsed = t.elapsed();
                            log::debug!(
                                "bsdf measurement simulation (one snapshot) took {} secs.",
                                elapsed.as_secs_f64()
                            );
                        }

                        for (j, (receiver, records, stats, _)) in receivers.iter_mut().enumerate() {
                            let n_wo = receiver.n_wo();
                            let recs = &mut records.as_mut_slice()
                                [i * n_wo * n_spectrum..(i + 1) * n_wo * n_spectrum];

                            #[cfg(feature = "bench")]
                            let t = std::time::Instant::now();

                            println!(
                                "        {} Collecting BSDF snapshot {}{}/{}{} to receiver #{}...",
                                ansi::YELLOW_GT,
                                ansi::BRIGHT_CYAN,
                                i + 1,
                                n_wi,
                                ansi::RESET,
                                j
                            );

                            // Print receiver number of patches
                            println!(
                                "Receiver number of patches: {}",
                                receiver.patches.n_patches()
                            );

                            // Collect the tracing data into raw bsdf snapshots.
                            receiver.collect(
                                &single_result,
                                &mut stats[i],
                                recs,
                                #[cfg(feature = "vdbg")]
                                orbit_radius,
                                #[cfg(feature = "vdbg")]
                                params.fresnel,
                                #[cfg(feature = "vdbg")]
                                &mut trajectories[i],
                                #[cfg(feature = "vdbg")]
                                &mut hit_points[i],
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
                    }
                }

                #[cfg(feature = "vdbg")]
                let trajectories = trajectories
                    .into_iter()
                    .map(|t| t.into_boxed_slice())
                    .collect::<Box<_>>();
                #[cfg(feature = "vdbg")]
                let hit_points = hit_points.into_boxed_slice();

                for (receiver, records, stats, rparams) in receivers {
                    let stats = unsafe {
                        std::mem::transmute::<
                            Box<[Option<SingleBsdfMeasurementStats>]>,
                            Box<[SingleBsdfMeasurementStats]>,
                        >(stats)
                    };
                    let raw = RawMeasuredBsdfData {
                        n_zenith_in: emitter.params.zenith.step_count_wrapped(),
                        spectrum: spectrum.clone(),
                        incoming: incoming.clone(),
                        outgoing: receiver.patches,
                        records,
                        stats: DyArr::from_slice([n_wi], &stats),
                        #[cfg(feature = "vdbg")]
                        trajectories: trajectories.clone(),
                        #[cfg(feature = "vdbg")]
                        hit_points: hit_points.clone(),
                    };
                    let bsdfs =
                        raw.compute_bsdfs(params.incident_medium, params.transmitted_medium);
                    let params = BsdfMeasurementParams {
                        kind: params.kind,
                        sim_kind: params.sim_kind,
                        incident_medium: params.incident_medium,
                        transmitted_medium: params.transmitted_medium,
                        emitter: params.emitter,
                        receivers: vec![rparams],
                        fresnel: params.fresnel,
                    };
                    measurements.push(Measurement {
                        name: surf.file_stem().unwrap().to_owned(),
                        source: MeasurementSource::Measured(Handle::with_id(surf.uuid)),
                        timestamp: Local::now(),
                        measured: Box::new(BsdfMeasurement { params, raw, bsdfs }),
                    });
                }
            },
            SimulationKind::WaveOptics => {
                println!(
                    "    {} Measuring {} with wave optics...",
                    ansi::YELLOW_GT,
                    params.kind
                );
                todo!("Wave optics simulation is not yet implemented")
            },
        }
    }

    measurements.into_boxed_slice()
}

/// Brdf measurement of a microfacet surface using the grid ray tracing.
fn rtc_simulation_grid<'a>(
    _params: &'a BsdfMeasurementParams,
    _surf: &'a MicroSurface,
    _mesh: &'a MicroSurfaceMesh,
    _emitter: &'a Emitter,
    _cache: &'a RawCache,
) -> Box<dyn Iterator<Item = SingleSimResult>> {
    // Temporary deactivated
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
    todo!()
}

/// Brdf measurement of a microfacet surface using the OptiX ray tracing.
#[cfg(feature = "optix")]
fn rtc_simulation_optix<'a>(
    _params: &'a BsdfMeasurementParams,
    _surf: &'a MicroSurfaceMesh,
    _emitter: &'a Emitter,
    _cache: &'a RawCache,
) -> Box<dyn Iterator<Item = SingleSimResult>> {
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

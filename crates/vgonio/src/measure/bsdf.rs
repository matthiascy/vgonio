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
            receiver::{BounceAndEnergy, CollectedData, DataRetrieval, PerPatchData, Receiver},
            rtc::{RayTrajectory, RtcMethod},
        },
        data::{MeasuredData, MeasurementData, MeasurementDataSource, SampledBrdf},
        microfacet::MeasuredAdfData,
        params::SimulationKind,
    },
};
#[cfg(any(feature = "visu-dbg", debug_assertions))]
use base::math::Vec3;
use base::{
    error::VgonioError,
    math::{circular_angle_dist, projected_barycentric_coords, rcp_f32, Sph2},
    partition::{PartitionScheme, SphericalPartition},
    units::{Degs, Nanometres, Radians, Rads},
};
use bxdf::brdf::measured::{MeasuredBrdf, VgonioBrdf, VgonioBrdfParameterisation};
use jabr::array::DyArr;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt::{Debug, Display, Formatter},
    path::Path,
};
use surf::{MicroSurface, MicroSurfaceMesh};

pub mod emitter;
pub(crate) mod params;
pub mod receiver;
pub mod rtc;

/// BSDF measurement data.
///
/// The Number of emitted rays, wavelengths, and bounces are invariant over
/// emitter's position.
///
/// At each emitter's position, each emitted ray carries an initial energy
/// equals to 1.
#[derive(Debug, Clone)]
pub struct MeasuredBsdfData {
    /// Parameters of the measurement.
    pub params: BsdfMeasurementParams,
    /// Snapshot of the BSDF at each incident direction of the emitter.
    /// See [`BsdfSnapshot`] for more details.
    pub snapshots: Box<[BsdfSnapshot]>,
    // TODO: write the max_values to the vgmo file.
    /// Maximum values of the BSDF samples for each incident direction and each
    /// wavelength; stored in 1D row major order array [ωi, λ].
    pub max_values: Box<[f32]>,
    /// Raw snapshots of the BSDF containing the full measurement data.
    /// This field is only available when the
    /// [`crate::measure::bsdf::params::DataRetrievalMode`] is set to
    /// `FullData`.
    /// See [`BsdfSnapshotRaw`] for more details.
    pub raw_snapshots: Option<Box<[BsdfSnapshotRaw<BounceAndEnergy>]>>,
}

impl MeasuredBsdfData {
    pub fn into_measured_brdf(self) -> VgonioBrdf {
        use rayon::{
            iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
            slice::ParallelSliceMut,
        };
        // TODO: NdArray ergonomics
        let mut incoming = DyArr::<Sph2>::zeros([self.snapshots.len()]);
        for (i, snapshot) in self.snapshots.iter().enumerate() {
            incoming[i] = snapshot.wi;
        }
        let params = VgonioBrdfParameterisation {
            incoming,
            outgoing: self.params.receiver.partitioning(),
        };
        let n_wi = self.snapshots.len();
        let n_wo = params.outgoing.n_patches();
        let n_spectrum = self.params.emitter.spectrum.step_count();
        let mut spectrum = DyArr::<Nanometres>::zeros([n_spectrum]);
        for (i, w) in self.params.emitter.spectrum.values().enumerate() {
            spectrum[[i]] = w;
        }
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        samples
            .as_mut_slice()
            .par_chunks_mut(n_wo * n_spectrum)
            .zip(self.snapshots.par_iter())
            .for_each(|(s, snapshot)| {
                s.copy_from_slice(snapshot.samples.as_slice());
            });
        VgonioBrdf::new(
            self.params.incident_medium,
            self.params.transmitted_medium,
            params,
            spectrum,
            samples,
        )
    }

    /// Writes the BSDF data to images in exr format.
    ///
    /// Only BSDF data are written to the images. The full measurement data
    /// are not written.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The path to the directory where the images will be saved.
    /// * `timestamp` - The timestamp of the measurement.
    /// * `resolution` - The resolution of the images.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let n_wi = self.snapshots.len();
        let n_wavelength = self.params.emitter.spectrum.step_count();
        let (w, h) = (resolution as usize, resolution as usize);
        let wavelengths = self
            .params
            .emitter
            .spectrum
            .values()
            .collect::<Vec<_>>()
            .into_boxed_slice();

        // The BSDF data are stored in a single flat row-major array, with the order of
        // the dimensions [wi, λ, y, x] where x is the width, y is the height, λ is the
        // wavelength, and wi is the incident direction.
        let mut bsdf_images = vec![0.0; n_wi * n_wavelength * w * h];
        // Pre-compute the patch index for each pixel.
        let mut patch_indices = vec![0i32; w * h].into_boxed_slice();
        let partition = self.params.receiver.partitioning();
        partition.compute_pixel_patch_indices(resolution, resolution, &mut patch_indices);
        {
            // Each snapshot is saved as a separate layer of the image.
            // Each channel of the layer stores the BSDF data for a single wavelength.
            for (l, snapshot) in self.snapshots.iter().enumerate() {
                let offset = l * n_wavelength * w * h;
                for i in 0..w {
                    for j in 0..h {
                        let idx = patch_indices[i + j * w];
                        if idx < 0 {
                            continue;
                        }
                        for k in 0..n_wavelength {
                            bsdf_images[offset + k * w * h + j * w + i] =
                                snapshot.samples[[idx as usize, k]];
                        }
                    }
                }
            }

            let layers = self
                .snapshots
                .iter()
                .enumerate()
                .map(|(snap_idx, snapshot)| {
                    let theta = format!("{:4.2}", snapshot.wi.theta.in_degrees().as_f32())
                        .replace(".", "_");
                    let phi =
                        format!("{:4.2}", snapshot.wi.phi.in_degrees().as_f32()).replace(".", "_");
                    let layer_attrib = LayerAttributes {
                        owner: Text::new_or_none("vgonio"),
                        capture_date: Text::new_or_none(base::utils::iso_timestamp_from_datetime(
                            timestamp,
                        )),
                        software_name: Text::new_or_none("vgonio"),
                        other: self.params.to_exr_extra_info(),
                        layer_name: Some(Text::new_or_panic(format!("θ{}.φ{}", theta, phi))),
                        ..LayerAttributes::default()
                    };
                    let offset = snap_idx * w * h * wavelengths.len();
                    let channels = wavelengths
                        .iter()
                        .enumerate()
                        .map(|(i, wavelength)| {
                            let name = Text::new_or_panic(format!("{}", wavelength));
                            AnyChannel::new(
                                name,
                                FlatSamples::F32(Cow::Borrowed(
                                    &bsdf_images[offset + i * w * h..offset + (i + 1) * w * h],
                                )),
                            )
                        })
                        .collect::<Vec<_>>();
                    Layer::new(
                        (w, h),
                        layer_attrib,
                        Encoding::FAST_LOSSLESS,
                        AnyChannels {
                            list: SmallVec::from(channels),
                        },
                    )
                })
                .collect::<Vec<_>>();

            let img_attrib = ImageAttributes::new(IntegerBounds::new((0, 0), (w, h)));
            let image = Image::from_layers(img_attrib, layers);
            image.write().to_file(filepath).map_err(|err| {
                VgonioError::new(
                    format!(
                        "Failed to write BSDF measurement data to image file: {}",
                        err
                    ),
                    Some(Box::new(err)),
                )
            })?;
        }

        if let Some(raw_snapshots) = &self.raw_snapshots {
            log::debug!("Writing BSDF bounces measurement data to image file.");
            let max_bounces = self
                .raw_snapshots
                .as_ref()
                .map(|snapshots| {
                    snapshots
                        .iter()
                        .map(|snap| snap.stats.n_bounce)
                        .max()
                        .unwrap()
                })
                .unwrap_or(0) as usize;
            if max_bounces == 0 {
                log::info!("No bounces to save.");
                return Ok(());
            }

            // Save the BSDF samples for each wavelength and each bounce.
            {
                let desired = n_wi * max_bounces * h * w;
                if desired > bsdf_images.len() {
                    // Try to reuse the bsdf_images buffer with the new size.
                    // [wi, bounce, y, x]
                    bsdf_images = vec![0.0; desired];
                }
            }

            // Save the percentage of rays that have bounced a certain number of times for
            // each patch.
            {
                let desired = n_wi * max_bounces * h * w;
                if desired > bsdf_images.len() {
                    // Try to reuse the bsdf_images buffer with the new size.
                    // [wi, bounce, y, x]
                    bsdf_images = vec![0.0; desired];
                }
                bsdf_images.fill(0.0);

                // Each snapshot is saved as a separate layer of the image.
                // Each channel of the layer stores the percentage of rays that have bounced a
                // certain number of times.
                for (snap_idx, snapshot) in raw_snapshots.iter().enumerate() {
                    let offset = snap_idx * max_bounces * h * w;
                    let rcp_n_received = rcp_f32(snapshot.stats.n_received as f32);
                    for i in 0..w {
                        for j in 0..h {
                            let patch_idx = patch_indices[i + j * w];
                            if patch_idx < 0 {
                                continue;
                            }
                            for k in 0..snapshot.stats.n_bounce as usize {
                                bsdf_images[offset + k * w * h + j * w + i] = snapshot.records
                                    [patch_idx as usize * n_wavelength]
                                    .n_ray_per_bounce[k]
                                    as f32
                                    * rcp_n_received;
                            }
                        }
                    }
                }

                let layers = raw_snapshots
                    .iter()
                    .enumerate()
                    .filter_map(|(snap_idx, snapshot)| {
                        let theta = format!("{:4.2}", snapshot.wi.theta.in_degrees().as_f32())
                            .replace(".", "_");
                        let phi = format!("{:4.2}", snapshot.wi.phi.in_degrees().as_f32())
                            .replace(".", "_");
                        let offset = snap_idx * max_bounces * h * w;
                        if snapshot.stats.n_bounce == 0 {
                            return None;
                        }
                        let n_bounce = snapshot.stats.n_bounce as usize;
                        let (percentage_per_bounce, channels): (
                            Vec<f32>,
                            Vec<AnyChannel<FlatSamples>>,
                        ) = (0..n_bounce)
                            .map(|bounce_idx| {
                                let name =
                                    Text::new_or_panic(format!("bounce-{:03}", bounce_idx + 1));
                                let samples = &bsdf_images[offset + bounce_idx * w * h
                                    ..offset + (bounce_idx + 1) * w * h];
                                // Only take the first wavelength.
                                let sum_percent = snapshot.stats.n_ray_per_bounce
                                    [0 * n_bounce + bounce_idx]
                                    as f32
                                    / snapshot.stats.n_received as f32;
                                (
                                    sum_percent,
                                    AnyChannel::new(name, FlatSamples::F32(Cow::Borrowed(samples))),
                                )
                            })
                            .unzip();
                        let mut other_info = self.params.to_exr_extra_info();
                        for (i, bounce_percent) in percentage_per_bounce.iter().enumerate() {
                            other_info.insert(
                                Text::new_or_panic(format!("bounce-{:03}", i + 1)),
                                AttributeValue::Text(Text::new_or_panic(format!(
                                    "{:.2}%",
                                    bounce_percent
                                ))),
                            );
                        }
                        let layer_attrib = LayerAttributes {
                            owner: Text::new_or_none("vgonio"),
                            capture_date: Text::new_or_none(
                                base::utils::iso_timestamp_from_datetime(timestamp),
                            ),
                            software_name: Text::new_or_none("vgonio"),
                            other: other_info,
                            layer_name: Some(Text::new_or_panic(format!("θ{}.φ{}", theta, phi))),
                            ..LayerAttributes::default()
                        };
                        Some(Layer::new(
                            (w, h),
                            layer_attrib,
                            Encoding::FAST_LOSSLESS,
                            AnyChannels {
                                list: SmallVec::from(channels),
                            },
                        ))
                    })
                    .collect::<Vec<_>>();

                let filename = format!(
                    "{}_ray_percent_per_patch_per_bounce.exr",
                    filepath.file_stem().unwrap().to_str().unwrap()
                );
                let img_attrib = ImageAttributes::new(IntegerBounds::new((0, 0), (w, h)));
                let image = Image::from_layers(img_attrib, layers);
                image
                    .write()
                    .to_file(filepath.with_file_name(filename))
                    .map_err(|err| {
                        VgonioError::new(
                            format!(
                                "Failed to write BSDF bounces measurement data to image file: {}",
                                err
                            ),
                            Some(Box::new(err)),
                        )
                    })?;
            }
        }
        Ok(())
    }

    /// Returns the number of samples (number of incident & outgoing direction
    /// pairs) in total without considering the wavelength.
    #[inline(always)]
    pub fn num_samples(&self) -> usize { self.params.receiver.num_patches() * self.snapshots.len() }

    #[cfg(feature = "visu-dbg")]
    /// Returns the trajectories of the rays for each BSDF snapshot.
    pub fn trajectories(&self) -> Vec<Vec<RayTrajectory>> {
        self.snapshots
            .iter()
            .map(|snapshot| snapshot.trajectories.clone())
            .collect()
    }

    #[cfg(feature = "visu-dbg")]
    /// Returns the hit points on the collector for each BSDF snapshot.
    pub fn hit_points(&self) -> Vec<Vec<Vec3>> {
        self.snapshots
            .iter()
            .map(|snapshot| snapshot.hit_points.clone().into_vec())
            .collect()
    }

    /// Extracts the NDF from the measured BSDF.
    pub fn extract_ndf(&self) -> MeasuredAdfData {
        todo!()
        // let params =  AdfMeasurementParams {
        //     azimuth: RangeByStepSizeInclusive {},
        //     zenith: RangeByStepSizeInclusive {},
        // };
        // for snapshot in &self.snapshots {
        //     for (patch, samples) in
        // snapshot.samples.iter().zip(ndf.samples.iter_mut()) {
        //         for (wavelength, sample) in samples.iter_mut().enumerate() {
        //             *sample += patch[wavelength];
        //         }
        //     }
        // }
        // ndf
    }

    /// Extracts the BRDF from the measured BSDF data.
    pub fn sampled_brdf(&self, s: &SampledBrdf, dense: bool, phi_offset: Radians) -> SampledBrdf {
        let spectrum = self
            .params
            .emitter
            .spectrum
            .values()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let n_lambda = spectrum.len();
        let n_wi = s.n_wi();
        let n_wo = s.n_wo();
        let n_wo_dense = if dense { n_wo * 4 } else { n_wo };

        // row-major [wi, wo, spectrum]
        let mut samples = vec![0.0; n_wi * n_wo_dense * n_lambda].into_boxed_slice();
        // row-major [wi, spectrum]
        let mut max_values = vec![0.0; n_wi * n_lambda].into_boxed_slice();

        let wo_step = Degs::new(2.5).to_radians();
        let wi_wo_pairs = if dense {
            s.wi_wo_pairs
                .iter()
                .map(|(wi, wos)| {
                    let wos_new = wos
                        .iter()
                        .flat_map(|wo| {
                            if wo.approx_eq(&Sph2::zero()) {
                                vec![Sph2::zero(); 1]
                            } else if (wo.theta - wi.theta).abs().as_f32() < 1e-6 {
                                let mut wos_out = vec![*wo; 7];
                                // The incident and outgoing directions are the same.
                                if wo.phi.as_f32().abs() < 1e-6 {
                                    // The azimuth of the incident direction is almost zero.
                                    for i in 0..4 {
                                        wos_out[i] = Sph2::new(
                                            wo.theta - wo_step * (3 - i) as f32,
                                            wo.phi + phi_offset,
                                        );
                                    }
                                    for i in 0..3 {
                                        wos_out[i + 4] = Sph2::new(
                                            wo.theta - wo_step * (3 - i) as f32,
                                            Rads::PI + phi_offset,
                                        );
                                    }
                                } else {
                                    for i in 0..3 {
                                        wos_out[i] = Sph2::new(
                                            wo.theta - wo_step * (3 - i) as f32,
                                            Rads::ZERO + phi_offset,
                                        );
                                    }
                                    for i in 0..4 {
                                        wos_out[i + 3] = Sph2::new(
                                            wo.theta - wo_step * (3 - i) as f32,
                                            wo.phi + phi_offset,
                                        );
                                    }
                                }
                                wos_out
                            } else {
                                let mut wos_out = vec![*wo; 4];
                                for i in 0..4 {
                                    wos_out[i] = Sph2::new(
                                        wo.theta - wo_step * (3 - i) as f32,
                                        wo.phi + phi_offset,
                                    );
                                }
                                wos_out
                            }
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    (*wi, wos_new)
                })
                .collect::<Vec<(Sph2, Box<[Sph2]>)>>()
                .into_boxed_slice()
        } else {
            s.wi_wo_pairs.clone()
        };

        // Get the interpolated samples for wi, wo, and spectrum.
        wi_wo_pairs.iter().enumerate().for_each(|(i, (wi, wos))| {
            let samples_offset_wi = i * n_wo_dense * n_lambda;
            let snap_idx = self
                .snapshots
                .iter()
                .position(|snap| snap.wi.approx_eq(&wi))
                .expect(
                    "The incident direction is not found in the BSDF snapshots. The incident \
                     direction must be one of the directions of the emitter.",
                );
            let max_offset = i * n_lambda;
            let use_original_max = std::env::var("ORIGINAL_MAX")
                .map(|v| v == "1")
                .unwrap_or(false);
            if use_original_max {
                max_values[max_offset..max_offset + n_lambda].copy_from_slice(
                    &self.max_values[snap_idx * n_lambda..(snap_idx + 1) * n_lambda],
                );
                log::trace!(
                    "Original max values: {:?}",
                    &self.max_values[snap_idx * n_lambda..(snap_idx + 1) * n_lambda]
                );
            }
            for (j, wo) in wos.iter().enumerate() {
                let samples_offset = samples_offset_wi + j * n_lambda;
                self.sample_at(
                    *wi,
                    *wo,
                    &mut samples[samples_offset..samples_offset + n_lambda],
                );
                if !use_original_max {
                    for k in 0..n_lambda {
                        max_values[max_offset + k] =
                            f32::max(max_values[max_offset + k], samples[samples_offset + k]);
                    }
                }
            }
            if !use_original_max {
                log::trace!(
                    "Max values: {:?}",
                    &max_values[max_offset..max_offset + n_lambda]
                );
            }
        });

        SampledBrdf {
            spectrum,
            samples,
            max_values,
            wi_wo_pairs,
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
    pub fn sample_at(&self, wi: Sph2, wo: Sph2, interpolated: &mut [f32]) {
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
        let snapshot = self
            .snapshots
            .iter()
            .find(|snap| snap.wi.approx_eq(&wi))
            .expect(
                "The incident direction is not found in the BSDF snapshots. The incident \
                 direction must be one of the directions of the emitter.",
            );
        log::trace!("  - Found snapshot at wi: {}", wi);
        let n_wavelengths = self.params.emitter.spectrum.step_count();
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
                    let patch0_samples = &snapshot.samples.as_slice()
                        [patch_idx.0 * n_wavelengths..(patch_idx.0 + 1) * n_wavelengths];
                    let patch1_samples = &snapshot.samples.as_slice()
                        [patch_idx.1 * n_wavelengths..(patch_idx.1 + 1) * n_wavelengths];
                    let patch2_samples = &snapshot.samples.as_slice()
                        [patch_idx.2 * n_wavelengths..(patch_idx.2 + 1) * n_wavelengths];
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
                    let patch0_samples = &snapshot.samples.as_slice()
                        [patch0_idx * n_wavelengths..(patch0_idx + 1) * n_wavelengths];
                    let patch1_samples = &snapshot.samples.as_slice()
                        [patch1_idx * n_wavelengths..(patch1_idx + 1) * n_wavelengths];
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
                    let upper_patch0_samples = &snapshot.samples.as_slice()[upper_patch_idx.0
                        * n_wavelengths
                        ..(upper_patch_idx.0 + 1) * n_wavelengths];
                    let upper_patch1_samples = &snapshot.samples.as_slice()[upper_patch_idx.1
                        * n_wavelengths
                        ..(upper_patch_idx.1 + 1) * n_wavelengths];
                    let lower_patch0_samples = &snapshot.samples.as_slice()[lower_patch_idx.0
                        * n_wavelengths
                        ..(lower_patch_idx.0 + 1) * n_wavelengths];
                    let lower_patch1_samples = &snapshot.samples.as_slice()[lower_patch_idx.1
                        * n_wavelengths
                        ..(lower_patch_idx.1 + 1) * n_wavelengths];
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

pub(crate) fn compute_bsdf_snapshots_max_values(
    snapshots: &[BsdfSnapshot],
    n_wavelength: usize,
) -> Box<[f32]> {
    let n_wi = snapshots.len();
    let mut max_values = vec![0.0; n_wi * n_wavelength].into_boxed_slice();
    snapshots.iter().enumerate().for_each(|(i, snapshot)| {
        let offset = i * n_wavelength;
        snapshot
            .samples
            .as_slice()
            .chunks(n_wavelength)
            .for_each(|patch_samples| {
                patch_samples.iter().enumerate().for_each(|(j, val)| {
                    max_values[offset + j] = f32::max(max_values[offset + j], *val);
                });
            });
    });
    max_values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::bsdf::{
        emitter::EmitterParams, receiver::ReceiverParams, rtc::RtcMethod::Grid,
    };
    use base::{
        medium::Medium, partition::SphericalDomain, range::RangeByStepSizeInclusive, units::nm,
    };
    use jabr::array::DynArr;

    #[test]
    fn test_measured_bsdf_sample() {
        let precision = Sph2 {
            theta: Rads::from_degrees(30.0),
            phi: Rads::from_degrees(2.0),
        };
        let partition =
            SphericalPartition::new(PartitionScheme::Beckers, SphericalDomain::Upper, precision);
        let samples = DyArr::ones([partition.n_patches(), 4]);
        let snapshots = Box::new([BsdfSnapshot {
            wi: Sph2 {
                theta: Rads::ZERO,
                phi: Rads::ZERO,
            },
            samples,
            trajectories: Vec::new(),
            hit_points: Vec::new().into_boxed_slice(),
        }]);
        let measured = MeasuredBsdfData {
            params: BsdfMeasurementParams {
                emitter: EmitterParams {
                    num_rays: 0,
                    max_bounces: 0,
                    zenith: RangeByStepSizeInclusive::new(Rads::ZERO, Rads::ZERO, Rads::ZERO),
                    azimuth: RangeByStepSizeInclusive::new(Rads::ZERO, Rads::ZERO, Rads::ZERO),
                    spectrum: RangeByStepSizeInclusive::new(nm!(100.0), nm!(400.0), nm!(100.0)),
                },
                receiver: ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision,
                    scheme: PartitionScheme::Beckers,
                    retrieval: Default::default(),
                },
                kind: BsdfKind::Brdf,
                sim_kind: SimulationKind::GeomOptics(Grid),
                incident_medium: Medium::Vacuum,
                fresnel: Default::default(),
                transmitted_medium: Medium::Vacuum,
            },
            snapshots,
            raw_snapshots: None,
            max_values: vec![1.0; 4].into_boxed_slice(),
        };
        let mut interpolated = vec![0.0, 0.0, 0.0, 0.0];
        let wi = Sph2 {
            theta: Rads::ZERO,
            phi: Rads::ZERO,
        };
        measured.sample_at(
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
            wi,
            Sph2 {
                theta: Rads::from_degrees(80.0),
                phi: Rads::from_degrees(30.0),
            },
            &mut interpolated,
        );
        assert_eq!(interpolated, vec![1.0, 1.0, 1.0, 1.0]);

        measured.sample_at(
            wi,
            Sph2 {
                theta: Rads::from_degrees(40.0),
                phi: Rads::from_degrees(30.0),
            },
            &mut interpolated,
        );
        assert_eq!(interpolated, vec![1.0, 1.0, 1.0, 1.0]);

        measured.sample_at(
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
#[derive(Clone)]
pub struct BsdfMeasurementStatsPoint {
    /// Number of bounces (actual bounce, not always equals to the maximum
    /// bounce limit).
    pub n_bounce: u32,
    /// Number of emitted rays that hit the surface; invariant over wavelength.
    pub n_received: u32,
    /// Number of wavelengths, which is the number of samples in the spectrum.
    pub n_wavelength: usize,
    /// Statistics of the number of emitted rays that either
    /// 1. hit the surface and were absorbed, or
    /// 2. hit the surface and were reflected, or
    /// 3. hit the surface and were captured by the collector.
    /// The statistics are variant over wavelength. The data is stored as a
    /// flat array in row-major order with the dimensions (statistics,
    /// wavelength). The index of the statistics is defined by the constants
    /// `ABSORBED_IDX`, `REFLECTED_IDX`, and `CAPTURED_IDX`.
    /// The total number of elements in the array is `3 * n_wavelength`.
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

impl PartialEq for BsdfMeasurementStatsPoint {
    fn eq(&self, other: &Self) -> bool {
        self.n_bounce == other.n_bounce
            && self.n_wavelength == other.n_wavelength
            && self.n_received == other.n_received
            && self.n_ray_stats == other.n_ray_stats
            && self.e_captured == other.e_captured
            && self.n_ray_per_bounce == other.n_ray_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}

impl BsdfMeasurementStatsPoint {
    /// Index of the number of absorbed rays statistics in the `n_ray_stats`
    /// array.
    pub const ABSORBED_IDX: usize = 0;
    /// Index of the number of reflected rays statistics in the `n_ray_stats`
    /// array.
    pub const REFLECTED_IDX: usize = 1;
    /// Index of the number of captured rays statistics in the `n_ray_stats`
    /// array.
    pub const CAPTURED_IDX: usize = 2;

    /// Creates an empty `BsdfMeasurementStatsPoint`.
    ///
    /// # Arguments
    /// * `n_wavelength`: Number of wavelengths.
    /// * `max_bounce`: Maximum number of bounces. This is used to pre-allocate
    ///   the memory for the histograms.
    pub fn new(n_wavelength: usize, max_bounce: usize) -> Self {
        Self {
            n_bounce: max_bounce as u32,
            n_received: 0,
            n_wavelength,
            n_ray_stats: vec![0; 3 * n_wavelength].into_boxed_slice(),
            e_captured: vec![0.0; n_wavelength].into_boxed_slice(),
            n_ray_per_bounce: vec![0; n_wavelength * max_bounce].into_boxed_slice(),
            energy_per_bounce: vec![0.0; n_wavelength * max_bounce].into_boxed_slice(),
        }
    }

    /// Returns the number of absorbed rays statistics per wavelength.
    pub fn n_absorbed(&self) -> &[u32] {
        let offset = Self::ABSORBED_IDX * self.n_wavelength;
        &self.n_ray_stats[offset..offset + self.n_wavelength]
    }

    /// Returns the number of absorbed rays statistics per wavelength.
    pub fn n_absorbed_mut(&mut self) -> &mut [u32] {
        let offset = Self::ABSORBED_IDX * self.n_wavelength;
        &mut self.n_ray_stats[offset..offset + self.n_wavelength]
    }

    /// Returns the number of reflected rays statistics per wavelength.
    pub fn n_reflected(&self) -> &[u32] {
        let offset = Self::REFLECTED_IDX * self.n_wavelength;
        &self.n_ray_stats[offset..offset + self.n_wavelength]
    }

    /// Returns the number of reflected rays statistics per wavelength.
    pub fn n_reflected_mut(&mut self) -> &mut [u32] {
        let offset = Self::REFLECTED_IDX * self.n_wavelength;
        &mut self.n_ray_stats[offset..offset + self.n_wavelength]
    }

    /// Returns the number of captured rays statistics per wavelength.
    pub fn n_captured(&self) -> &[u32] {
        let offset = Self::CAPTURED_IDX * self.n_wavelength;
        &self.n_ray_stats[offset..offset + self.n_wavelength]
    }

    /// Returns the number of captured rays statistics per wavelength.
    pub fn n_captured_mut(&mut self) -> &mut [u32] {
        let offset = Self::CAPTURED_IDX * self.n_wavelength;
        &mut self.n_ray_stats[offset..offset + self.n_wavelength]
    }
}

impl Default for BsdfMeasurementStatsPoint {
    fn default() -> Self { Self::new(0, 0) }
}

impl Debug for BsdfMeasurementStatsPoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"BsdfMeasurementPointStats:
    - n_bounces: {},
    - n_received: {},
    - n_absorbed: {:?},
    - n_reflected: {:?},
    - n_captured: {:?},
    - total_energy_captured: {:?},
    - num_rays_per_bounce: {:?},
    - energy_per_bounce: {:?},
"#,
            self.n_bounce,
            self.n_received,
            self.n_absorbed(),
            self.n_reflected(),
            self.n_captured(),
            self.e_captured,
            self.n_ray_per_bounce,
            self.energy_per_bounce
        )
    }
}

/// A snapshot of the BSDF during the measurement.
///  
/// This is the data collected at a single incident direction of the emitter.
///
/// It contains the statistics of the measurement and the data collected
/// for all the patches of the collector at the incident direction.
#[derive(Clone)]
pub struct BsdfSnapshotRaw<D>
where
    D: PerPatchData,
{
    /// Incident direction in the unit spherical coordinates.
    pub wi: Sph2,
    /// Statistics of the measurement at the point.
    pub stats: BsdfMeasurementStatsPoint,
    /// A list of data collected for each patch of the collector.
    /// The data is stored as a flat array in row-major order with the
    /// dimensions (patch, wavelength).
    pub records: Box<[D]>,
    /// Extra ray trajectory data for debugging purposes.
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    pub trajectories: Vec<RayTrajectory>,
    /// Hit points on the collector.
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    pub hit_points: Box<[Vec3]>,
}

impl<D: Debug + PerPatchData> Debug for BsdfSnapshotRaw<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BsdfMeasurementPoint")
            .field("stats", &self.stats)
            .field("data", &self.records)
            .finish()
    }
}

impl<D: PerPatchData + PartialEq> PartialEq for BsdfSnapshotRaw<D> {
    fn eq(&self, other: &Self) -> bool {
        self.stats == other.stats && self.records == other.records
    }
}

/// A snapshot of the measured BSDF at a single incident direction.
#[derive(Debug, Clone, PartialEq)]
pub struct BsdfSnapshot {
    /// Incident direction in the unit spherical coordinates.
    pub wi: Sph2,
    /// BSDF values for each patch of the collector and each wavelength.
    /// Two-dimensional data (patch, wavelength) (ωo, λ) stored in a flat array
    /// in row-major order.
    pub samples: DyArr<f32, 2>,
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    /// Extra ray trajectory data for debugging purposes.
    pub trajectories: Vec<RayTrajectory>,
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    /// Hit points on the collector for debugging purposes.
    pub hit_points: Box<[Vec3]>,
}

/// Ray tracing simulation result for a single incident direction of a surface.
pub struct SimulationResultPoint {
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
) -> Box<[MeasurementData]> {
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = cache.get_micro_surfaces(handles);
    let emitter = Emitter::new(&params.emitter);
    let receiver = Receiver::new(&params.receiver, &params, cache);

    log::debug!(
        "Measuring BSDF of {} surfaces from {} measurement points.",
        surfaces.len(),
        emitter.measpts.len()
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

        let sim_result_points = match sim_kind {
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
        let mut collected = CollectedData::empty(Handle::with_id(surf.uuid), &receiver.patches);
        for sim_result_point in sim_result_points {
            log::debug!("Collecting BSDF snapshot at {}", sim_result_point.wi);

            #[cfg(feature = "bench")]
            let t = std::time::Instant::now();

            // Collect the tracing data into raw bsdf snapshots.
            receiver.collect(
                &sim_result_point,
                &mut collected,
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

        let snapshots = collected.compute_bsdf(&params);
        let raw_snapshots = match params.receiver.retrieval {
            DataRetrieval::FullData => Some(collected.snapshots.into_boxed_slice()),
            DataRetrieval::BsdfOnly => None,
        };
        let max_values =
            compute_bsdf_snapshots_max_values(&snapshots, params.emitter.spectrum.step_count());
        measurements.push(MeasurementData {
            name: surf.file_stem().unwrap().to_owned(),
            source: MeasurementDataSource::Measured(collected.surface),
            timestamp: chrono::Local::now(),
            measured: MeasuredData::Bsdf(MeasuredBsdfData {
                params,
                snapshots,
                raw_snapshots,
                max_values,
            }),
        })
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
) -> Box<dyn Iterator<Item = SimulationResultPoint>> {
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
) -> Box<dyn Iterator<Item = SimulationResultPoint>> {
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

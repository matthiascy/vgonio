use crate::{
    cache::ComputeCache,
    measurement::{Measurement, MeasurementSource},
    params::{NdfMeasurementMode, NdfMeasurementParams},
};
use std::{
    fs::File,
    io::{BufWriter, Seek, Write},
    path::Path,
};
use vgn_bxdf::impl_any_measured_trait;
use vgn_core::{
    error::VgonioError,
    math,
    math::{theta, Sph2, Vec3Swizzles},
    res::Handle,
    units,
    units::{rad, Radians},
    utils::{
        partition::{DataCarriedOnHemisphereImageWriter, SphericalDomain, SphericalPartition},
        range::StepRangeIncl,
    },
    MeasurementKind,
};
use vgn_io::{MicroSurface, MicroSurfaceMesh};

use vgn_bxdf::AnyMeasured;

/// Structure holding the data for microfacet area distribution measurement.
///
/// TODO: add distribution for the microfacet slope and normal.
///
/// D(m) is the micro-facet area (normal) distribution function, which gives the
/// relative number of facets oriented in any given direction, or, more
/// precisely, the relative total facet surface area per unit solid angle of
/// surface normals pointed in any given direction.
///
/// Microfacet area distribution function (MADF)
/// Microfacet slope distribution function (MSDF)
/// Microfacet normal distribution function (MNDF)
#[derive(Debug, Clone)]
pub struct MeasuredNdfData {
    /// The measurement parameters.
    pub params: NdfMeasurementParams,
    /// The distribution data. The outermost index is the azimuthal angle of the
    /// microfacet normal, and the inner index is the zenith angle of the
    /// microfacet normal.
    pub samples: Box<[f32]>,
}

impl_any_measured_trait!(MeasuredNdfData, Ndf);

impl MeasuredNdfData {
    /// Returns the Area Distribution Function data slice for the given
    /// azimuthal angle in radians.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    ///
    /// Azimuthal angle will be wrapped around to the range [0, 2π).
    ///
    /// 2π will be mapped to 0.
    ///
    /// # Arguments
    ///
    /// * `azim` - Azimuthal angle of the microfacet normal in radians.
    pub fn slice_at(&self, azim: Radians) -> (&[f32], Option<&[f32]>) {
        if self.params.mode.is_by_points() {
            let (azi, zen) = self.measurement_range().unwrap();
            let azim_m = azim.wrap_to_tau();
            let azim_m_idx = azi.index_of(azim_m);
            let opposite_azim_m = azim_m.opposite();
            let opposite_azim_idx = if azi.start <= opposite_azim_m && opposite_azim_m <= azi.stop {
                Some(azi.index_of(opposite_azim_m))
            } else {
                None
            };
            let zen_step_count = zen.step_count_wrapped();
            (
                &self.samples[azim_m_idx * zen_step_count..(azim_m_idx + 1) * zen_step_count],
                opposite_azim_idx.map(|index| {
                    &self.samples[index * zen_step_count..(index + 1) * zen_step_count]
                }),
            )
        } else {
            todo!("Implement slice_at for the partition mode.")
            // TODO: check implementation in cmd_plot -> ndf
        }
    }

    // TODO: remove or make it internal for quick testing.
    /// Writes the measured data as an EXR file.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        let mut writer = BufWriter::new(
            File::create(filepath)
                .map_err(|e| VgonioError::from_io_error(e, "Failed to create NDF EXR file."))?,
        );
        self.write_as_exr_to(&mut writer, timestamp, resolution)?;
        writer
            .flush()
            .map_err(|e| VgonioError::from_io_error(e, "Failed to flush NDF EXR file."))
    }

    /// Writes the NDF as a single-encoding EXR into `writer` (anything
    /// `Write + Seek`). The path-taking [`Self::write_as_exr`] wraps this.
    pub fn write_as_exr_to<W: Write + Seek>(
        &self,
        writer: &mut W,
        timestamp: &chrono::DateTime<chrono::Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let partition = SphericalPartition::new(
            self.params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            self.params.mode.partition_precision_for_data_collection(),
        );

        // Collect the data following the patches.
        let mut samples_per_patch = vec![0.0; partition.n_patches()];
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, azimuth } => {
                assert!(
                    zenith.step_size > rad!(0.0) && azimuth.step_size > rad!(0.0),
                    "The step size of zenith and azimuth must be greater than 0."
                );
                let n_theta = StepRangeIncl::zero_to_half_pi(zenith.step_size).step_count_wrapped();
                let n_phi = StepRangeIncl::zero_to_tau(azimuth.step_size).step_count_wrapped();
                // NDF samples in ByPoints mode are stored by azimuth first, then by zenith.
                // We need to rearrange the data to match the patch order, which is by zenith
                // first, then by azimuth.
                samples_per_patch
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i_p, v)| {
                        let i_theta = i_p / n_phi;
                        let i_phi = i_p % n_phi;
                        let i_adf = i_phi * n_theta + i_theta;
                        // In case, the number of samples is less than the number of patches.
                        if i_adf < self.samples.len() {
                            *v = self.samples[i_adf];
                        }
                    });
            },
            NdfMeasurementMode::ByPartition { .. } => {
                samples_per_patch.copy_from_slice(&self.samples);
            },
        }

        DataCarriedOnHemisphereImageWriter::new(&partition, resolution).write_as_exr_to(
            &samples_per_patch,
            writer,
            timestamp,
            |_| Some(Text::from("NDF")),
            |_| Text::from("NDF"),
        )
    }

    // TODO: review the necessity of this method.
    /// Returns the measurement range of the azimuthal and zenith angles.
    /// The azimuthal angle is in the range [0, 2π] and the zenith angle is in
    /// the range [0, π/2].
    pub fn measurement_range(&self) -> Option<(StepRangeIncl<Radians>, StepRangeIncl<Radians>)> {
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, azimuth } => Some((azimuth, zenith)),
            NdfMeasurementMode::ByPartition { .. } => {
                eprintln!("Measurement range is not available for the partition mode.");
                None
            },
        }
    }

    /// Returns the zenith range of the measurement only if the measurement is
    /// in the ByPoints mode.
    pub fn zenith_range(&self) -> Option<StepRangeIncl<Radians>> {
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, .. } => Some(zenith),
            NdfMeasurementMode::ByPartition { .. } => {
                eprintln!("Zenith range is not available for the partition mode.");
                None
            },
        }
    }

    /// Writes the NDF as a .vgndf archive (zip container with three EXR encodings).
    pub fn write_as_vgndf(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        disc_res: u32,
    ) -> Result<(), VgonioError> {
        let file = File::create(filepath)
            .map_err(|e| VgonioError::from_io_error(e, "Failed to create .vgndf archive."))?;
        self.write_as_vgndf_to(BufWriter::new(file), timestamp, disc_res)
            .map(|_| ())
    }

    /// Writes the NDF as a `.vgndf` archive into `writer` (anything
    /// `Write + Seek`), returning the writer (so in-memory callers can recover
    /// the bytes). The path-taking [`Self::write_as_vgndf`] wraps this.
    pub fn write_as_vgndf_to<W: Write + Seek>(
        &self,
        writer: W,
        timestamp: &chrono::DateTime<chrono::Local>,
        disc_res: u32,
    ) -> Result<W, VgonioError> {
        use vgn_io::vgbsdf::{manifest::*, spectrum::SpectrumToml, OutputKind, VgbsdfWriter};

        let partition = SphericalPartition::new(
            self.params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            self.params.mode.partition_precision_for_data_collection(),
        );

        // Collect the data following the patches.
        let mut samples_per_patch = vec![0.0; partition.n_patches()];
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, azimuth } => {
                assert!(
                    zenith.step_size > rad!(0.0) && azimuth.step_size > rad!(0.0),
                    "The step size of zenith and azimuth must be greater than 0."
                );
                let n_theta = StepRangeIncl::zero_to_half_pi(zenith.step_size).step_count_wrapped();
                let n_phi = StepRangeIncl::zero_to_tau(azimuth.step_size).step_count_wrapped();
                // NDF samples in ByPoints mode are stored by azimuth first, then by zenith.
                // We need to rearrange the data to match the patch order, which is by zenith
                // first, then by azimuth.
                samples_per_patch
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i_p, v)| {
                        let i_theta = i_p / n_phi;
                        let i_phi = i_p % n_phi;
                        let i_adf = i_phi * n_theta + i_theta;
                        // In case, the number of samples is less than the number of patches.
                        if i_adf < self.samples.len() {
                            *v = self.samples[i_adf];
                        }
                    });
            },
            NdfMeasurementMode::ByPartition { .. } => {
                samples_per_patch.copy_from_slice(&self.samples);
            },
        }

        let layer = vgn_core::utils::partition::HemisphereLayer {
            layer_name: "NDF".into(),
            channel_names: vec!["value".into()],
            samples: &samples_per_patch,
        };

        let n_phi = partition
            .rings
            .iter()
            .map(|r| r.patch_count as u32)
            .max()
            .unwrap_or(1);
        let n_theta = partition.rings.len() as u32;

        let mut vgb = VgbsdfWriter::new(writer);
        vgb.write_metadata(
            &Manifest {
                vgonio: VgonioBlock {
                    version: env!("CARGO_PKG_VERSION").into(),
                    conventions: "v1".into(),
                },
                archive: ArchiveBlock {
                    created: vgn_core::utils::iso_timestamp_from_datetime(timestamp),
                    kind: "ndf".into(),
                    description: None,
                },
                material: None,
                bsdf: None,
                provenance: None,
            },
            &partition,
            None,
            &SpectrumToml::scalar(), // NDF carries one scalar "value" channel
        )?;
        vgb.write_level(
            "l0",
            &partition,
            std::slice::from_ref(&layer),
            OutputKind::Ndf,
            timestamp,
            disc_res,
            (n_phi, n_theta),
        )?;
        let mut w = vgb.finish()?;
        w.flush()
            .map_err(|e| VgonioError::from_io_error(e, "Failed to flush .vgndf archive."))?;
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgn_io::vgbsdf::{conventions::OutgoingEncoding, VgbsdfReader};

    fn read_exr_pixels(path: &std::path::Path) -> Vec<f32> {
        use exr::prelude::*;
        let img = exr::prelude::read_first_flat_layer_from_file(path).unwrap();
        let layer = &img.layer_data;
        let ch = &layer.channel_data.list[0];
        match &ch.sample_data {
            FlatSamples::F32(v) => v.to_vec(),
            _ => panic!("expected F32 samples"),
        }
    }

    /// Smoke test: drive `write_as_vgndf` end-to-end with a tiny `ByPartition`
    /// NDF, then re-open with `VgbsdfReader` and run the lazy validator. This
    /// covers (1) writer-side metadata, (2) all three EXR encodings being
    /// well-formed, and (3) reader-writer agreement on the scalar-channel
    /// contract.
    #[test]
    fn write_as_vgndf_smoke() {
        let params = NdfMeasurementParams {
            mode: NdfMeasurementMode::ByPartition {
                precision: rad!(0.3),
            },
            crop_to_disk: false,
            use_facet_area: false,
        };
        let partition = SphericalPartition::new(
            params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            params.mode.partition_precision_for_data_collection(),
        );
        let samples: Box<[f32]> = (0..partition.n_patches())
            .map(|i| i as f32 * 0.01)
            .collect::<Vec<_>>()
            .into();
        let data = MeasuredNdfData { params, samples };

        let tmp = tempfile::Builder::new()
            .suffix(".vgndf")
            .tempfile()
            .unwrap();
        let path = tmp.path().to_path_buf();
        let ts = chrono::Local::now();

        data.write_as_vgndf(&path, &ts, 64).unwrap();

        let mut reader = VgbsdfReader::open(&path).unwrap();
        assert_eq!(reader.manifest.archive.kind, "ndf");
        assert_eq!(reader.manifest.vgonio.conventions, "v1");
        assert!(
            reader.spectrum.is_scalar(),
            "NDF archive must declare scalar spectrum, got {:?}",
            reader.spectrum
        );
        assert_eq!(
            reader.partition.n_patches as usize,
            partition.n_patches(),
            "partition.toml n_patches drifted from the source partition"
        );
        assert!(
            reader.incident_grid.is_none(),
            "NDF archives must not carry an incident grid"
        );
        // Lazy validator: opens each EXR header, checks dim + channel count.
        reader.validate_levels().unwrap();
    }

    /// Equivalence test: the disc EXR inside a `.vgndf` (written by
    /// `write_as_vgndf`) must contain bit-identical pixel data to the
    /// single-disc `.exr` from `write_as_exr` for the same input. Both
    /// writers ultimately rasterize via `compute_pixel_patch_indices`, so
    /// any drift indicates a regression in one of the two paths.
    ///
    /// We only compare pixel samples, not EXR attributes / layer names —
    /// the new writer adds VGONIO Conventions v1 attrs and renames the
    /// channel from "NDF" to "value", which is intentional.
    #[test]
    fn write_as_vgndf_disc_matches_write_as_exr() {
        let params = NdfMeasurementParams {
            mode: NdfMeasurementMode::ByPartition {
                precision: rad!(0.3),
            },
            crop_to_disk: false,
            use_facet_area: false,
        };
        let partition = SphericalPartition::new(
            params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            params.mode.partition_precision_for_data_collection(),
        );
        let samples: Box<[f32]> = (0..partition.n_patches())
            .map(|i| 0.123 + i as f32 * 0.01)
            .collect::<Vec<_>>()
            .into();
        let data = MeasuredNdfData { params, samples };
        let ts = chrono::Local::now();
        let resolution = 64u32;

        let dir = tempfile::tempdir().unwrap();
        let exr_path = dir.path().join("old.exr");
        let vgndf_path = dir.path().join("new.vgndf");

        data.write_as_exr(&exr_path, &ts, resolution).unwrap();
        data.write_as_vgndf(&vgndf_path, &ts, resolution).unwrap();

        // Old path: read the .exr directly.
        let old_pixels = read_exr_pixels(&exr_path);

        // New path: extract l0/disc.exr from the .vgndf zip, stage to a tempfile, read.
        let mut reader = VgbsdfReader::open(&vgndf_path).unwrap();
        let disc_bytes = reader.read_exr_bytes("l0", OutgoingEncoding::Disc).unwrap();
        let disc_path = dir.path().join("disc.exr");
        std::fs::write(&disc_path, &disc_bytes).unwrap();
        let new_pixels = read_exr_pixels(&disc_path);

        assert_eq!(
            old_pixels.len(),
            new_pixels.len(),
            "pixel count differs: old={} new={}",
            old_pixels.len(),
            new_pixels.len()
        );
        assert_eq!(
            old_pixels.len(),
            (resolution * resolution) as usize,
            "expected {} pixels, got {}",
            resolution * resolution,
            old_pixels.len()
        );
        let mut max_delta = 0.0f32;
        for (i, (a, b)) in old_pixels.iter().zip(new_pixels.iter()).enumerate() {
            let d = (a - b).abs();
            if d > max_delta {
                max_delta = d;
            }
            assert!(
                d < 1e-6,
                "pixel {i}: write_as_exr={a} write_as_vgndf={b} (Δ={d})"
            );
        }
        // Sanity: at least one non-zero pixel survived (i.e. we're not comparing
        // two all-zero images and falsely passing).
        assert!(
            old_pixels.iter().any(|&v| v != 0.0),
            "all pixels are zero — comparison would pass vacuously"
        );
        eprintln!(
            "write_as_vgndf vs write_as_exr disc pixels: max Δ = {max_delta} over {} pixels",
            old_pixels.len()
        );
    }
}

/// Size of the chunk of facets to process in parallel.
const FACET_CHUNK_SIZE: usize = 4096;

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_area_distribution(
    params: NdfMeasurementParams,
    handles: &[Handle],
    cache: &ComputeCache,
) -> Box<[Measurement]> {
    #[cfg(feature = "bench")]
    let start = std::time::Instant::now();
    log::info!("Measuring microfacet area distribution...");

    let surfs = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = handles.iter().zip(surfs.iter()).zip(meshes.iter());

    let measurements = match params.mode {
        NdfMeasurementMode::ByPoints { .. } => {
            measure_area_distribution_by_points(surfaces, params)
        },
        NdfMeasurementMode::ByPartition { .. } => {
            measure_area_distribution_by_partition(surfaces, params)
        },
    };

    #[cfg(feature = "bench")]
    {
        let elapsed = start.elapsed();
        log::info!("NDF measurement took {} ms.", elapsed.as_millis());
    }
    measurements.into_boxed_slice()
}

/// Measure the microfacet area distribution function by sampling the hemisphere
/// with equal angle steps.
fn measure_area_distribution_by_points<'a>(
    surfaces: impl Iterator<
        Item = (
            (&'a Handle, &'a Option<&'a MicroSurface>),
            &'a Option<&'a MicroSurfaceMesh>,
        ),
    >,
    params: NdfMeasurementParams,
) -> Vec<Measurement> {
    use rayon::prelude::*;
    let (azimuth, zenith) = params.mode.as_mode_by_points().unwrap();
    surfaces
        .filter_map(|((hdl, surface), mesh)| {
            if surface.is_none() || mesh.is_none() {
                log::debug!("Skipping a surface because it is not loaded {:?}.", mesh);
                return None;
            }
            let mesh = mesh.unwrap();
            let half_zenith_bin_width = zenith.step_size * 0.5;
            let half_zenith_bin_width_cos = half_zenith_bin_width.cos();
            log::info!(
                "  -- Measuring the NDF of surface: {}",
                surface.unwrap().file_stem().unwrap()
            );
            log::trace!(
                "  -- macro surface area (mesh): {}",
                mesh.macro_surface_area()
            );
            log::trace!("  -- macro surface area: {}", surface.unwrap().macro_area());
            log::trace!("  -- micro facet total area: {}", mesh.facet_total_area);
            log::trace!("  -- micro facet count: {}", mesh.facet_normals.len());

            let zenith_step_count = zenith.step_count_wrapped();
            // Sort the facets into bins according to their normal direction's zenith angle.
            let mut facets_bins = vec![vec![]; zenith_step_count].into_boxed_slice();
            let macro_area = if !params.crop_to_disk {
                for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                    let zen = math::theta(normal);
                    let idxs = classify_normal_by_zenith(zen, zenith, 1.0);
                    for idx in idxs {
                        if idx == 0xFF {
                            continue;
                        }
                        facets_bins[idx as usize].push(facet_idx);
                    }
                }
                if params.use_facet_area {
                    log::debug!("  -- macro surface area: {}", mesh.macro_surface_area());
                    log::debug!("  -- micro facet total area: {}", mesh.facet_total_area);
                    mesh.macro_surface_area()
                } else {
                    log::debug!(
                        "  -- macro surface area (normals count): {}",
                        mesh.facet_normals.len()
                    );
                    mesh.facet_normals.len() as f32
                }
            } else {
                let mut macro_area = 0.0; // Reset the macro surface area.
                let mut num_normals = 0u32;
                let extent = mesh.bounds.extent();
                let radius = extent.x.min(extent.y) * 0.5;
                for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                    let center = mesh.center_of_facet(facet_idx).xy();
                    if center.length() > radius {
                        continue;
                    }
                    macro_area +=
                        mesh.facet_areas[facet_idx] * theta(&mesh.facet_normals[facet_idx]).cos();
                    num_normals += 1;
                    let zen = math::theta(normal);
                    for idx in classify_normal_by_zenith(zen, zenith, 1.0) {
                        if idx == 0xFF {
                            continue;
                        }
                        facets_bins[idx as usize].push(facet_idx);
                    }
                }
                if params.use_facet_area {
                    log::debug!("  -- macro surface area (cropped): {}", macro_area);
                    macro_area
                } else {
                    log::debug!(
                        "  -- macro surface area (cropped, normal count): {}",
                        num_normals
                    );
                    num_normals as f32
                }
            };

            let solid_angle = units::solid_angle_of_spherical_cap(zenith.step_size).value();
            let denom_rcp = math::rcp_f32(macro_area * solid_angle);
            log::debug!("  -- solid angle: {}", solid_angle);
            log::debug!("  -- denom_rcp: {}", denom_rcp);

            let mut samples =
                vec![0.0f32; azimuth.step_count_wrapped() * zenith.step_count_wrapped()]
                    .into_boxed_slice();
            for azi_idx in 0..azimuth.step_count_wrapped() {
                for zen_idx in 0..zenith.step_count_wrapped() {
                    let azimuth = azi_idx as f32 * azimuth.step_size;
                    let zen = zen_idx as f32 * zenith.step_size;
                    let dir = math::sph_to_cart(zen, azimuth);
                    let facets = &facets_bins[zen_idx];

                    let facets_area = if params.use_facet_area {
                        facets
                            .par_chunks(FACET_CHUNK_SIZE)
                            .map(|idxs| {
                                idxs.iter().fold(0.0, |sum, idx| {
                                    let n = &mesh.facet_normals[*idx];
                                    let a = mesh.facet_areas[*idx];
                                    if n.dot(dir) <= half_zenith_bin_width_cos {
                                        sum
                                    } else {
                                        sum + a
                                    }
                                })
                            })
                            .sum::<f32>()
                    } else {
                        facets
                            .par_chunks(FACET_CHUNK_SIZE)
                            .map(|idxs| {
                                idxs.iter().fold(0, |sum, idx| {
                                    let n = &mesh.facet_normals[*idx];
                                    if n.dot(dir) <= half_zenith_bin_width_cos {
                                        sum
                                    } else {
                                        sum + 1
                                    }
                                })
                            })
                            .sum::<u32>() as f32
                    };

                    let sample_idx = azi_idx * zenith.step_count_wrapped() + zen_idx;
                    samples[sample_idx] = facets_area * denom_rcp;
                    log::trace!(
                        "-- φ: {}, θ: {}  | facet area: {} => {}",
                        azimuth.prettified(),
                        zen.prettified(),
                        facets_area,
                        samples[sample_idx]
                    );
                }
            }

            Some(Measurement {
                name: surface.unwrap().file_stem().unwrap().to_owned(),
                source: MeasurementSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: Box::new(MeasuredNdfData { params, samples }),
            })
        })
        .collect()
}

/// Measure the microfacet area distribution function by partitioning the
/// hemisphere into patches and calculating the corresponding values.
fn measure_area_distribution_by_partition<'a>(
    surfaces: impl Iterator<
        Item = (
            (&'a Handle, &'a Option<&'a MicroSurface>),
            &'a Option<&'a MicroSurfaceMesh>,
        ),
    >,
    params: NdfMeasurementParams,
) -> Vec<Measurement> {
    use rayon::prelude::*;
    let precision = match params.mode {
        NdfMeasurementMode::ByPoints { .. } => {
            panic!("The partition mode is not supported for the ByPoints mode.")
        },
        NdfMeasurementMode::ByPartition { precision } => precision,
    };
    let partition = SphericalPartition::new_beckers(SphericalDomain::Upper, precision);
    log::info!(
        "  -- Partitioning the hemisphere into {} patches.",
        partition.n_patches()
    );
    // Data buffer for data of each patch.
    let mut samples = vec![0.0; partition.n_patches()];
    samples.shrink_to_fit();
    surfaces
        .filter_map(|((hdl, surf), mesh)| {
            if surf.is_none() || mesh.is_none() {
                log::debug!("Skipping a surface because it is not loaded {:?}.", mesh);
                return None;
            }
            // Reset the patch data.
            samples.iter_mut().for_each(|v| *v = 0.0);
            let mesh = mesh.unwrap();

            log::info!(
                "  -- Measuring the NDF of surface: {}",
                surf.unwrap().file_stem().unwrap()
            );
            log::debug!(
                "  -- macro surface area (mesh): {}",
                mesh.macro_surface_area()
            );
            log::debug!("  -- macro surface area: {}", surf.unwrap().macro_area());
            log::debug!("  -- micro facet total area: {}", mesh.facet_total_area);
            log::debug!("  -- micro facet count: {}", mesh.facet_normals.len());

            let mut normals_per_patch = vec![vec![]; partition.n_patches()];

            let macro_area = if !params.crop_to_disk {
                for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                    match partition.contains(Sph2::from_cartesian(*normal)) {
                        None => {
                            log::warn!("Facet normal {} is not contained in any patch.", normal);
                        },
                        Some(patch_idx) => {
                            normals_per_patch[patch_idx].push(facet_idx);
                        },
                    }
                }
                if params.use_facet_area {
                    log::debug!("  -- macro surface area: {}", mesh.macro_surface_area());
                    mesh.macro_surface_area()
                } else {
                    log::debug!(
                        "  -- macro surface area (normals count): {}",
                        mesh.facet_normals.len()
                    );
                    mesh.facet_normals.len() as f32
                }
            } else {
                let mut macro_area = 0.0; // Reset the macro surface area.
                let mut num_normals = 0u32;
                let extent = mesh.bounds.extent();
                let radius = extent.x.min(extent.y) * 0.5;
                for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                    let center = mesh.center_of_facet(facet_idx).xy();
                    if center.length() > radius {
                        continue;
                    }
                    macro_area += mesh.facet_areas[facet_idx];
                    num_normals += 1;
                    match partition.contains(Sph2::from_cartesian(*normal)) {
                        None => {
                            log::warn!("Facet normal {} is not contained in any patch.", normal);
                        },
                        Some(patch_idx) => {
                            normals_per_patch[patch_idx].push(facet_idx);
                        },
                    }
                }
                if params.use_facet_area {
                    log::debug!("  -- macro surface area (cropped): {}", macro_area);
                    macro_area
                } else {
                    log::debug!(
                        "  -- macro surface area (cropped, normal count): {}",
                        num_normals
                    );
                    num_normals as f32
                }
            };

            normals_per_patch
                .par_iter()
                .enumerate()
                .zip(samples.par_iter_mut())
                .for_each(|((patch_idx, facet_idxs), sample)| {
                    let patch = partition.patches[patch_idx];
                    let solid_angle = patch.solid_angle();
                    let denom_rcp = math::rcp_f32(macro_area * solid_angle.as_f32());
                    let facets_area = if params.use_facet_area {
                        facet_idxs
                            .par_chunks(FACET_CHUNK_SIZE)
                            .map(|idxs| {
                                idxs.iter()
                                    .fold(0.0, |sum, idx| sum + mesh.facet_areas[*idx])
                            })
                            .sum::<f32>()
                    } else {
                        facet_idxs.len() as f32
                    };
                    *sample = facets_area * denom_rcp;
                });

            Some(Measurement {
                name: surf.unwrap().file_stem().unwrap().to_owned(),
                source: MeasurementSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: Box::new(MeasuredNdfData {
                    params,
                    samples: samples.clone().into_boxed_slice(),
                }),
            })
        })
        .collect()
}

/// Calculates the surface area of a spherical cap.
///
/// <https://en.wikipedia.org/wiki/Spherical_cap>
pub fn surface_area_of_spherical_cap(zenith: Radians, radius: f32) -> f32 {
    2.0 * std::f32::consts::PI * radius * radius * (1.0 - zenith.cos())
}

/// Classifies the zenith angle of a microfacet normal into a bin index.
/// The zenith angle is measured from the top of the hemisphere. The center of
/// the zenith bin is at the zenith angle calculated from the zenith range.
///
///
/// # Returns
///
/// The indices of the bins that the zenith angle falls into.
fn classify_normal_by_zenith(
    zenith: Radians,
    zenith_range: StepRangeIncl<Radians>,
    bin_width_scale: f32,
) -> [u8; 2] {
    let mut indices = [0xFF; 2];
    let mut i = 0;
    let half_bin_width = zenith_range.step_size * 0.5 * bin_width_scale;
    for (j, bin_center) in zenith_range.values().enumerate() {
        if (bin_center - half_bin_width..=bin_center + half_bin_width).contains(&zenith) {
            indices[i] = j as u8;
            i += 1;
        }
        if i >= 2 {
            break;
        }
    }
    indices
}

#[test]
fn test_normal_classification_by_zenith() {
    use vgn_core::units::deg;
    let range = StepRangeIncl::new(Radians::ZERO, Radians::HALF_PI, deg!(30.0).to_radians());
    assert_eq!(
        classify_normal_by_zenith(deg!(0.0).to_radians(), range, 1.0),
        [0, 0xff]
    );
    assert_eq!(
        classify_normal_by_zenith(deg!(7.5).to_radians(), range, 1.0),
        [0, 0xff]
    );
    assert_eq!(
        classify_normal_by_zenith(deg!(15.0).to_radians(), range, 1.0),
        [0, 1]
    );
    assert_eq!(
        classify_normal_by_zenith(deg!(22.5).to_radians(), range, 1.0),
        [1, 0xff]
    );
    assert_eq!(
        classify_normal_by_zenith(deg!(30.0).to_radians(), range, 1.0),
        [1, 0xff]
    );
    assert_eq!(
        classify_normal_by_zenith(deg!(37.5).to_radians(), range, 1.0),
        [1, 0xff]
    );
    assert_eq!(
        classify_normal_by_zenith(deg!(45.0).to_radians(), range, 1.0),
        [1, 2]
    );
}

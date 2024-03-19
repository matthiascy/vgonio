use crate::{
    app::cache::{Handle, RawCache},
    measure::{
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::{AdfMeasurementMode, AdfMeasurementParams},
    },
    partition::SphericalPartition,
    SphericalDomain,
};
use base::{
    error::VgonioError,
    math,
    math::{Sph2, Vec3Swizzles},
    range::RangeByStepSizeInclusive,
    units,
    units::{rad, Radians},
};
use std::{borrow::Cow, path::Path};
use surf::{MicroSurface, MicroSurfaceMesh};

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
pub struct MeasuredAdfData {
    /// The measurement parameters.
    pub params: AdfMeasurementParams,
    /// The distribution data. The outermost index is the azimuthal angle of the
    /// microfacet normal, and the inner index is the zenith angle of the
    /// microfacet normal.
    pub samples: Box<[f32]>,
}

impl MeasuredAdfData {
    /// Writes the measured data as an EXR file.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let (w, h) = (resolution, resolution);
        let partition = SphericalPartition::new(
            self.params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            self.params.mode.partition_precision_for_data_collection(),
        );

        // Collect the data following the patches.
        let mut patch_data = vec![0.0; partition.num_patches()];
        match self.params.mode {
            AdfMeasurementMode::ByPoints { zenith, azimuth } => {
                assert!(
                    zenith.step_size > rad!(0.0) && azimuth.step_size > rad!(0.0),
                    "The step size of zenith and azimuth must be greater than 0."
                );
                let n_theta = RangeByStepSizeInclusive::zero_to_half_pi(zenith.step_size)
                    .step_count_wrapped();
                let n_phi =
                    RangeByStepSizeInclusive::zero_to_tau(azimuth.step_size).step_count_wrapped();
                // ADF samples in ByPoints mode is stored by azimuth first, then by zenith.
                // We need to rearrange the data to match the patch order, which is by zenith
                // first, then by azimuth.
                patch_data.iter_mut().enumerate().for_each(|(i_p, v)| {
                    let i_theta = i_p / n_phi;
                    let i_phi = i_p % n_phi;
                    let i_adf = i_phi * n_theta + i_theta;
                    // In case the number of samples is less than the number of patches.
                    if i_adf < self.samples.len() {
                        *v = self.samples[i_adf];
                    }
                });
            }
            AdfMeasurementMode::ByPartition { .. } => {
                patch_data.copy_from_slice(&self.samples);
            }
        }

        // Calculate the patch index for each pixel.
        let mut indices = vec![0i32; (w * h) as usize];
        partition.compute_pixel_patch_indices(w, h, &mut indices);

        let mut samples = vec![0.0f32; (w * h) as usize];
        // Collect the data into the buffer.
        for i in 0..w {
            // x, width, column
            for j in 0..h {
                // y, height, row
                let idx = i + j * w;
                let patch_idx = indices[idx as usize];
                if patch_idx >= 0 {
                    samples[idx as usize] = patch_data[patch_idx as usize];
                }
            }
        }

        let layer = Layer::new(
            (w as usize, h as usize),
            LayerAttributes {
                layer_name: Some(Text::from("NDF")),
                capture_date: Text::new_or_none(base::utils::iso_timestamp_from_datetime(
                    timestamp,
                )),
                ..LayerAttributes::default()
            },
            Encoding::FAST_LOSSLESS,
            AnyChannels {
                list: SmallVec::from_vec(vec![AnyChannel::new(
                    "NDF",
                    FlatSamples::F32(Cow::Borrowed(&samples)),
                )]),
            },
        );
        let image = Image::from_layer(layer);
        image
            .write()
            .to_file(filepath)
            .map_err(|err| VgonioError::new("Failed to write NDF EXR file.", Some(Box::new(err))))
    }
}

/// Size of the chunk of facets to process in parallel.
const FACET_CHUNK_SIZE: usize = 4096;

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_area_distribution(
    params: AdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    cache: &RawCache,
) -> Box<[MeasurementData]> {
    #[cfg(feature = "bench")]
    let start = std::time::Instant::now();
    log::info!("Measuring microfacet area distribution...");

    let surfs = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = handles.iter().zip(surfs.iter()).zip(meshes.iter());

    let measurements = match params.mode {
        AdfMeasurementMode::ByPoints { .. } => {
            measure_area_distribution_by_points(surfaces, params)
        }
        AdfMeasurementMode::ByPartition { .. } => {
            measure_area_distribution_by_partition(surfaces, params)
        }
    };

    #[cfg(feature = "bench")]
    {
        let elapsed = start.elapsed();
        log::info!("ADF measurement took {} ms.", elapsed.as_millis());
    }
    measurements.into_boxed_slice()
}

/// Measure the microfacet area distribution function by sampling the hemisphere
/// with equal angle steps.
fn measure_area_distribution_by_points<'a>(
    surfaces: impl Iterator<
        Item = (
            (&'a Handle<MicroSurface>, &'a Option<&'a MicroSurface>),
            &'a Option<&'a MicroSurfaceMesh>,
        ),
    >,
    params: AdfMeasurementParams,
) -> Vec<MeasurementData> {
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

            // Sort the facets into bins according to their normal direction's zenith angle.
            let mut facets_bins = vec![vec![]; zenith.step_count_wrapped()];
            let macro_area = if !params.crop_to_disk {
                for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                    let zen = math::theta(normal);
                    let idxs = classify_normal_by_zenith(zen, zenith, 1.2);
                    for idx in idxs {
                        if idx == 0xFF {
                            continue;
                        }
                        facets_bins[idx as usize].push(facet_idx);
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
                    let zen = math::theta(normal);
                    for idx in classify_normal_by_zenith(zen, zenith, 1.2) {
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

            // TODO: allocate exactly the needed size for the samples. Boxed slice
            let mut samples =
                vec![0.0f32; azimuth.step_count_wrapped() * zenith.step_count_wrapped()];
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

            Some(MeasurementData {
                name: surface.unwrap().file_stem().unwrap().to_owned(),
                source: MeasurementDataSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: MeasuredData::Adf(MeasuredAdfData {
                    params,
                    samples: samples.into_boxed_slice(),
                }),
            })
        })
        .collect()
}

/// Measure the microfacet area distribution function by partitioning the
/// hemisphere into patches and calculating the corresponding values.
fn measure_area_distribution_by_partition<'a>(
    surfaces: impl Iterator<
        Item = (
            (&'a Handle<MicroSurface>, &'a Option<&'a MicroSurface>),
            &'a Option<&'a MicroSurfaceMesh>,
        ),
    >,
    params: AdfMeasurementParams,
) -> Vec<MeasurementData> {
    use rayon::prelude::*;
    let (precision, scheme) = params.mode.as_mode_by_partition().unwrap();
    let partition = SphericalPartition::new(scheme, SphericalDomain::Upper, precision);
    log::info!(
        "  -- Partitioning the hemisphere into {} patches.",
        partition.num_patches()
    );
    // TODO: Boxed slice
    // Data buffer for data of each patch.
    let mut samples = vec![0.0; partition.num_patches()];
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

            let mut normals_per_patch = vec![vec![]; partition.num_patches()];

            let macro_area = if !params.crop_to_disk {
                for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                    match partition.contains(Sph2::from_cartesian(*normal)) {
                        None => {
                            log::warn!("Facet normal {} is not contained in any patch.", normal);
                        }
                        Some(patch_idx) => {
                            normals_per_patch[patch_idx].push(facet_idx);
                        }
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
                        }
                        Some(patch_idx) => {
                            normals_per_patch[patch_idx].push(facet_idx);
                        }
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

            Some(MeasurementData {
                name: surf.unwrap().file_stem().unwrap().to_owned(),
                source: MeasurementDataSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: MeasuredData::Adf(MeasuredAdfData {
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
    zenith_range: RangeByStepSizeInclusive<Radians>,
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
    use base::units::deg;
    let range =
        RangeByStepSizeInclusive::new(Radians::ZERO, Radians::HALF_PI, deg!(30.0).to_radians());
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

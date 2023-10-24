use crate::{
    app::cache::{Handle, InnerCache},
    measure::{
        bsdf::receiver::{PartitionScheme, ReceiverParams},
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::AdfMeasurementParams,
    },
    RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalDomain,
};
use std::{borrow::Cow, path::Path};
use vgcore::{
    error::VgonioError,
    math,
    math::{Sph2, Vec3},
    units,
    units::{rad, Radians},
};
use vgsurf::MicroSurface;

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
    pub samples: Vec<f32>,
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
        let (w, h) = (resolution as usize, resolution as usize);
        let receiver = ReceiverParams {
            domain: SphericalDomain::Upper,
            precision: Sph2::new(self.params.zenith.step_size, self.params.azimuth.step_size),
            scheme: PartitionScheme::EqualAngle,
            retrieval: Default::default(),
        };
        let partition = receiver.partitioning();
        // Collect the data following the patches.
        let n_theta = self.params.zenith.step_count();
        let n_phi = self.params.azimuth.step_count_wrapped();
        debug_assert_eq!(
            n_theta * n_phi,
            self.samples.len(),
            "The number of samples does not match the number of patches."
        );
        let mut patch_data = vec![0.0; partition.num_patches()];
        patch_data.iter_mut().enumerate().for_each(|(i_p, v)| {
            let i_theta = i_p / n_phi;
            let i_phi = i_p % n_phi;
            let i_adf = i_phi * n_theta + i_theta;
            *v = self.samples[i_adf];
        });
        let half_phi = self.params.azimuth.step_size * 0.5;
        // Calculate the patch index for each pixel.
        let mut patch_indices = vec![0; w * h];
        for i in 0..w {
            // x, width, column
            for j in 0..h {
                // y, height, row
                let x = ((2 * i) as f32 / w as f32 - 1.0) * std::f32::consts::SQRT_2; // Flip the y-axis to match the BSDF coordinate system.
                let y = -((2 * j) as f32 / h as f32 - 1.0) * std::f32::consts::SQRT_2;
                let r_disc = (x * x + y * y).sqrt();
                let theta = 2.0 * (r_disc / 2.0).asin();
                let phi = (rad!((y).atan2(x)) + half_phi).wrap_to_tau();
                patch_indices[i + j * w] = match partition.contains(Sph2::new(rad!(theta), phi)) {
                    None => -1,
                    Some(idx) => idx as i32,
                }
            }
        }
        let mut samples = vec![0.0; w * h];
        for i in 0..w {
            for j in 0..h {
                let idx = patch_indices[i + j * w];
                if idx < 0 {
                    continue;
                }
                samples[i + j * w] = patch_data[idx as usize];
            }
        }
        let layer = Layer::new(
            (w, h),
            LayerAttributes {
                layer_name: Some(Text::from("NDF")),
                capture_date: Text::new_or_none(&vgcore::utils::iso_timestamp_from_datetime(
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

#[rustfmt::skip]
/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_area_distribution(
    params: AdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    cache: &InnerCache,
) -> Vec<MeasurementData> {
    #[cfg(feature = "bench")]
    let start = std::time::Instant::now();

    use rayon::prelude::*;
    log::info!("Measuring microfacet area distribution...");
    let surfaces = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);

    let measurements = handles
        .iter()
        .zip(surfaces.iter())
        .zip(meshes.iter())
        .filter_map(|((hdl, surface), mesh)| {
            if surface.is_none() || mesh.is_none() {
                log::debug!("Skipping a surface because it is not loaded {:?}.", mesh);
                return None;
            }
            let mesh = mesh.unwrap();
            let macro_area = mesh.macro_surface_area();
            let half_zenith_bin_width = params.zenith.step_size * 0.5;
            let half_zenith_bin_width_cos = half_zenith_bin_width.cos();

            const FACET_CHUNK_SIZE: usize = 4096;
            let solid_angle = units::solid_angle_of_spherical_cap(params.zenith.step_size).value();
            let denom_rcp = math::rcp_f32(macro_area * solid_angle);
            log::info!(
                "-- Measuring the NDF of surface: {}",
                surface.unwrap().file_stem().unwrap()
            );
            log::debug!("  -- macro surface area (mesh): {}", macro_area);
            log::debug!("  -- macro surface area: {}", surface.unwrap().macro_area());
            log::debug!("  -- micro facet total area: {}", mesh.facet_total_area);
            log::debug!("  -- micro facet count: {}", mesh.facet_normals.len());
            log::debug!("  -- solid angle: {}", solid_angle);
            log::debug!("  -- denom_rcp: {}", denom_rcp);

            // Sort the facets into bins according to their normal direction's zenith angle.
            let mut facets_bins = vec![vec![]; params.zenith.step_count_wrapped()];
            for (facet_idx, normal) in mesh.facet_normals.iter().enumerate() {
                let zenith = math::theta_of(normal);
                let idxs = classify_normal_by_zenith(zenith, params.zenith, 1.2);
                for idx in idxs {
                    if idx == 0xFF {
                        continue;
                    }
                    facets_bins[idx as usize].push(facet_idx);
                }
            }
            let mut samples = vec![
                0.0f32;
                params.azimuth.step_count_wrapped()
                    * params.zenith.step_count_wrapped()
            ];
            for azi_idx in 0..params.azimuth.step_count_wrapped() {
                for zen_idx in 0..params.zenith.step_count_wrapped() {
                    let azimuth = azi_idx as f32 * params.azimuth.step_size;
                    let zenith = zen_idx as f32 * params.zenith.step_size;
                    let dir = math::spherical_to_cartesian(1.0, zenith, azimuth).normalize();
                    let facets = &facets_bins[zen_idx];
                    let facets_area = facets
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
                        .sum::<f32>();
                    let sample_idx = azi_idx * params.zenith.step_count_wrapped() + zen_idx;
                    samples[sample_idx] = facets_area * denom_rcp;
                    log::trace!(
                        "-- φ: {}, θ: {}  | facet area: {} => {}",
                        azimuth.prettified(),
                        zenith.prettified(),
                        facets_area,
                        samples[sample_idx]
                    );
                }
            }
            
            Some(MeasurementData {
                name: surface.unwrap().file_stem().unwrap().to_owned(),
                source: MeasurementDataSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: MeasuredData::Adf(MeasuredAdfData { params, samples }),
            })
        })
        .collect();
    #[cfg(feature = "bench")]
    {
        let elapsed = start.elapsed();
        log::info!("ADF measurement took {} ms.", elapsed.as_millis());
    }
    measurements
}

/// Calculates the surface area of a spherical cap.
///
/// https://en.wikipedia.org/wiki/Spherical_cap
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
    use vgcore::units::deg;
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

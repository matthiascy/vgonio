use crate::{
    app::cache::{Handle, InnerCache},
    measure::{
        bsdf::receiver::{PartitionScheme, ReceiverParams},
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::AdfMeasurementParams,
    },
    SphericalDomain,
};
use exr::prelude::WritableImage;
use std::path::Path;
use vgcore::{
    error::VgonioError,
    math,
    math::Sph2,
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
            retrieval_mode: Default::default(),
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
                list: SmallVec::from_vec(vec![AnyChannel::new("NDF", FlatSamples::F32(samples))]),
            },
        );
        let image = Image::from_layer(layer);
        image
            .write()
            .to_file(filepath)
            .map_err(|err| VgonioError::new("Failed to write NDF EXR file.", Some(Box::new(err))))
    }
}

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_area_distribution(
    params: AdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    cache: &InnerCache,
) -> Vec<MeasurementData> {
    use rayon::prelude::*;
    log::info!("Measuring microfacet area distribution...");
    let surfaces = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    handles
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
            let half_zenith_bin_size_cos = (params.zenith.step_size / 2.0).cos();

            log::info!(
                "-- Measuring the NDF of surface: {}",
                surface.unwrap().file_stem().unwrap()
            );
            log::debug!("  -- macro surface area (mesh): {}", macro_area);
            log::debug!("  -- macro surface area: {}", surface.unwrap().macro_area());
            log::debug!("  -- micro facet total area: {}", mesh.facet_total_area);

            let samples = {
                let solid_angle =
                    units::solid_angle_of_spherical_cap(params.zenith.step_size).value();
                let denominator = macro_area * solid_angle;
                (0..params.azimuth.step_count_wrapped())
                    .flat_map(move |azimuth_idx| {
                        // NOTE: the zenith angle is measured from the top of the
                        // hemisphere. The center of the zenith/azimuth bin are at the
                        // zenith/azimuth angle calculated below.
                        (0..params.zenith.step_count_wrapped()).map(move |zenith_idx| {
                            let azimuth = azimuth_idx as f32 * params.azimuth.step_size;
                            let zenith = zenith_idx as f32 * params.zenith.step_size;
                            let dir =
                                math::spherical_to_cartesian(1.0, zenith, azimuth).normalize();
                            let facets_surface_area = mesh
                                .facet_normals
                                .par_iter()
                                .enumerate()
                                .filter_map(|(idx, normal)| {
                                    if normal.dot(dir) >= half_zenith_bin_size_cos {
                                        Some(idx)
                                    } else {
                                        None
                                    }
                                })
                                .fold(|| 0.0, |area, facet| area + mesh.facet_surface_area(facet))
                                .reduce(|| 0.0, |a, b| a + b);
                            let value = facets_surface_area / denominator;
                            log::trace!(
                                "-- azimuth: {}, zenith: {}  | facet area: {} => {}",
                                azimuth.prettified(),
                                zenith.prettified(),
                                facets_surface_area,
                                value
                            );
                            value
                        })
                    })
                    .collect::<Vec<_>>()
            };
            Some(MeasurementData {
                name: surface.unwrap().file_stem().unwrap().to_owned(),
                source: MeasurementDataSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: MeasuredData::Adf(MeasuredAdfData { params, samples }),
            })
        })
        .collect()
}

/// Calculates the surface area of a spherical cap.
///
/// https://en.wikipedia.org/wiki/Spherical_cap
pub fn surface_area_of_spherical_cap(zenith: Radians, radius: f32) -> f32 {
    2.0 * std::f32::consts::PI * radius * radius * (1.0 - zenith.cos())
}

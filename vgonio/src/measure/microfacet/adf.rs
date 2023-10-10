use std::path::Path;
use vgcore::{error::VgonioError, math, units, units::Radians};
use vgsurf::MicroSurface;
// NOTE(yang): The number of bins is determined by the bin size and the range of
// the azimuth and zenith angles. How do we decide the size of the bins (solid
// angle)? How do we arrange each bin on top of the hemisphere? Circle packing?
use crate::{
    app::cache::{Handle, InnerCache},
    measure::{
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::AdfMeasurementParams,
    },
};

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
    // TODO: configurable output width and height.
    /// Writes the measured data as an EXR file.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
    ) -> Result<(), VgonioError> {
        // Generate equal angle partitioned hemisphere.1
        // Calculate the patch index for each pixel.
        // Write the data to the EXR file.
        todo!("write_as_exr")
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

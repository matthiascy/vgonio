// NOTE(yang): The number of bins is determined by the bin size and the range of
// the azimuth and zenith angles. How do we decide the size of the bins (solid
// angle)? How do we arrange each bin on top of the hemisphere? Circle packing?

use std::{io::BufWriter, path::Path};

use crate::{
    app::cache::{Cache, Handle},
    error::Error,
    io::{vgmo, CompressionScheme, FileEncoding, WriteFileError},
    math,
    measure::measurement::{MadfMeasurementParams, MeasurementKind},
    msurf::MicroSurface,
    units::{self, Radians},
    Handedness,
};

/// Structure holding the data for microfacet area distribution measurement.
///
/// D(m) is the micro-facet area (normal) distribution function, which gives the
/// relative number of facets oriented in any given direction, or, more
/// precisely, the relative total facet surface area per unit solid angle of
/// surface normals pointed in any given direction.
#[derive(Debug, Clone)]
pub struct MicrofacetAreaDistribution {
    /// The measurement parameters.
    pub params: MadfMeasurementParams,
    /// The distribution data. The first index is the azimuthal angle, and the
    /// second index is the zenith angle.
    pub samples: Vec<f32>,
}

impl MicrofacetAreaDistribution {
    /// Save the microfacet distribution to a file
    pub fn write_to_file(
        &self,
        filepath: &Path,
        encoding: FileEncoding,
        compression: CompressionScheme,
    ) -> Result<(), Error> {
        assert_eq!(
            self.samples.len(),
            self.params.azimuth.step_count_wrapped() * self.params.zenith.step_count_wrapped(),
            "The number of samples does not match the number of bins."
        );
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filepath)?;
        let header = vgmo::Header::Madf {
            meta: vgmo::HeaderMeta {
                kind: MeasurementKind::MicrofacetAreaDistribution,
                encoding,
                compression,
            },
            // azimuth_range: AngleRange {
            //     start: self.azimuth_start.value,
            //     end: self.azimuth_stop.value,
            //     bin_count: self.azimuth_bins_count_inclusive as u32,
            //     bin_width: self.azimuth_bin_width.value,
            // },
            // zenith_range: AngleRange {
            //     start: self.zenith_start.value,
            //     end: self.zenith_stop.value,
            //     bin_count: self.zenith_bins_count_inclusive as u32,
            //     bin_width: self.zenith_bin_width.value,
            // },
            // sample_count: self.samples.len() as u32,
            madf: self.params,
        };
        let mut writer = BufWriter::new(file);
        vgmo::write(&mut writer, header, &self.samples).map_err(|err| {
            Error::WriteFile(WriteFileError {
                path: filepath.to_path_buf().into_boxed_path(),
                kind: err,
            })
        })
    }
}

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_area_distribution(
    params: MadfMeasurementParams,
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
) -> Vec<MicrofacetAreaDistribution> {
    use rayon::prelude::*;
    log::info!("Measuring microfacet area distribution...");
    let surfaces = cache.get_micro_surface_meshes_by_surfaces(surfaces);
    surfaces
        .iter()
        .filter_map(|surface| {
            if surface.is_none() {
                log::debug!("Skipping a surface because it is not loaded {:?}.", surface);
                return None;
            }
            let surface = surface.as_ref().unwrap();
            let macro_area = surface.macro_surface_area();
            let solid_angle = units::solid_angle_of_spherical_cap(params.zenith.step_size).value();
            let divisor = macro_area * solid_angle;
            let half_zenith_bin_size_cos = (params.zenith.step_size / 2.0).cos();
            log::debug!("-- macro surface area: {}", macro_area);
            log::debug!("-- solid angle per measurement: {}", solid_angle);
            let samples = (0..params.azimuth.step_count_wrapped())
                .flat_map(move |azimuth_idx| {
                    // NOTE: the zenith angle is measured from the top of the
                    // hemisphere. The center of the zenith/azimuth bin are at the zenith/azimuth
                    // angle calculated below.
                    (0..params.zenith.step_count_wrapped()).map(move |zenith_idx| {
                        let azimuth = azimuth_idx as f32 * params.azimuth.step_size;
                        let zenith = zenith_idx as f32 * params.zenith.step_size;
                        let dir = math::spherical_to_cartesian(
                            1.0,
                            zenith,
                            azimuth,
                            Handedness::RightHandedYUp,
                        )
                        .normalize();
                        let facets_surface_area = surface
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
                            .fold(
                                || 0.0,
                                |area, facet| area + surface.facet_surface_area(facet),
                            )
                            .reduce(|| 0.0, |a, b| a + b);
                        let value = facets_surface_area / divisor;
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
                .collect::<Vec<_>>();
            Some(MicrofacetAreaDistribution { params, samples })
        })
        .collect()
}

/// Calculates the surface area of a spherical cap.
///
/// https://en.wikipedia.org/wiki/Spherical_cap
pub fn surface_area_of_spherical_cap(zenith: Radians, radius: f32) -> f32 {
    2.0 * std::f32::consts::PI * radius * radius * (1.0 - zenith.cos())
}

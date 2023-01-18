use std::{
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    app::cache::{Cache, MicroSurfaceHandle},
    error::Error,
    units::{self, Radians},
};

use super::{
    measurement::{MicrofacetDistributionMeasurement, MicrofacetShadowingMaskingMeasurement},
    spherical_to_cartesian, Handedness,
};

use rayon::prelude::*;

// NOTE(yang): The number of bins is determined by the bin size and the range of
// the azimuth and zenith angles. How do we decide the size of the bins (solid
// angle)? How do we arrange each bin on top of the hemisphere?
//
// TODO(yang): let user decide the bin size

/// Strcture holding the data for micro facet distribution measurement.
///
/// D(m) is the micro facet distribution function, which gives the relative
/// number of facets oriented in any given direction, or, more precisely, the
/// relative total facet surface area per unit solid angle of surface normals
/// pointed in any given direction.
#[derive(Debug, Clone)]
pub struct MicrofacetDistribution {
    /// The bin size of azimuthal angle when sampling the microfacet
    /// distribution.
    pub azimuth_bin_size: Radians,
    /// The bin size of zenith angle when sampling the microfacet
    /// distribution.
    pub zenith_bin_size: Radians,
    /// The number of bins in the azimuthal angle.
    pub azimuth_bins_count: usize,
    /// The number of bins in the zenith angle.
    pub zenith_bins_count: usize,
    /// The distribution data. The first index is the azimuthal angle, and the
    /// second index is the zenith angle.
    pub samples: Vec<f32>,
}

impl MicrofacetDistribution {
    /// Save the microfacet distribution to a file in ascii format.
    pub fn save_ascii(&self, filepath: &Path) -> Result<(), Error> {
        log::info!(
            "Saving microfacet distribution in ascii format to {}",
            filepath.display()
        );
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(filepath)?;
        let mut writter = BufWriter::new(file);
        let header = format!(
            "microfacet distribution\nazimuth - bin size: {}, bins count: {}\nzenith - bin size: \
             {}, bins count: {}\n",
            self.azimuth_bin_size.in_degrees().prettified(),
            self.azimuth_bins_count,
            self.zenith_bin_size.in_degrees().prettified(),
            self.zenith_bins_count
        );
        let _ = writter.write(header.as_bytes())?;
        self.samples.iter().for_each(|s| {
            let value = format!("{s} ");
            let _ = writter.write(value.as_bytes()).unwrap();
        });
        Ok(())
    }
}

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_microfacet_distribution(
    desc: MicrofacetDistributionMeasurement,
    surfaces: &[MicroSurfaceHandle],
    cache: &Cache,
) -> Vec<MicrofacetDistribution> {
    log::info!("Measuring micro facet distribution...");
    let surfaces = cache.get_micro_surface_meshes(surfaces);
    surfaces
        .iter()
        .map(|surface| {
            let macro_area = surface.macro_surface_area();
            let solid_angle = units::solid_angle_of_spherical_cap(desc.zenith.step_size).value();
            let divisor = macro_area * solid_angle;
            let half_zenith_bin_size_cos = (desc.zenith.step_size / 2.0).cos();
            log::debug!("-- macro surface area: {}", macro_area);
            log::debug!("-- solid angle per measurement: {}", solid_angle);
            let samples = (0..desc.azimuth.step_count())
                .flat_map(move |azimuth_idx| {
                    // NOTE: the zenith angle is measured from the top of the
                    // hemisphere. The center of the zenith/azimuth bin are at the zenith/azimuth
                    // angle calculated below.
                    (0..desc.zenith.step_count() + 1).map(move |zenith_idx| {
                        let azimuth = azimuth_idx as f32 * desc.azimuth.step_size;
                        let zenith = zenith_idx as f32 * desc.zenith.step_size;
                        let dir = spherical_to_cartesian(
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
            MicrofacetDistribution {
                azimuth_bin_size: desc.azimuth.step_size,
                zenith_bin_size: desc.zenith.step_size,
                azimuth_bins_count: desc.azimuth.step_count(),
                zenith_bins_count: desc.zenith.step_count() + 1,
                samples,
            }
        })
        .collect()
}

/// Calculates the surface area of a spherical cap.
///
/// https://en.wikipedia.org/wiki/Spherical_cap
pub fn surface_area_of_spherical_cap(zenith: Radians, radius: f32) -> f32 {
    2.0 * std::f32::consts::PI * radius * radius * (1.0 - zenith.cos())
}

/// Strcture holding the data for microfacet shadowing and masking measurement.
///
/// G(i, o, m) is the micro facet shadowing-masking function, which describes
/// the fraction of microfacets with normal m that are visible from both the
/// incident direction i and the outgoing direction o.
///
/// The Smith microfacet shadowing-masking function is defined as:
/// G(i, o, m) = G1(i, m) * G1(o, m)
pub struct MicrofacetShadowingMaskingFunction {
    /// The bin size of azimuthal angle when sampling the microfacet
    /// distribution.
    pub azimuth_bin_size: Radians,
    /// The bin size of zenith angle when sampling the microfacet
    /// distribution.
    pub zenith_bin_size: Radians,
    /// The number of bins in the azimuthal angle.
    pub azimuth_bins_count: usize,
    /// The number of bins in the zenith angle.
    pub zenith_bins_count: usize,
    /// The distribution data. The first index is the azimuthal angle, and the
    /// second index is the zenith angle.
    pub samples: Vec<f32>,
}

impl MicrofacetShadowingMaskingFunction {
    /// Saves the microfacet shadowing and masking function in ascii format.
    pub fn save_ascii(&self, filepath: &Path) -> Result<(), std::io::Error> {
        log::info!(
            "Saving microfacet shadowing and masking distribution in ascii format to {}",
            filepath.display()
        );
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(filepath)?;
        let mut writter = BufWriter::new(file);
        let header = format!(
            "microfacet shadowing and masking function\nazimuth - bin size: {}, bins count: \
             {}\nzenith - bin size: {}, bins count: {}\n",
            self.azimuth_bin_size.in_degrees().prettified(),
            self.azimuth_bins_count,
            self.zenith_bin_size.in_degrees().prettified(),
            self.zenith_bins_count
        );
        let _ = writter.write(header.as_bytes())?;
        self.samples.iter().for_each(|s| {
            let value = format!("{s} ");
            let _ = writter.write(value.as_bytes()).unwrap();
        });
        Ok(())
    }
}

pub fn measure_microfacet_shadowing_masking(
    desc: MicrofacetShadowingMaskingMeasurement,
    surfaces: &[MicroSurfaceHandle],
    cache: &Cache,
) -> Vec<MicrofacetShadowingMaskingFunction> {
    // TODO(yang):
    vec![]
}

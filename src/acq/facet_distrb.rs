use crate::{
    app::cache::{Cache, MicroSurfaceHandle},
    units::{self, degrees, Degrees, Radians},
};

use super::{spherical_to_cartesian, Handedness};

pub const AZIMUTH_BIN_SIZE: Degrees = degrees!(5.0);
pub const ZENITH_BIN_SIZE: Degrees = degrees!(2.0);
pub const AZIMUTH_BIN_SIZE_HALF: Degrees = degrees!(2.5);
pub const ZENITH_BIN_SIZE_HALF: Degrees = degrees!(1.0);
pub const AZIMUTH_BIN_COUNT: usize = 72;
pub const ZENITH_BIN_COUNT: usize = 17;

// NOTE(yang): The number of bins is determined by the bin size and the range of
// the azimuth and zenith angles. How do we decide the size of the bins (solid
// angle)? How do we arrange each bin on top of the hemisphere?
//
// TODO(yang): let user decide the bin size

/// Strcture holding the data for micro facet distribution.
///
/// D(m) is the micro facet distribution function, which gives the relative
/// number of facets oriented in any given direction, or, more precisely, the
/// relative total facet surface area per unit solid angle of surface normals
/// pointed in any given direction.
pub struct MicrofacetDistribution {
    /// The bin size of azimuthal angle when sampling the microfacet
    /// distribution.
    pub azimuth_bin_size: Degrees,
    /// The bin size of zenith angle when sampling the microfacet
    /// distribution.
    pub zenith_bin_size: Degrees,
    /// The number of bins in the azimuthal angle.
    pub azimuth_bins_count: usize,
    /// The number of bins in the zenith angle.
    pub zenith_bins_count: usize,
    /// The distribution data. The first index is the azimuthal angle, and the
    /// second index is the zenith angle.
    pub samples: Vec<f32>,
}

pub fn measure_micro_facet_distribution(
    surfaces: &[MicroSurfaceHandle],
    cache: &Cache,
) -> Vec<MicrofacetDistribution> {
    let surfaces = cache.get_micro_surface_meshes(surfaces);
    surfaces
        .iter()
        .map(|surface| {
            let macro_area = surface.macro_surface_area();
            let samples = (0..AZIMUTH_BIN_COUNT)
                .flat_map(move |azimuth_idx| {
                    (0..ZENITH_BIN_COUNT).map(move |zenith_idx| {
                        let azimuth = azimuth_idx as f32 * AZIMUTH_BIN_SIZE;
                        let zenith = zenith_idx as f32 * ZENITH_BIN_SIZE;
                        let dir = spherical_to_cartesian(
                            1.0,
                            zenith.in_radians(),
                            azimuth.in_radians(),
                            Handedness::RightHandedYUp,
                        )
                        .normalize();
                        let solid_angle =
                            units::solid_angle_of_spherical_cap(ZENITH_BIN_SIZE.in_radians());
                        let facets_surface_area = surface
                            .facet_normals
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, normal)| {
                                if normal.dot(dir) > 0.0 {
                                    Some(idx)
                                } else {
                                    None
                                }
                            })
                            .fold(0.0, |area, facet| area + surface.facet_surface_area(facet));
                        facets_surface_area / (solid_angle.value() * macro_area)
                    })
                })
                .collect::<Vec<_>>();
            MicrofacetDistribution {
                azimuth_bin_size: AZIMUTH_BIN_SIZE,
                zenith_bin_size: ZENITH_BIN_SIZE,
                azimuth_bins_count: AZIMUTH_BIN_COUNT,
                zenith_bins_count: ZENITH_BIN_COUNT,
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

use vgcore::{math, math::Handedness, units, units::Radians};
use vgsurf::MicroSurface;
// NOTE(yang): The number of bins is determined by the bin size and the range of
// the azimuth and zenith angles. How do we decide the size of the bins (solid
// angle)? How do we arrange each bin on top of the hemisphere? Circle packing?
use crate::{
    app::cache::{Cache, Handle},
    measure::measurement::{
        MadfMeasurementParams, MeasuredData, MeasurementData, MeasurementDataSource,
    },
    RangeByStepSizeInclusive,
};

/// Structure holding the data for microfacet area distribution measurement.
///
/// D(m) is the micro-facet area (normal) distribution function, which gives the
/// relative number of facets oriented in any given direction, or, more
/// precisely, the relative total facet surface area per unit solid angle of
/// surface normals pointed in any given direction.
#[derive(Debug, Clone)]
pub struct MeasuredMadfData {
    /// The measurement parameters.
    pub params: MadfMeasurementParams,
    /// The distribution data. The outermost index is the azimuthal angle of the
    /// microfacet normal, and the inner index is the zenith angle of the
    /// microfacet normal.
    pub samples: Vec<f32>,
}

impl MeasuredMadfData {
    /// Accumulate each slice (two opposite azimuthal angles) of the
    /// distribution data into a single slice.
    ///
    /// The output slice has the length of N * 2 - 1, where N is the number of
    /// zenith angle bins, minus 1 because the zenith angle of 0 is shared by
    /// the two slices. The first half of the output slice will be considered
    /// having negative zenith angles, and the second half positive zenith
    /// angles.
    pub fn accumulated_slice(&self) -> Vec<(f32, f32)> {
        let num_zenith_bins = self.params.zenith.step_count();
        let num_azimuth_bins = self.params.azimuth.step_count_wrapped();
        let half_num_azimuth_bins = num_azimuth_bins / 2;
        let mut accumulated = vec![(0.0, 0.0); num_zenith_bins * 2 - 1];
        let center_zenith_idx = (num_zenith_bins * 2 - 1) / 2;
        for (i, sample) in self.samples.chunks(num_zenith_bins).enumerate() {
            if i / half_num_azimuth_bins == 0 {
                // The second half of the output slice. The zenith angles are positive.
                for (j, s) in sample.iter().enumerate() {
                    accumulated[j + center_zenith_idx].1 += s;
                }
            } else {
                // The first half of the output slice. The zenith angles are negative.
                for (j, s) in sample.iter().rev().enumerate() {
                    accumulated[j].1 += s;
                }
            }
        }
        for (i, sample) in accumulated.iter_mut().enumerate() {
            let theta = i as f32 * self.params.zenith.step_size
                - (num_zenith_bins - 1) as f32 * self.params.zenith.step_size;
            sample.0 = theta.as_f32();
        }

        accumulated
    }
}

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_area_distribution(
    mut params: MadfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    cache: &Cache,
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

            let samples = if params.single_slice {
                params.azimuth =
                    RangeByStepSizeInclusive::new(Radians::new(0.0), Radians::TAU, Radians::PI);
                (0..2)
                    .flat_map(move |azimuth_idx| {
                        let condition = move |phi: f32| {
                            if azimuth_idx == 0 {
                                (phi >= 0.0 && phi < std::f32::consts::FRAC_PI_2)
                                    || (phi >= std::f32::consts::FRAC_PI_2 * 3.0
                                        && phi < std::f32::consts::TAU)
                            } else {
                                let diff = (phi - std::f32::consts::PI).abs();
                                diff >= std::f32::consts::FRAC_PI_2
                                    && diff < std::f32::consts::FRAC_PI_2 * 3.0
                            }
                        };
                        let azimuth = azimuth_idx as f32 * Radians::PI;
                        (0..params.zenith.step_count_wrapped()).map(move |zenith_idx| {
                            let zenith = zenith_idx as f32 * params.zenith.step_size;
                            let solid_angle = if zenith_idx == 0 {
                                units::solid_angle_of_spherical_cap(params.zenith.step_size)
                                    .as_f32()
                            } else {
                                units::solid_angle_of_spherical_strip(
                                    zenith + params.zenith.step_size / 2.0,
                                    zenith - params.zenith.step_size / 2.0,
                                )
                                .as_f32()
                            };
                            let facets_surface_area = mesh
                                .facet_normals
                                .par_iter()
                                .enumerate()
                                .filter_map(|(idx, normal)| {
                                    // Assume the right-handed y-up coordinate system.
                                    let phi = ((normal.z).atan2(normal.x) + std::f32::consts::TAU)
                                        % std::f32::consts::TAU;
                                    let theta_diff = ((normal.y).acos() - zenith.as_f32()).abs();
                                    if theta_diff <= half_zenith_bin_size_cos && condition(phi) {
                                        Some(idx)
                                    } else {
                                        None
                                    }
                                })
                                .fold(|| 0.0, |area, facet| area + mesh.facet_surface_area(facet))
                                .reduce(|| 0.0, |a, b| a + b);
                            facets_surface_area / (macro_area * solid_angle)
                        })
                    })
                    .collect::<Vec<_>>()
            } else {
                let solid_angle =
                    units::solid_angle_of_spherical_cap(params.zenith.step_size).value();
                let denominator = macro_area * solid_angle;
                log::debug!("-- macro surface area: {}", macro_area);
                log::debug!("-- solid angle per measurement: {}", solid_angle);
                (0..params.azimuth.step_count_wrapped())
                    .flat_map(move |azimuth_idx| {
                        // NOTE: the zenith angle is measured from the top of the
                        // hemisphere. The center of the zenith/azimuth bin are at the
                        // zenith/azimuth angle calculated below.
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
                measured: MeasuredData::Madf(MeasuredMadfData { params, samples }),
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

#[cfg(test)]
mod tests {
    use crate::{
        measure::{measurement::MadfMeasurementParams, microfacet::MeasuredMadfData},
        RangeByStepSizeInclusive,
    };
    use vgcore::units::{deg, rad, Radians};

    #[test]
    fn madf_accumulated_slice() {
        let params = MadfMeasurementParams {
            azimuth: RangeByStepSizeInclusive::new(rad!(0.0), Radians::TAU, Radians::HALF_PI),
            zenith: RangeByStepSizeInclusive::new(
                rad!(0.0),
                Radians::HALF_PI,
                deg!(30.0).in_radians(),
            ),
        };
        let data = MeasuredMadfData {
            params,
            samples: vec![
                1.0, 2.0, 3.0, 4.0, // phi = 0
                1.0, 7.0, 8.0, 9.0, // phi = 90
                1.0, 12.0, 13.0, 14.0, // phi = 180
                1.0, 17.0, 18.0, 19.0, // phi = 270
            ],
        };
        let accumulated = data.accumulated_slice();
        assert_eq!(accumulated.len(), 7);
        assert_eq!(accumulated[0].1, 33.0);
        assert_eq!(accumulated[1].1, 31.0);
        assert_eq!(accumulated[2].1, 29.0);
        assert_eq!(accumulated[3].1, 4.0);
        assert_eq!(accumulated[4].1, 9.0);
        assert_eq!(accumulated[5].1, 11.0);
        assert_eq!(accumulated[6].1, 13.0);
    }
}

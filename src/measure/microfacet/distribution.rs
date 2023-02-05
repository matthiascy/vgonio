// NOTE(yang): The number of bins is determined by the bin size and the range of
// the azimuth and zenith angles. How do we decide the size of the bins (solid
// angle)? How do we arrange each bin on top of the hemisphere? Circle packing?

use std::{
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    app::cache::{Cache, Handle},
    common::{DataEncoding, Handedness},
    error::Error,
    measure::{self, measurement::MicrofacetDistributionMeasurement},
    msurf::MicroSurface,
    units::{self, Radians},
};

/// Structure holding the data for micro facet distribution measurement.
///
/// D(m) is the micro facet distribution function, which gives the relative
/// number of facets oriented in any given direction, or, more precisely, the
/// relative total facet surface area per unit solid angle of surface normals
/// pointed in any given direction.
#[derive(Debug, Clone)]
pub struct MicrofacetDistribution {
    /// Start angle of the azimuth.
    pub azimuth_start: Radians,
    /// End angle of the azimuth.
    pub azimuth_stop: Radians,
    /// The bin size of azimuthal angle when sampling the microfacet
    /// distribution.
    pub azimuth_bin_size: Radians,
    /// Start angle of the zenith.
    pub zenith_start: Radians,
    /// End angle of the zenith.
    pub zenith_stop: Radians,
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
    /// Save the microfacet distribution to a file
    pub fn save(&self, filepath: &Path, encoding: DataEncoding) -> Result<(), Error> {
        assert_eq!(
            self.samples.len(),
            self.azimuth_bins_count * self.zenith_bins_count,
            "The number of samples does not match the number of bins."
        );
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filepath)?;
        let mut writter = BufWriter::new(file);
        match encoding {
            DataEncoding::Ascii => {
                log::info!("Saving microfacet distribution as plain text.");
                writter.write_all(b"VGMO\x01#")?; // 6 bytes
                writter.write_all(&self.azimuth_start.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.azimuth_stop.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.azimuth_bin_size.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&(self.azimuth_bins_count as u32).to_le_bytes())?; // 4 bytes
                writter.write_all(&self.zenith_start.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.zenith_stop.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.zenith_bin_size.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&(self.zenith_bins_count as u32).to_le_bytes())?; // 4 bytes
                writter.write_all(b"\x20\n")?; // 2 bytes
                self.samples.iter().enumerate().for_each(|(i, s)| {
                    let val = if i % self.zenith_bins_count == self.zenith_bins_count - 1 {
                        format!("{s}\n")
                    } else {
                        format!("{s} ")
                    };
                    writter.write_all(val.as_bytes()).unwrap();
                });
            }
            DataEncoding::Binary => {
                log::info!("Saving microfacet distribution as binary.");
                writter.write_all(b"VGMO\x01!")?; // 6 bytes
                writter.write_all(&self.azimuth_start.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.azimuth_stop.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.azimuth_bin_size.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&(self.azimuth_bins_count as u32).to_le_bytes())?; // 4 bytes
                writter.write_all(&self.zenith_start.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.zenith_stop.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&self.zenith_bin_size.value.to_le_bytes())?; // 4 bytes
                writter.write_all(&(self.zenith_bins_count as u32).to_le_bytes())?; // 4 bytes
                writter.write_all(b"\x20\n")?; // 2 bytes
                writter.write_all(
                    &self
                        .samples
                        .iter()
                        .map(|x| x.to_le_bytes())
                        .flatten()
                        .collect::<Vec<_>>(),
                )?;
            }
        }
        Ok(())
    }
}

/// Measure the microfacet distribution of a list of micro surfaces.
pub fn measure_microfacet_distribution(
    desc: MicrofacetDistributionMeasurement,
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
) -> Vec<MicrofacetDistribution> {
    use rayon::prelude::*;
    log::info!("Measuring micro facet distribution...");
    let surfaces = cache.micro_surface_meshes_by_surface_ids(surfaces);
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
                        let dir = measure::spherical_to_cartesian(
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
                azimuth_start: desc.azimuth.start,
                azimuth_stop: desc.azimuth.stop,
                zenith_start: desc.zenith.start,
                zenith_stop: desc.zenith.stop,
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

use crate::{
    app::cache::{Handle, InnerCache},
    measure::{
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::SdfMeasurementParams,
    },
    RangeByStepSizeInclusive,
};
use image::FlatSamples;
use std::{borrow::Cow, path::Path};
use vgcore::{
    error::VgonioError,
    math,
    math::{IVec2, UVec2, Vec2},
    units::{rad, Radians},
};
use vgsurf::MicroSurface;

/// Slope of the microfacet normal, i.e. the normal of the microfacet in the
/// tangent space or the slope space.
type Slope2 = Vec2;

/// Measured data of the slope distribution function (SDF).
#[derive(Debug, Clone)]
pub struct MeasuredSdfData {
    /// Parameters of the measurement.
    pub params: SdfMeasurementParams,
    /// Slopes of all microfacets.
    pub slopes: Vec<Slope2>,
}

/// Data of the slope distribution function (SDF) of a microsurface.
pub struct SdfPmf {
    /// The number of azimuth bins.
    pub azi_bin_count: usize,
    /// The number of zenith bins.
    pub zen_bin_count: usize,
    /// The width of each azimuth bin (in radians).
    pub azi_bin_width: Radians,
    /// The width of each zenith bin (in radians).
    pub zen_bin_width: Radians,
    /// The data of each bin, stored in the order of zenith first then azimuth,
    /// i.e. the zenith bins are stored in the inner loop, and the azimuth bins
    /// are stored in the outer loop.
    pub hist: Vec<f32>,
}

impl MeasuredSdfData {
    /// Computes the slope distribution function (SDF) of the measured data then
    /// writes it as an OpenEXR file.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let pixels_count = resolution * resolution;
        let mut pixels = vec![0.0; pixels_count as usize];
        let mut num_slopes_out_of_range = 0;
        for slope in &self.slopes {
            // Scale the slope to the range [-1, 1] by dividing it by the
            // maximum slope.
            let slope = *slope / self.params.max_slope;
            // Scale the slope to the range [0, 1] by multiplying it by 0.5 and
            // then adding 0.5.
            let slope = slope * 0.5 + 0.5;
            // Convert the slope value to a pixel coordinate.
            let bin: IVec2 = (slope * resolution as f32).as_ivec2();

            // There may be some slopes that are outside the range [0, 1] due to
            // the fact that the maximum slope is not always the largest slope.
            if bin.x < 0 || bin.x >= resolution as i32 {
                num_slopes_out_of_range += 1;
                continue;
            }
            if bin.y < 0 || bin.y >= resolution as i32 {
                num_slopes_out_of_range += 1;
                continue;
            }
            // Compute the pixel index.
            let index = bin.y as u32 * resolution + bin.x as u32;
            pixels[index as usize] += 1.0;
        }
        log::info!(
            "Percentage of slopes out of range: %{}",
            (num_slopes_out_of_range as f32 / self.slopes.len() as f32) * 100.0
        );
        // Normalize the pixels.
        pixels.iter_mut().for_each(|v| *v /= pixels_count as f32);
        // Write the pixels as an OpenEXR file.
        let layer = Layer::new(
            (resolution as usize, resolution as usize),
            LayerAttributes {
                layer_name: Some(Text::from("SDF")),
                capture_date: Text::new_or_none(&vgcore::utils::iso_timestamp_from_datetime(
                    timestamp,
                )),
                ..LayerAttributes::default()
            },
            Encoding::FAST_LOSSLESS,
            AnyChannels {
                list: SmallVec::from_vec(vec![AnyChannel::new(
                    "SDF",
                    FlatSamples::F32(Cow::Borrowed(&pixels)),
                )]),
            },
        );
        let image = Image::from_layer(layer);
        image
            .write()
            .to_file(filepath)
            .map_err(|err| VgonioError::new("Failed to write SDF EXR file.", Some(Box::new(err))))
    }

    /// Returns the histogram of the slope (PMF of the slope distribution).
    ///
    /// # Arguments
    ///
    /// * `azi_bin_width` - The width of each azimuth bin (in angles).
    /// * `zen_bin_width` - The width of each zenith bin (in angles).
    pub fn pmf(&self, azi_bin_width: Radians, zen_bin_width: Radians) -> SdfPmf {
        let azi_range = RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TAU, azi_bin_width);
        let zen_range =
            RangeByStepSizeInclusive::new(Radians::ZERO, Radians::HALF_PI, zen_bin_width);
        let azi_bin_count = azi_range.step_count_wrapped();
        let zen_bin_count = zen_range.step_count_wrapped();
        // Bins are stored in the order of zenith first then azimuth, i.e. the
        // zenith bins are stored in the inner loop, and the azimuth bins are
        // stored in the outer loop.
        let mut hist = vec![0.0; azi_bin_count * zen_bin_count];
        for s in &self.slopes {
            // Compute the azimuth and zenith angles of the slope.
            // Convert from [-pi, pi] to [0, 2pi].
            let phi = rad!(-s.y.atan2(-s.x)).wrap_to_tau();
            let theta = rad!((s.x * s.x + s.y * s.y).sqrt().atan());
            // Compute the azimuth and zenith bin indices.
            let azi_bin = azi_range.index_of(phi);
            let zen_bin = zen_range.index_of(theta);
            // Compute the index of the bin.
            let bin_index = azi_bin * zen_bin_count + zen_bin;
            // Increment the bin.
            hist[bin_index] += 1.0;
        }
        // let central_val = hist.iter().step_by(zen_bin_count).sum();
        // hist.iter_mut()
        //     .step_by(zen_bin_count)
        //     .for_each(|v| *v = central_val);
        let count_rcp = math::rcp_f32(self.slopes.len() as f32);
        hist.iter_mut().for_each(|v| *v *= count_rcp);
        SdfPmf {
            azi_bin_count,
            zen_bin_count,
            azi_bin_width,
            zen_bin_width,
            hist,
        }
    }
}

/// Measures the slope distribution function (SDF) of the given microsurfaces.
pub fn measure_slope_distribution(
    handles: &[Handle<MicroSurface>],
    params: SdfMeasurementParams,
    cache: &InnerCache,
) -> Vec<MeasurementData> {
    #[cfg(feature = "bench")]
    let start = std::time::Instant::now();

    log::info!("Measuring the slope distribution function (SDF)...");
    let surfaces = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let measurements = handles
        .iter()
        .zip(surfaces.iter())
        .zip(meshes.iter())
        .filter_map(|((hdl, surf), mesh)| {
            if surf.is_none() || mesh.is_none() {
                log::debug!("Skipping a surface because it is not loaded {:?}.", mesh);
                return None;
            }
            let mesh = mesh.unwrap();
            let mut slopes = vec![Vec2::ZERO; mesh.facet_normals.len()];
            // Iterate over all facet normals to compute the slopes.
            for (n, s) in mesh.facet_normals.iter().zip(slopes.iter_mut()) {
                // Compute the slope of the microfacet normal.
                *s = Vec2::new(-n.x, -n.y) / n.z;
            }
            Some(MeasurementData {
                name: format!("sdf-{}", surf.unwrap().file_stem().unwrap()),
                source: MeasurementDataSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: MeasuredData::Sdf(MeasuredSdfData { params, slopes }),
            })
        })
        .collect();

    #[cfg(feature = "bench")]
    {
        let elapsed = start.elapsed();
        log::info!("SDF measurement took {} ms.", elapsed.as_millis());
    }

    measurements
}

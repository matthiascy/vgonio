use crate::{
    app::cache::{Handle, InnerCache},
    measure::{
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::SdfMeasurementParams,
    },
};
use image::FlatSamples;
use std::path::Path;
use vgcore::{
    error::VgonioError,
    math::{IVec2, UVec2, Vec2},
    units::Radians,
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
                list: SmallVec::from_vec(vec![AnyChannel::new("SDF", FlatSamples::F32(pixels))]),
            },
        );
        let image = Image::from_layer(layer);
        image
            .write()
            .to_file(filepath)
            .map_err(|err| VgonioError::new("Failed to write SDF EXR file.", Some(Box::new(err))))
    }

    /// Returns the histogram of the slopes on the x-direction with the given
    /// bin width.
    ///
    /// # Arguments
    ///
    /// * `bin_width` - The width of each bin. Note that the width here is the
    /// width of the bin (in angles) in the normal space, not the slope space,
    /// which means that the actual width of the bin in the slope space is
    /// the tangent of the bin width.
    pub fn hist_x(&self, bin_width: Radians) -> Vec<f32> {
        let bins_count = (Radians::HALF_PI / bin_width).floor();
        for slope in &self.slopes {
            // Compute the slope angle.
            let angle = slope.x.atan();
            // Compute the bin index.
            let bin_index = (angle / bin_width).as_f32().floor() as usize;
            // Increment the bin.
            hist[bin_index] += 1.0;
        }
        hist
    }

    /// Returns the histogram of the slopes on the y-direction with the given
    /// bin width.
    ///
    /// # Arguments
    ///
    /// * `bin_width` - The width of each bin. Note that the width here is the
    /// width of the bin (in angles) in the normal space, not the slope space,
    /// which means that the actual width of the bin in the slope space is
    /// the tangent of the bin width.
    pub fn hist_y(&self, bin_width: Radians) -> Vec<f32> { todo!("hist_slope_y") }

    /// Returns the histogram of the slopes with the given bin width.
    pub fn hist(&self, bin_width: f32) -> Vec<f32> { todo!("hist_slope") }
}

/// Measures the slope distribution function (SDF) of the given microsurfaces.
pub fn measure_slope_distribution(
    handles: &[Handle<MicroSurface>],
    params: SdfMeasurementParams,
    cache: &InnerCache,
) -> Vec<MeasurementData> {
    log::info!("Measuring the slope distribution function (SDF)...");
    let surfaces = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    handles
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
                name: surf.unwrap().file_name().unwrap().to_owned(),
                source: MeasurementDataSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: MeasuredData::Sdf(MeasuredSdfData { params, slopes }),
            })
        })
        .collect()
}

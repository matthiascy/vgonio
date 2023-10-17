use crate::{
    app::cache::{Handle, InnerCache},
    measure::data::MeasurementData,
};
use std::path::Path;
use vgcore::error::VgonioError;
use vgsurf::MicroSurface;

#[derive(Debug, Clone)]
pub struct MeasuredSdfData {}

impl MeasuredSdfData {
    /// Writes the measured data as an EXR file.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        max_slope: f32,
    ) -> Result<(), VgonioError> {
        todo!("write_as_exr")
    }
}

pub fn measure_slope_distribution(
    surfaces: &[Handle<MicroSurface>],
    cache: &InnerCache,
) -> Vec<MeasurementData> {
    todo!("measure_slope_distribution")
}

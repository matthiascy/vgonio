use crate::{
    app::cache::{Handle, InnerCache},
    measure::data::MeasurementData,
};
use vgsurf::MicroSurface;

#[derive(Debug, Clone)]
pub struct MeasuredSdfData {}

pub fn measure_slope_distribution(
    surfaces: &[Handle<MicroSurface>],
    cache: &InnerCache,
) -> Vec<MeasurementData> {
    todo!("measure_slope_distribution")
}

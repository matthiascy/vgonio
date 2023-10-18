//! Microfacet Slope Distribution Function (MSDF) measurement implementation.

use crate::measure::params::SdfMeasurementParams;

#[derive(Debug)]
pub struct SdfMeasurementTab {
    pub params: SdfMeasurementParams,
}

impl SdfMeasurementTab {
    pub fn new() -> Self {
        Self {
            params: SdfMeasurementParams {
                max_slope: std::f32::consts::PI,
            },
        }
    }

    pub fn ui(&mut self, _ui: &mut egui::Ui) {}
}

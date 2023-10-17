//! Microfacet Slope Distribution Function (MSDF) measurement implementation.

#[derive(Debug)]
pub struct SdfMeasurementTab;

impl SdfMeasurementTab {
    pub fn new() -> Self { Self }

    pub fn ui(&mut self, _ui: &mut egui::Ui) {}
}

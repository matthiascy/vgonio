use crate::{
    app::gui::{
        event::{EventLoopProxy, MeasureEvent, VgonioEvent},
        widgets::SurfaceSelector,
    },
    measure::{
        bsdf::{
            detector::{DetectorParams, DetectorScheme},
            emitter::EmitterParams,
            BsdfKind,
        },
        params::BsdfMeasurementParams,
    },
    Medium, SphericalDomain,
};
use std::hash::Hash;

impl BsdfKind {
    /// Creates the UI for selecting the BSDF kind.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_source(id_source)
            .selected_text(format!("{}", self))
            .show_ui(ui, |ui| {
                ui.selectable_value(self, BsdfKind::Brdf, "BRDF");
                ui.selectable_value(self, BsdfKind::Btdf, "BTDF");
                ui.selectable_value(self, BsdfKind::Bssdf, "BSSDF");
                ui.selectable_value(self, BsdfKind::Bssrdf, "BSSRDF");
                ui.selectable_value(self, BsdfKind::Bssrdf, "BSSRDF");
            });
    }
}

impl Medium {
    /// Creates the UI for selecting the medium.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_source(id_source)
            .selected_text(format!("{:?}", self))
            .show_ui(ui, |ui| {
                ui.selectable_value(self, Medium::Air, "Air");
                ui.selectable_value(self, Medium::Copper, "Copper");
                ui.selectable_value(self, Medium::Aluminium, "Aluminium");
                ui.selectable_value(self, Medium::Vacuum, "Vacuum");
            });
    }
}

impl EmitterParams {
    /// Creates the UI for parameterizing the emitter.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Emitter")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("emitter_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Number of rays: ");
                        ui.add(
                            egui::DragValue::new(&mut self.num_rays)
                                .speed(1.0)
                                .clamp_range(1..=100_000_000),
                        );
                        ui.end_row();

                        ui.label("Max. bounces: ");
                        ui.add(
                            egui::DragValue::new(&mut self.max_bounces)
                                .speed(1.0)
                                .clamp_range(1..=100),
                        );
                        ui.end_row();

                        ui.label("Azimuthal range φ: ");
                        self.azimuth.ui(ui);
                        ui.end_row();

                        ui.label("Zenith range θ: ");
                        self.zenith.ui(ui);
                        ui.end_row();

                        ui.label("Wavelength range: ");
                        self.spectrum.ui(ui);
                        ui.end_row();
                    });
            });
    }
}

impl SphericalDomain {
    /// Creates the UI for parameterizing the spherical domain.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(*self == SphericalDomain::Upper, "Upper")
                .on_hover_text("The upper hemisphere.")
                .clicked()
            {
                *self = SphericalDomain::Upper;
            }

            if ui
                .selectable_label(*self == SphericalDomain::Lower, "Lower")
                .on_hover_text("The lower hemisphere.")
                .clicked()
            {
                *self = SphericalDomain::Lower;
            }

            if ui
                .selectable_label(*self == SphericalDomain::Whole, "Whole")
                .on_hover_text("The whole sphere.")
                .clicked()
            {
                *self = SphericalDomain::Whole;
            }
        });
    }
}

pub struct BsdfSimulation {
    pub params: BsdfMeasurementParams,
    pub(crate) selector: SurfaceSelector,
    event_loop: EventLoopProxy,
}

impl BsdfSimulation {
    /// Creates a new BSDF simulation UI.
    pub fn new(event_loop: EventLoopProxy) -> Self {
        Self {
            params: BsdfMeasurementParams::default(),
            selector: SurfaceSelector::multiple(),
            event_loop,
        }
    }

    /// UI for BSDF simulation parameters.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("bsdf_sim_grid")
            .striped(true)
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("BSDF type:");
                self.params.kind.selectable_ui("bsdf_kind_choice", ui);
                ui.end_row();

                ui.label("Incident medium:");
                self.params
                    .incident_medium
                    .selectable_ui("incident_medium_choice", ui);
                ui.end_row();

                ui.label("Surface medium:");
                self.params
                    .transmitted_medium
                    .selectable_ui("transmitted_medium_choice", ui);
                ui.end_row();

                ui.label("Micro-surfaces: ");
                self.selector.ui("micro_surface_selector", ui);
                ui.end_row();
            });

        self.params.emitter.ui(ui);
        self.params.detector.ui(ui);
        if ui.button("Simulate").clicked() {
            self.event_loop
                .send_event(VgonioEvent::Measure(MeasureEvent::Bsdf {
                    params: self.params,
                    surfaces: self.selector.selected().collect(),
                }))
                .unwrap();
        }
    }
}

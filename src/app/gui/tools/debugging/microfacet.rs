use crate::{
    app::gui::VgonioEvent,
    units::{degrees, Degrees},
};
use egui::{Response, Ui};
use winit::event_loop::EventLoopProxy;

pub struct MicrofacetMeasurementPane {
    event_loop: EventLoopProxy<VgonioEvent>,
    m_azimuth: Degrees,
    m_zenith: Degrees,
    opening_angle: Degrees,
}

impl MicrofacetMeasurementPane {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>) -> Self {
        Self {
            event_loop,
            m_azimuth: degrees!(0.0),
            m_zenith: degrees!(0.0),
            opening_angle: degrees!(0.0),
        }
    }
}

impl egui::Widget for &mut MicrofacetMeasurementPane {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.horizontal_wrapped(|ui| {
            ui.label("view direction");
            ui.add(
                egui::DragValue::new(&mut self.m_azimuth.value)
                    .speed(0.1)
                    .prefix("φ: ")
                    .suffix("°"),
            );
            ui.add(
                egui::DragValue::new(&mut self.m_zenith.value)
                    .speed(0.1)
                    .prefix("θ: ")
                    .suffix("°"),
            );
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("opening angle");
            ui.add(
                egui::DragValue::new(&mut self.opening_angle.value)
                    .speed(0.1)
                    .suffix("°"),
            );
        });

        ui.horizontal_wrapped(|ui| {
            if ui.button("check").clicked()
                && self
                    .event_loop
                    .send_event(VgonioEvent::CheckVisibleFacets {
                        m_azimuth: self.m_azimuth,
                        m_zenith: self.m_zenith,
                        opening_angle: self.opening_angle,
                    })
                    .is_err()
            {
                log::warn!("Failed to send event VgonioEvent::UpdateCellPos");
            }
        })
        .response
    }
}

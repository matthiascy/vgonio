use crate::app::gui::{VgonioEvent, VgonioEventLoop};
use egui::{Response, Ui};
use vgcore::units::{degrees, Degrees};

pub struct MicrofacetDebugging {
    event_loop: VgonioEventLoop,
    m_azimuth: Degrees,
    m_zenith: Degrees,
    opening_angle: Degrees,
}

impl MicrofacetDebugging {
    pub fn new(event_loop: VgonioEventLoop) -> Self {
        Self {
            event_loop,
            m_azimuth: degrees!(0.0),
            m_zenith: degrees!(0.0),
            opening_angle: degrees!(0.0),
        }
    }
}

impl egui::Widget for &mut MicrofacetDebugging {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.horizontal_wrapped(|ui| {
            ui.label("view direction");
            ui.add(
                egui::DragValue::new(&mut self.m_azimuth.value())
                    .speed(0.1)
                    .prefix("φ: ")
                    .suffix("°"),
            );
            ui.add(
                egui::DragValue::new(&mut self.m_zenith.value())
                    .speed(0.1)
                    .prefix("θ: ")
                    .suffix("°"),
            );
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("opening angle");
            ui.add(
                egui::DragValue::new(&mut self.opening_angle.value())
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

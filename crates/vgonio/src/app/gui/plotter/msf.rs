#[cfg(debug_assertions)]
use crate::app::gui::plotter::{debug_print_angle, debug_print_angle_pair};
use crate::{
    app::{
        cache::{Cache, Handle, RawCache},
        gui::{
            event::EventLoopProxy,
            plotter::{angle_knob, Curve, VariantData},
        },
    },
    fitting::FittedModel,
    measure::data::MeasurementData,
};
use base::{
    range::RangeByStepSizeInclusive,
    units::{rad, Radians},
};
use bxdf::distro::MicrofacetDistribution;
use egui::{Align, Ui};
use std::any::Any;

pub struct MaskingShadowingExtra {
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    pub azimuth_m: Radians,
    /// The zenith angle of (facet normal m) the slice to be displayed, in
    /// radians.
    pub zenith_m: Radians,
    /// The azimuthal angle (incident/outgoing direction v) of the slice to be
    /// displayed,
    pub azimuth_v: Radians,
    /// The azimuthal angle range of the measured data, in radians.
    pub azimuth_range: RangeByStepSizeInclusive<Radians>,
    /// The zenith angle range of the measured data, in radians.
    pub zenith_range: RangeByStepSizeInclusive<Radians>,
    /// Curve cache extracted from the measurement data. The first index is the
    /// azimuthal angle of microfacet normal m, the second index is the zenith
    /// angle of microfacet normal m, the third index is the azimuthal angle of
    /// incident/outgoing direction v.
    pub curves: Vec<Curve>,
    // /// The fitted curves together with the fitted model.
    pub fitted: Vec<(Box<dyn MicrofacetDistribution<Params = [f64; 2]>>, Curve)>,
}

impl Default for MaskingShadowingExtra {
    fn default() -> Self {
        Self {
            azimuth_m: rad!(0.0),
            zenith_m: rad!(0.0),
            azimuth_v: rad!(0.0),
            azimuth_range: RangeByStepSizeInclusive {
                start: rad!(0.0),
                stop: rad!(0.0),
                step_size: rad!(0.0),
            },
            zenith_range: RangeByStepSizeInclusive {
                start: rad!(0.0),
                stop: rad!(0.0),
                step_size: rad!(0.0),
            },
            curves: vec![],
            fitted: vec![],
        }
    }
}

impl VariantData for MaskingShadowingExtra {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &RawCache) {
        let measurement = cache.get_measurement_data(data).unwrap();
        let (azimuth, zenith) = measurement.measured.adf_or_msf_angle_ranges().unwrap();
        self.azimuth_range = azimuth;
        self.zenith_range = zenith;
        // let zenith_step_count = self.zenith_range.step_count_wrapped();
        for phi_m in self.azimuth_range.values_wrapped() {
            for theta_m in self.zenith_range.values() {
                for phi_v in self.azimuth_range.values_wrapped() {
                    let (starting, opposite) = measurement.msf_data_slice(phi_m, theta_m, phi_v);
                    let first_part_points = starting
                        .iter()
                        .zip(self.zenith_range.values())
                        .map(|(y, x)| [x.as_f64(), *y as f64]);
                    self.curves.push(Curve::from_adf_or_msf_data(
                        first_part_points,
                        opposite,
                        &self.zenith_range,
                    ));
                }
            }
        }
    }

    fn current_curve(&self) -> Option<&Curve> {
        let phi_m_idx = self.azimuth_range.index_of(self.azimuth_m.wrap_to_tau());
        let theta_m_idx = self.zenith_range.index_of(self.zenith_m);
        let phi_v_idx = self.azimuth_range.index_of(self.azimuth_v.wrap_to_tau());
        let azimuth_step_count = self.azimuth_range.step_count_wrapped();
        let zenith_step_count = self.zenith_range.step_count_wrapped();
        let index = phi_m_idx * zenith_step_count * azimuth_step_count
            + theta_m_idx * azimuth_step_count
            + phi_v_idx;
        debug_assert!(index < self.curves.len(), "Curve index out of bounds!");
        self.curves.get(index)
    }

    fn update_fitted_curves(&mut self, _models: &[FittedModel]) {
        // let theta_values = self
        //     .zenith_range
        //     .values_rev()
        //     .map(|x| -x.as_f64())
        //     .chain(self.zenith_range.values().skip(1).map(|x| x.as_f64()))
        //     .collect::<Vec<_>>();
        // let to_add = models
        //     .iter()
        //     .filter(|model| {
        //         !self
        //             .fitted
        //             .iter()
        //             .any(|(existing_model, _)| model.family() ==
        // existing_model.family())     })
        //     .collect::<Vec<_>>();
        //
        // if to_add.is_empty() {
        //     return;
        // } else {
        //     for fitted in to_add {
        //         match fitted {
        //             FittedModel::Mmsf(model) => {
        //                 // Generate the curve for this model.
        //                 let points = theta_values
        //                     .iter()
        //                     .zip(
        //                         theta_values
        //                             .iter()
        //                             .map(|theta_m|
        //     model.eval_with_cos_theta_v(theta_m.cos())),                 )
        //                     .map(|(x, y)| [*x, y]);
        //                 self.fitted.push((model.clone_box(),
        // Curve::from(points)));             }
        //             _ => {
        //                 unreachable!("Wrong model type for
        // masking-shadowing!")             }
        //         }
        //     }
        //     return;
        // }
    }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn ui(
        &mut self,
        ui: &mut Ui,
        _event_loop: &EventLoopProxy,
        _data: Handle<MeasurementData>,
        _cache: &Cache,
    ) {
        ui.allocate_ui_with_layout(
            egui::Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("Microfacet normal: ");
                angle_knob(
                    ui,
                    true,
                    &mut self.zenith_m,
                    self.zenith_range.range_bound_inclusive_f32(),
                    self.zenith_range.step_size,
                    48.0,
                    |v| format!("θ = {:>6.2}°", v.to_degrees()),
                );
                angle_knob(
                    ui,
                    true,
                    &mut self.azimuth_m,
                    self.azimuth_range.range_bound_inclusive_f32(),
                    self.azimuth_range.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                {
                    debug_print_angle_pair(
                        self.azimuth_m,
                        &self.azimuth_range,
                        ui,
                        "debug_print_φ",
                    );
                    debug_print_angle(self.zenith_m, &self.zenith_range, ui, "debug_print_θ");
                }
            },
        );
        ui.allocate_ui_with_layout(
            egui::Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("incident direction: ");
                let mut opposite = self.azimuth_v.opposite();
                angle_knob(
                    ui,
                    false,
                    &mut opposite,
                    self.azimuth_range.range_bound_inclusive_f32(),
                    self.azimuth_range.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                angle_knob(
                    ui,
                    true,
                    &mut self.azimuth_v,
                    self.azimuth_range.range_bound_inclusive_f32(),
                    self.azimuth_range.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                debug_print_angle_pair(
                    self.azimuth_v,
                    &self.azimuth_range,
                    ui,
                    "debug_print_φ_pair",
                );
            },
        );
    }
}

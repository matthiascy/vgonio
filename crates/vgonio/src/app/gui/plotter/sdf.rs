use crate::{
    app::{
        cache::{Cache, Handle, RawCache},
        gui::{
            event::EventLoopProxy,
            misc::drag_angle,
            plotter::{angle_knob, Curve, VariantData},
        },
    },
    fitting::FittedModel,
    measure::data::MeasurementData,
    RangeByStepSizeInclusive,
};
use base::units::{deg, rad, Radians};
use egui::{Align, Ui};
use std::any::Any;

#[cfg(debug_assertions)]
use crate::app::gui::plotter::debug_print_angle_pair;

pub struct SlopeDistributionExtra {
    /// The azimuth of the facet normal.
    pub azimuth_m: Radians,
    /// The range of azimuth and it's bin size.
    pub azi_range: RangeByStepSizeInclusive<Radians>,
    /// The range of zenith and it's bin size.
    pub zen_range: RangeByStepSizeInclusive<Radians>,
    /// Whether to apply the Jacobian from SDF to NDF.
    pub apply_jacobian: bool,
    /// The curves of the slope distribution function estimated from the
    /// measured data according to the azimuth and zenith bin sizes.
    pub curves: Vec<Curve>,
}

impl Default for SlopeDistributionExtra {
    fn default() -> Self {
        Self {
            azimuth_m: rad!(0.0),
            azi_range: RangeByStepSizeInclusive::new(
                Radians::ZERO,
                Radians::TAU,
                deg!(5.0).to_radians(),
            ),
            zen_range: RangeByStepSizeInclusive::new(
                Radians::ZERO,
                Radians::HALF_PI,
                deg!(2.0).to_radians(),
            ),
            apply_jacobian: false,
            curves: Vec::new(),
        }
    }
}

impl SlopeDistributionExtra {
    /// Generates the curves of the slope distribution function estimated from
    /// the measured data according to the azimuth and zenith bin sizes.
    fn generate_curves(&mut self, data: &MeasurementData) {
        let data = data
            .measured
            .as_sdf()
            .expect("Trying to generate SDF curves from non-SDF data");
        self.curves.clear();
        let pmf = data.pmf(self.azi_range.step_size, self.zen_range.step_size);
        for azi_idx in 0..pmf.azi_bin_count {
            // let azi = self.azi_range.step(azi_idx);
            // let opposite_azi = azi.opposite();
            // let opposite_azi_idx = self.azi_range.index_of(opposite_azi);
            // let opposite_pmf = &pmf.hist
            //     [opposite_azi_idx * pmf.zen_bin_count..(opposite_azi_idx + 1) *
            // pmf.zen_bin_count]; let starting =
            //     &pmf.hist[azi_idx * pmf.zen_bin_count..(azi_idx + 1) *
            // pmf.zen_bin_count]; self.curves.push(Curve::from(
            //     opposite_pmf
            //         .iter()
            //         .rev()
            //         .zip(self.zen_range.values_rev().map(|x| -x.as_f64()))
            //         .take(pmf.zen_bin_count - 1)
            //         .map(|(y, x)| [x, *y as f64])
            //         .chain(
            //             starting
            //                 .iter()
            //                 .zip(self.zen_range.values())
            //                 .map(|(y, x)| [x.as_f64(), *y as f64]),
            //         ),
            // ));
            self.curves.push(Curve::from(
                pmf.hist[azi_idx * pmf.zen_bin_count..(azi_idx + 1) * pmf.zen_bin_count]
                    .iter()
                    .zip(self.zen_range.values())
                    .map(|(y, x)| [x.as_f64(), *y as f64]),
            ));
        }
    }

    pub fn current_azimuth_idx(&self) -> usize {
        let azi_m = self.azimuth_m.wrap_to_tau();
        RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TAU, self.azi_range.step_size)
            .index_of(azi_m)
    }
}

impl VariantData for SlopeDistributionExtra {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &RawCache) {
        self.generate_curves(cache.get_measurement_data(data).unwrap());
    }

    fn current_curve(&self) -> Option<&Curve> { self.curves.get(self.current_azimuth_idx()) }

    fn update_fitted_curves(&mut self, _fitted: &[FittedModel]) {
        // TODO: update fitted curves
    }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn ui(
        &mut self,
        ui: &mut Ui,
        event_loop: &EventLoopProxy,
        data: Handle<MeasurementData>,
        cache: &Cache,
    ) {
        let azi_step_size = self.azi_range.step_size;
        let zen_step_size = self.zen_range.step_size;

        ui.allocate_ui_with_layout(
            egui::Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("Microfacet normal: ");
                let mut opposite = self.azimuth_m.wrap_to_tau().opposite();
                angle_knob(
                    ui,
                    false,
                    &mut opposite,
                    0.0..=std::f32::consts::TAU,
                    azi_step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                angle_knob(
                    ui,
                    true,
                    &mut self.azimuth_m,
                    0.0..=std::f32::consts::TAU,
                    azi_step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                if ui
                    .button("📋")
                    .on_hover_text("Copy the current slice data to the clipboard.")
                    .clicked()
                {
                    ui.output_mut(|o| {
                        if let Some(curve) = self.current_curve() {
                            let mut content = String::new();
                            for [x, y] in &curve.points {
                                content.push_str(&format!("{}, {}\n", x, y));
                            }
                            o.copied_text = content;
                        }
                    });
                }
                #[cfg(debug_assertions)]
                debug_print_angle_pair(self.azimuth_m, &self.azi_range, ui, "debug_print_φ_pair");
            },
        );

        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.apply_jacobian, "Apply Jacobian");
        });

        ui.horizontal(|ui| {
            ui.label("Azimuth bin width: ");
            ui.add(drag_angle(&mut self.azi_range.step_size, ""));
            ui.label("Zenith bin width: ");
            ui.add(drag_angle(&mut self.zen_range.step_size, ""));

            if ui.button("Update").clicked() {
                cache.read(|cache| {
                    self.generate_curves(cache.get_measurement_data(data).unwrap());
                });
            }
        });
    }
}

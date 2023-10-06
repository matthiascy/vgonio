use crate::{
    app::{
        cache::{Handle, InnerCache},
        gui::{
            event::{EventLoopProxy, VgonioEvent},
            plotter::{angle_knob, Curve, VariantData},
        },
    },
    fitting::FittedModel,
    RangeByStepSizeInclusive,
};
use egui::{Align, Ui};
use std::any::Any;
use vgbxdf::{MicrofacetDistributionModel, MicrofacetDistributionModelKind};
use vgcore::units::{rad, Radians};

#[cfg(debug_assertions)]
use crate::app::gui::plotter::debug_print_angle_pair;
use crate::{
    fitting::{FittingProblemKind, MicrofacetDistributionFittingMethod},
    measure::data::MeasurementData,
};

struct ModelSelector {
    model: MicrofacetDistributionModelKind,
}

impl ModelSelector {
    fn ui(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Model: ");
            ui.selectable_value(
                &mut self.model,
                MicrofacetDistributionModelKind::Beckmann,
                "Beckmann",
            );
            ui.selectable_value(
                &mut self.model,
                MicrofacetDistributionModelKind::TrowbridgeReitz,
                "Trowbridge-Reitz",
            );
        });
    }
}

/// Extra data for the area distribution plot.
pub struct AreaDistributionExtra {
    /// The azimuthal angle range of the measured data, in radians.
    pub azimuth_range: RangeByStepSizeInclusive<Radians>,
    /// The zenith angle range of the measured data, in radians.
    pub zenith_range: RangeByStepSizeInclusive<Radians>,
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    pub azimuth_m: Radians,
    /// Curve cache extracted from the measurement data, indexed by the
    /// azimuthal angle. The first one is the curve accumulated from the
    /// azimuthal angles.
    pub curves: Vec<Curve>,
    /// The fitted curves together with the fitted model.
    pub fitted: Vec<(
        Box<dyn MicrofacetDistributionModel>,
        Vec<Curve>, // Only one curve for isotropic model, otherwise one for each azimuthal angle.
    )>,
    /// Selected model to fit the data.
    selected: ModelSelector,
}

impl Default for AreaDistributionExtra {
    fn default() -> Self {
        Self {
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
            azimuth_m: rad!(0.0),
            curves: vec![],
            fitted: vec![],
            selected: ModelSelector {
                model: MicrofacetDistributionModelKind::Beckmann,
            },
        }
    }
}

impl AreaDistributionExtra {
    pub fn current_azimuth_idx(&self) -> usize {
        let azimuth_m = self.azimuth_m.wrap_to_tau();
        self.azimuth_range.index_of(azimuth_m)
    }
}

impl VariantData for AreaDistributionExtra {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &InnerCache) {
        let measurement = cache.get_measurement_data(data).unwrap();
        let (azimuth, zenith) = measurement.measured.adf_or_msf_angle_ranges().unwrap();
        self.azimuth_range = azimuth;
        self.zenith_range = zenith;
        for phi in self.azimuth_range.values_wrapped() {
            let (starting, opposite) = measurement.ndf_data_slice(phi);
            let first_part_points = starting
                .iter()
                .zip(self.zenith_range.values())
                .map(|(y, x)| [x.as_f64(), *y as f64]);
            self.curves.push(Curve::from_mndf_or_mmsf_data(
                first_part_points,
                opposite,
                &self.zenith_range,
            ));
        }
    }

    fn current_curve(&self) -> Option<&Curve> {
        let index = self.current_azimuth_idx();
        debug_assert!(index < self.curves.len(), "Curve index out of bounds!");
        self.curves.get(index)
    }

    fn update_fitted_curves(&mut self, models: &[FittedModel]) {
        let theta_values = self
            .zenith_range
            .values_rev()
            .map(|x| -x.as_f64())
            .chain(self.zenith_range.values().skip(1).map(|x| x.as_f64()))
            .collect::<Vec<_>>();
        let phi_values = self
            .azimuth_range
            .values_wrapped()
            .map(|phi| {
                // REVIEW: The opposite angle may not exist in the measurement range.
                let opposite = phi.wrap_to_tau().opposite();
                (opposite.as_f64(), phi.as_f64())
            })
            .collect::<Vec<_>>();
        let to_add = models
            .iter()
            .filter(|fitted_model| match fitted_model {
                FittedModel::Bsdf() | FittedModel::Msf(_) => todo!("Not implemented yet!"),
                FittedModel::Adf(model) => !self
                    .fitted
                    .iter()
                    .any(|(existing, _)| model.kind() == existing.kind()),
            })
            .collect::<Vec<_>>();

        if to_add.is_empty() {
            return;
        } else {
            for fitted in to_add {
                match fitted {
                    FittedModel::Adf(model) => {
                        // Generate the curve for this model.
                        let curves = phi_values
                            .iter()
                            .map(|(phi_l, phi_r)| {
                                let points = {
                                    theta_values
                                        .iter()
                                        .zip(theta_values.iter().map(|theta| {
                                            let phi = if theta < &0.0 { *phi_l } else { *phi_r };
                                            model.eval_adf(theta.cos(), phi.cos())
                                        }))
                                        .map(|(x, y)| [*x, y])
                                };
                                Curve::from(points)
                            })
                            .collect();
                        self.fitted.push((model.clone_box(), curves));
                    }
                    _ => {
                        unreachable!("Wrong model type for area distribution!")
                    }
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn ui(&mut self, ui: &mut Ui, event_loop: &EventLoopProxy, data: Handle<MeasurementData>) {
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
                    self.azimuth_range
                        .map(|x| x.value())
                        .range_bound_inclusive(),
                    self.azimuth_range.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                angle_knob(
                    ui,
                    true,
                    &mut self.azimuth_m,
                    self.azimuth_range
                        .map(|x| x.value())
                        .range_bound_inclusive(),
                    self.azimuth_range.step_size,
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
                debug_print_angle_pair(
                    self.azimuth_m,
                    &self.azimuth_range,
                    ui,
                    "debug_print_φ_pair",
                );
            },
        );

        egui::CollapsingHeader::new("Fitting")
            .default_open(true)
            .show(ui, |ui| {
                self.selected.ui(ui);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button(
                            egui::WidgetText::RichText(egui::RichText::from("Fit"))
                                .text_style(egui::TextStyle::Monospace),
                        )
                        .clicked()
                    {
                        event_loop.send_event(VgonioEvent::Fitting {
                            kind: FittingProblemKind::Mdf {
                                model: self.selected.model,
                                method: MicrofacetDistributionFittingMethod::Adf,
                            },
                            data,
                        });
                    }
                });

                ui.label("Fitted models: ");
                if self.fitted.is_empty() {
                    ui.label("None");
                } else {
                    for (model, _) in &self.fitted {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::from(model.kind().to_str())
                                    .text_style(egui::TextStyle::Monospace),
                            );
                            ui.label(egui::RichText::from(format!(
                                "αx = {:.4}, αy = {:.4}",
                                model.alpha_x(),
                                model.alpha_y()
                            )));
                        });
                    }
                }
            });
    }
}

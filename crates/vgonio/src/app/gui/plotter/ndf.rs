#[cfg(feature = "fitting")]
use crate::fitting::{FittedModel, FittingProblemKind};

#[cfg(debug_assertions)]
use crate::app::gui::plotter::debug_print_angle_pair;
use crate::{
    app::{
        cache::{Cache, RawCache},
        gui::{
            event::{EventLoopProxy, VgonioEvent},
            plotter::{angle_knob, Curve, VariantData},
        },
    },
    measure::{mfd::MeasuredNdfData, Measurement},
};
use base::{
    handle::Handle,
    range::RangeByStepSizeInclusive,
    units::{rad, Radians},
    Isotropy,
};
use bxdf::distro::{MicrofacetDistribution, MicrofacetDistroKind};
use egui::{Align, Ui};
use std::any::Any;

struct ModelSelector {
    model: MicrofacetDistroKind,
}

impl ModelSelector {
    fn ui(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Model: ");
            ui.selectable_value(&mut self.model, MicrofacetDistroKind::Beckmann, "Beckmann");
            ui.selectable_value(
                &mut self.model,
                MicrofacetDistroKind::TrowbridgeReitz,
                "Trowbridge-Reitz",
            );
        });
    }
}

/// Extra data for the normal distribution plot.
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
    /// The scale factor of the current curve.
    pub scale_factor: f32,
    /// Enable the Slope Distribution plot.
    pub show_slope_distribution: bool,
    /// The fitted curves together with the fitted model.
    pub fitted: Vec<(
        Box<dyn MicrofacetDistribution<Params = [f64; 2]>>,
        f32,
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
            scale_factor: 1.0,
            show_slope_distribution: false,
            fitted: vec![],
            selected: ModelSelector {
                model: MicrofacetDistroKind::Beckmann,
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
    fn pre_process(&mut self, data: Handle<Measurement>, cache: &RawCache) {
        let measurement = cache.get_measurement(data).unwrap();
        let ndf = measurement
            .measured
            .downcast_ref::<MeasuredNdfData>()
            .unwrap();
        let (azimuth, zenith) = ndf.measurement_range().unwrap();
        self.azimuth_range = azimuth;
        self.zenith_range = zenith;
        for phi in self.azimuth_range.values_wrapped() {
            let (starting, opposite) = ndf.slice_at(phi);
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

    fn current_curve(&self) -> Option<&Curve> {
        let index = self.current_azimuth_idx();
        debug_assert!(index < self.curves.len(), "Curve index out of bounds!");
        self.curves.get(index)
    }

    fn scale_factor(&self) -> f32 { self.scale_factor }

    #[cfg(feature = "fitting")]
    fn update_fitted_curves(&mut self, models: &[FittedModel]) {
        let to_add = models
            .iter()
            .filter(|fitted_model| match fitted_model {
                FittedModel::Bsdf(_) | FittedModel::Msf(_) => todo!("Not implemented yet!"),
                FittedModel::Ndf(model, scale) => {
                    !self.fitted.iter().any(|(existing, existing_scale, _)| {
                        model.kind() == existing.kind()
                            && *scale == *existing_scale
                            && model.isotropy() == existing.isotropy()
                    })
                },
            })
            .collect::<Vec<_>>();

        if to_add.is_empty() {
            return;
        }

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

        for fitted in to_add {
            match fitted {
                FittedModel::Ndf(model, scale) => {
                    // Generate the curve for this model.
                    let curves = phi_values
                        .iter()
                        .map(|(phi_l, phi_r)| {
                            let points = {
                                theta_values
                                    .iter()
                                    .zip(theta_values.iter().map(|theta| {
                                        let phi = if theta < &0.0 { *phi_l } else { *phi_r };
                                        model.eval_ndf(theta.cos(), phi.cos())
                                    }))
                                    .map(|(x, y)| [*x, y])
                            };
                            Curve::from(points)
                        })
                        .collect();
                    self.fitted.push((model.clone_box(), *scale, curves));
                },
                _ => {
                    unreachable!("Wrong model type for area distribution!")
                },
            }
        }
    }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn ui(
        &mut self,
        ui: &mut Ui,
        event_loop: &EventLoopProxy,
        data: Handle<Measurement>,
        _cache: &Cache,
    ) {
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
                            log::debug!("Copied data: {}", content);
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

        ui.horizontal(|ui| {
            ui.label("Scale factor: ");
            ui.add(
                egui::DragValue::new(&mut self.scale_factor)
                    .speed(0.01)
                    .clamp_range(1.0..=1024.0),
            );
            ui.toggle_value(&mut self.show_slope_distribution, "Show slope distribution");
        });

        #[cfg(feature = "fitting")]
        egui::CollapsingHeader::new("Fitting")
            .default_open(true)
            .show(ui, |ui| {
                self.selected.ui(ui);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button(
                            egui::WidgetText::RichText(egui::RichText::from(
                                "Fit with scale (iso)",
                            ))
                            .text_style(egui::TextStyle::Monospace),
                        )
                        .clicked()
                    {
                        event_loop.send_event(VgonioEvent::Fitting {
                            kind: FittingProblemKind::Mfd {
                                model: self.selected.model,
                                isotropy: Isotropy::Isotropic,
                            },
                            data,
                            scale: self.scale_factor,
                        });
                    }

                    if ui
                        .button(
                            egui::WidgetText::RichText(egui::RichText::from(
                                "Fit with scale (aniso)",
                            ))
                            .text_style(egui::TextStyle::Monospace),
                        )
                        .clicked()
                    {
                        event_loop.send_event(VgonioEvent::Fitting {
                            kind: FittingProblemKind::Mfd {
                                model: self.selected.model,
                                isotropy: Isotropy::Anisotropic,
                            },
                            data,
                            scale: 1.0,
                        });
                    }
                });

                ui.label("Fitted models: ");
                if self.fitted.is_empty() {
                    ui.label("None");
                } else {
                    for (model, scale, _) in &self.fitted {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::from(model.kind().to_str())
                                    .text_style(egui::TextStyle::Monospace),
                            );
                            ui.label(egui::RichText::from(format!(
                                "αx = {:.4}, αy = {:.4}, scale = {:.4}, {}",
                                model.params()[0],
                                model.params()[1],
                                scale,
                                model.isotropy().to_string().to_lowercase()
                            )));
                        });
                    }
                }
            });
    }
}

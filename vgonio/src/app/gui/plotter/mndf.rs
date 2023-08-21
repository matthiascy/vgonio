use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{
            event::{EventLoopProxy, VgonioEvent},
            plotter::{angle_knob, Curve, VariantData},
        },
    },
    fitting::{
        AreaDistributionFittingMode, FittedModel, MicrofacetAreaDistributionModel,
        MicrofacetModelFamily, ReflectionModelFamily,
    },
    measure::measurement::{MeasurementData, MeasurementKind},
    RangeByStepSizeInclusive,
};
use egui::{Align, Ui};
use rand_distr::num_traits::real::Real;
use std::any::Any;
use vgcore::units::{rad, Radians};

#[cfg(debug_assertions)]
use crate::app::gui::plotter::debug_print_angle_pair;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MndfModel {
    BeckmannSpizzichino,
    BeckmannSpizzichinoAnisotropic,
    #[cfg(feature = "scaled-ndf-fitting")]
    ScaledBeckmannSpizzichino,
    #[cfg(feature = "scaled-ndf-fitting")]
    ScaledBeckmannSpizzichinoAnisotropic,
    TrowbridgeReitz,
    TrowbridgeReitzAnisotropic,
    #[cfg(feature = "scaled-ndf-fitting")]
    ScaledTrowbridgeReitz,
    #[cfg(feature = "scaled-ndf-fitting")]
    ScaledTrowbridgeReitzAnisotropic,
}

impl MndfModel {
    pub fn is_isotropic(&self) -> bool {
        match self {
            MndfModel::BeckmannSpizzichino | MndfModel::TrowbridgeReitz => true,
            #[cfg(feature = "scaled-ndf-fitting")]
            MndfModel::ScaledBeckmannSpizzichino | MndfModel::ScaledTrowbridgeReitz => true,
            _ => false,
        }
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
        Box<dyn MicrofacetAreaDistributionModel>,
        AreaDistributionFittingMode,
        Curve,
    )>,
    /// Model to be fitted.
    selected_model: MndfModel,
    /// Fitting mode.
    mode: AreaDistributionFittingMode,
    /// Whether to show the accumulated curve.
    pub show_accumulated: bool,
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
            selected_model: MndfModel::BeckmannSpizzichino,
            mode: AreaDistributionFittingMode::Complete,
            show_accumulated: false,
        }
    }
}

impl VariantData for AreaDistributionExtra {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache) {
        let measurement = cache.get_measurement_data(data).unwrap();
        self.azimuth_range = measurement.measured.madf_or_mmsf_azimuth().unwrap();
        self.zenith_range = measurement.measured.madf_or_mmsf_zenith().unwrap();
        self.curves.push(Curve::from(
            measurement
                .measured
                .madf_data()
                .unwrap()
                .accumulated_slice()
                .into_iter()
                .map(|(theta, value)| [theta as f64, value as f64]),
        ));

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
        let azimuth_m = self.azimuth_m.wrap_to_tau();
        let index = self.azimuth_range.index_of(azimuth_m);
        debug_assert!(index < self.curves.len(), "Curve index out of bounds!");
        self.curves.get(index + 1)
    }

    fn update_fitted_curves(&mut self, models: &[FittedModel]) {
        let theta_values = self
            .zenith_range
            .values_rev()
            .map(|x| -x.as_f64())
            .chain(self.zenith_range.values().skip(1).map(|x| x.as_f64()))
            .collect::<Vec<_>>();
        let to_add = models
            .iter()
            .filter(|fitted_model| match fitted_model {
                FittedModel::Bsdf(_) | FittedModel::Mmsf(_) => {
                    #[cfg(feature = "scaled-ndf-fitting")]
                    {
                        !self.fitted.iter().any(|(existing_model, _, _)| {
                            fitted_model.family() == existing_model.family()
                                && fitted_model.is_scaled() == existing_model.scale().is_some()
                        })
                    }

                    #[cfg(not(feature = "scaled-ndf-fitting"))]
                    {
                        !self.fitted.iter().any(|(existing_model, _, _)| {
                            fitted_model.family() == existing_model.family()
                        })
                    }
                }
                FittedModel::Mndf { model, mode } => {
                    #[cfg(feature = "scaled-ndf-fitting")]
                    {
                        !self.fitted.iter().any(|(existing_model, fitting_mode, _)| {
                            fitted_model.family() == existing_model.family()
                                && fitted_model.is_scaled() == existing_model.scale().is_some()
                                && mode == fitting_mode
                                && model.is_isotropic() == existing_model.is_isotropic()
                        })
                    }
                    #[cfg(not(feature = "scaled-ndf-fitting"))]
                    {
                        !self.fitted.iter().any(|(existing_model, fitting_mode, _)| {
                            fitted_model.family() == existing_model.family()
                                && mode == fitting_mode
                                && model.is_isotropic() == existing_model.is_isotropic()
                        })
                    }
                }
            })
            .collect::<Vec<_>>();

        if to_add.is_empty() {
            return;
        } else {
            for fitted in to_add {
                match fitted {
                    FittedModel::Mndf { model, mode } => {
                        // Generate the curve for this model.
                        let points = {
                            theta_values
                                .iter()
                                .zip(theta_values.iter().map(|theta_m| {
                                    #[cfg(feature = "scaled-ndf-fitting")]
                                    {
                                        model.eval_with_theta_m(*theta_m)
                                            * model.scale().unwrap_or(1.0)
                                    }
                                    #[cfg(not(feature = "scaled-ndf-fitting"))]
                                    {
                                        model.eval_with_theta_m(*theta_m)
                                    }
                                }))
                                .map(|(x, y)| [*x, y])
                        };

                        self.fitted
                            .push((model.clone_box(), *mode, Curve::from(points)));
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
                ui.checkbox(&mut self.show_accumulated, "Accumulated");
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
                ui.horizontal_wrapped(|ui| {
                    ui.label("Model: ");
                    ui.vertical(|ui| {
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::BeckmannSpizzichino,
                            "Beckmann-Spizzichino",
                        );
                        #[cfg(feature = "scaled-ndf-fitting")]
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::ScaledBeckmannSpizzichino,
                            "Scaled Beckmann-Spizzichino",
                        );
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::TrowbridgeReitz,
                            "Trowbridge-Reitz",
                        );
                        #[cfg(feature = "scaled-ndf-fitting")]
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::ScaledTrowbridgeReitz,
                            "Scaled Trowbridge-Reitz",
                        );
                    });
                    ui.vertical(|ui| {
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::BeckmannSpizzichinoAnisotropic,
                            "Beckmann-Spizzichino Anisotropic",
                        );
                        #[cfg(feature = "scaled-ndf-fitting")]
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::ScaledBeckmannSpizzichinoAnisotropic,
                            "Scaled Beckmann-Spizzichino Anisotropic",
                        );
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::TrowbridgeReitzAnisotropic,
                            "Trowbridge-Reitz Anisotropic",
                        );
                        #[cfg(feature = "scaled-ndf-fitting")]
                        ui.selectable_value(
                            &mut self.selected_model,
                            MndfModel::ScaledTrowbridgeReitzAnisotropic,
                            "Scaled Trowbridge-Reitz Anisotropic",
                        );
                    });
                });

                ui.horizontal_wrapped(|ui| {
                    if self.selected_model.is_isotropic() {
                        ui.selectable_value(
                            &mut self.mode,
                            AreaDistributionFittingMode::Complete,
                            "Complete",
                        );
                        ui.selectable_value(
                            &mut self.mode,
                            AreaDistributionFittingMode::Accumulated,
                            "Accumulated",
                        );
                    }
                });

                if ui
                    .button(
                        egui::WidgetText::RichText(egui::RichText::from("Fit"))
                            .text_style(egui::TextStyle::Monospace),
                    )
                    .clicked()
                {
                    let family = match self.selected_model {
                        #[cfg(feature = "scaled-ndf-fitting")]
                        MndfModel::BeckmannSpizzichino
                        | MndfModel::ScaledBeckmannSpizzichino
                        | MndfModel::ScaledBeckmannSpizzichinoAnisotropic
                        | MndfModel::BeckmannSpizzichinoAnisotropic => {
                            ReflectionModelFamily::Microfacet(
                                MicrofacetModelFamily::BeckmannSpizzichino,
                            )
                        }
                        #[cfg(feature = "scaled-ndf-fitting")]
                        MndfModel::TrowbridgeReitz
                        | MndfModel::ScaledTrowbridgeReitz
                        | MndfModel::ScaledTrowbridgeReitzAnisotropic
                        | MndfModel::TrowbridgeReitzAnisotropic => {
                            ReflectionModelFamily::Microfacet(
                                MicrofacetModelFamily::TrowbridgeReitz,
                            )
                        }
                        #[cfg(not(feature = "scaled-ndf-fitting"))]
                        MndfModel::BeckmannSpizzichino
                        | MndfModel::BeckmannSpizzichinoAnisotropic => {
                            ReflectionModelFamily::Microfacet(
                                MicrofacetModelFamily::BeckmannSpizzichino,
                            )
                        }
                        #[cfg(not(feature = "scaled-ndf-fitting"))]
                        MndfModel::TrowbridgeReitz | MndfModel::TrowbridgeReitzAnisotropic => {
                            ReflectionModelFamily::Microfacet(
                                MicrofacetModelFamily::TrowbridgeReitz,
                            )
                        }
                    };
                    #[cfg(feature = "scaled-ndf-fitting")]
                    let scaled = match self.selected_model {
                        MndfModel::ScaledBeckmannSpizzichino
                        | MndfModel::ScaledTrowbridgeReitz
                        | MndfModel::ScaledBeckmannSpizzichinoAnisotropic
                        | MndfModel::ScaledTrowbridgeReitzAnisotropic => true,
                        _ => false,
                    };
                    log::debug!(
                        "Fitting with model {:?}, isotropic {}, mode {:?}",
                        family,
                        self.selected_model.is_isotropic(),
                        self.mode
                    );
                    event_loop
                        .send_event(
                            #[cfg(feature = "scaled-ndf-fitting")]
                            {
                                VgonioEvent::Fitting {
                                    kind: MeasurementKind::Mndf,
                                    family,
                                    data,
                                    mode: Some(self.mode),
                                    isotropic: self.selected_model.is_isotropic(),
                                    scaled,
                                }
                            },
                            #[cfg(not(feature = "scaled-ndf-fitting"))]
                            {
                                VgonioEvent::Fitting {
                                    kind: MeasurementKind::Mndf,
                                    family,
                                    data,
                                    mode: Some(self.mode),
                                    isotropic: self.selected_model.is_isotropic(),
                                }
                            },
                        )
                        .unwrap();
                }

                ui.label("Fitted models: ");
                if self.fitted.is_empty() {
                    ui.label("None");
                } else {
                    for (model, mode, _) in &self.fitted {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::from(format!(
                                    "{}{}",
                                    model.name(),
                                    mode.as_suffix_str()
                                ))
                                .text_style(egui::TextStyle::Monospace),
                            );
                            ui.label(if model.is_isotropic() {
                                #[cfg(feature = "scaled-ndf-fitting")]
                                {
                                    match model.scale() {
                                        None => egui::RichText::from(format!(
                                            "α = {:.4}",
                                            model.params()[0],
                                        )),
                                        Some(scale) => egui::RichText::from(format!(
                                            "α = {:.4}, scale = {:.4}",
                                            model.params()[0],
                                            scale
                                        ))
                                        .text_style(egui::TextStyle::Button),
                                    }
                                }
                                #[cfg(not(feature = "scaled-ndf-fitting"))]
                                {
                                    egui::RichText::from(format!("α = {:.4}", model.params()[0],))
                                }
                            } else {
                                #[cfg(feature = "scaled-ndf-fitting")]
                                {
                                    match model.scale() {
                                        None => egui::RichText::from(format!(
                                            "αx = {:.4}, αy = {:.4}",
                                            model.params()[0],
                                            model.params()[1]
                                        )),
                                        Some(scale) => egui::RichText::from(format!(
                                            "αx = {:.4}, αy = {:.4}, scale = {:.4}",
                                            model.params()[0],
                                            model.params()[1],
                                            scale
                                        ))
                                        .text_style(egui::TextStyle::Button),
                                    }
                                }
                                #[cfg(not(feature = "scaled-ndf-fitting"))]
                                {
                                    egui::RichText::from(format!(
                                        "αx = {:.4}, αy = {:.4}",
                                        model.params()[0],
                                        model.params()[1]
                                    ))
                                }
                            })
                        });
                    }
                }
            });
    }
}

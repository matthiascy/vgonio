mod msf;
mod ndf;
mod sdf;

pub use msf::*;
pub use ndf::*;

#[cfg(feature = "fitting")]
use crate::fitting::FittedModel;

use crate::{
    app::{
        cache::{Cache, Handle, RawCache},
        gui::{
            data::PropertyData,
            docking::{Dockable, WidgetKind},
            event::EventLoopProxy,
            plotter::sdf::SlopeDistributionExtra,
            widgets::{AngleKnob, AngleKnobWinding},
        },
    },
    measure::{
        mfd::{MeasuredMsfData, MeasuredNdfData},
        Measurement,
    },
};
use base::{
    math,
    range::RangeByStepSizeInclusive,
    units::{deg, rad, Radians},
    MeasurementKind,
};
use bxdf::distro::{BeckmannDistribution, MicrofacetDistribution, TrowbridgeReitzDistribution};
use egui::{Context, Response, Ui, WidgetText};
use egui_plot::*;
use std::{
    any::Any,
    ops::{Deref, RangeInclusive},
    sync::{Arc, RwLock},
};
use uuid::Uuid;

const LINE_COLORS: [egui::Color32; 16] = [
    egui::Color32::from_rgb(254, 128, 127),
    egui::Color32::from_rgb(35, 152, 13),
    egui::Color32::from_rgb(119, 52, 147),
    egui::Color32::from_rgb(177, 230, 50),
    egui::Color32::from_rgb(26, 101, 135),
    egui::Color32::from_rgb(72, 182, 234),
    egui::Color32::from_rgb(37, 80, 38),
    egui::Color32::from_rgb(57, 242, 122),
    egui::Color32::from_rgb(156, 64, 80),
    egui::Color32::from_rgb(131, 217, 150),
    egui::Color32::from_rgb(237, 75, 4),
    egui::Color32::from_rgb(241, 203, 213),
    egui::Color32::from_rgb(109, 76, 43),
    egui::Color32::from_rgb(226, 198, 39),
    egui::Color32::from_rgb(210, 14, 77),
    egui::Color32::from_rgb(108, 142, 69),
];

pub trait PlottingWidget {
    fn uuid(&self) -> Uuid;

    fn name(&self) -> &str;

    fn ui(&mut self, ui: &mut Ui);

    fn show(&mut self, ctx: &Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::new([false, true])
                    .min_scrolled_height(640.0)
                    .show(ui, |ui| {
                        self.ui(ui);
                    });
            });
    }

    fn measurement_data_handle(&self) -> Handle<Measurement>;

    fn measurement_data_kind(&self) -> MeasurementKind;
}

/// Trait for extra data to be used by the plotting inspector.
pub trait VariantData {
    /// Initialise the extra data.
    fn pre_process(&mut self, data: Handle<Measurement>, cache: &RawCache);

    /// Returns the curve to be displayed.
    fn current_curve(&self) -> Option<&Curve>;

    /// Returns the scale factor for the curve.
    fn scale_factor(&self) -> f32 { 1.0 }

    #[cfg(feature = "fitting")]
    /// Updates the fitted curves according to the given fitted models.
    fn update_fitted_curves(&mut self, fitted: &[FittedModel]);

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn ui(
        &mut self,
        ui: &mut Ui,
        event_loop: &EventLoopProxy,
        data: Handle<Measurement>,
        cache: &Cache,
    );
}

pub struct PlotInspector {
    /// Unique ID for the plot widget.
    uuid: Uuid,
    /// Name for the plot.
    name: String,
    /// The handle to the data to be plotted.
    data_handle: Handle<Measurement>,
    /// Cache of the application.
    cache: Cache,
    /// Inspector properties data might be used by the plot.
    props: Arc<RwLock<PropertyData>>,
    /// The legend to be displayed
    legend: Legend,
    /// Variant data for the plot depending on the type of the measurement data.
    variant: Option<Box<dyn VariantData>>,
    new_curve_kind: CurveKind,
    // (model, uuid)
    ndf_models: Vec<(Box<dyn MicrofacetDistribution<Params = [f64; 2]>>, Uuid)>,
    // mmsf_models: Vec<Box<dyn MicrofacetGeometricalAttenuationModel>>, TODO: implement
    /// The event loop.
    event_loop: EventLoopProxy,

    azimuth_m: Radians,
}

/// Indicates which plot is currently being displayed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum BsdfPlotMode {
    /// The BSDF is plotted as a function of the incident direction wi (θ,φ) and
    /// the azimuthal angle of the outgoing direction wo (φ).
    Slice2D,
    /// The BSDF is plotted as a function of the incident direction wi (θ,φ).
    Slice3D,
}

pub struct BsdfPlotExtraData {
    azimuth_i: Radians,
    zenith_i: Radians,
    azimuth_o: Radians,
    mode: BsdfPlotMode,
    // view_id: egui::TextureId,
    changed: bool,
}

impl BsdfPlotExtraData {
    pub fn new(/*view_id: egui::TextureId*/) -> Self {
        Self {
            azimuth_i: rad!(0.0),
            zenith_i: rad!(0.0),
            azimuth_o: rad!(0.0),
            mode: BsdfPlotMode::Slice2D,
            // view_id,
            changed: true,
        }
    }
}

/// A curve to be plotted.
pub struct Curve {
    /// The data to be plotted.
    pub points: Vec<[f64; 2]>,
    /// Maximum value of the data.
    pub max_val: [f64; 2],
}

impl Curve {
    /// Creates a curve from a ADF or MSF data.
    ///
    /// # Arguments
    ///
    /// * `first_part` - The first part of the data, which is the measured data
    /// for the selected azimuthal angle.
    /// * `opposite` - The second part of the data, which is the measured data
    /// for the opposite azimuthal angle. If `None`, the curve is created from
    /// the first part only.
    fn from_adf_or_msf_data(
        first_part: impl Iterator<Item = [f64; 2]>,
        opposite: Option<&[f32]>,
        zenith_range: &RangeByStepSizeInclusive<Radians>,
    ) -> Curve {
        let zenith_step_count = zenith_range.step_count_wrapped();
        match opposite {
            None => Curve::from(first_part),
            Some(opposite) => {
                let second_part_points = opposite
                    .iter()
                    .rev()
                    .zip(zenith_range.values_rev().map(|x| -x.as_f64()))
                    .take(zenith_step_count - 1)
                    .map(|(y, x)| [x, *y as f64]);
                Curve::from(second_part_points.chain(first_part))
            }
        }
    }
}

impl<I> From<I> for Curve
where
    I: Iterator<Item = [f64; 2]>,
{
    fn from(values: I) -> Self {
        let points: Vec<[f64; 2]> = values.collect();
        let max_val = points.iter().fold([0.01; 2], |[max_x, max_y], [x, y]| {
            [x.abs().max(max_x), y.abs().max(max_y)]
        });
        Self { points, max_val }
    }
}

impl AsRef<[[f64; 2]]> for Curve {
    fn as_ref(&self) -> &[[f64; 2]] { &self.points }
}

impl Deref for Curve {
    type Target = Vec<[f64; 2]>;

    fn deref(&self) -> &Self::Target { &self.points }
}

impl VariantData for BsdfPlotExtraData {
    fn pre_process(&mut self, _data: Handle<Measurement>, _cache: &RawCache) {
        // TODO: pre-process data
    }

    fn current_curve(&self) -> Option<&Curve> { todo!("bsdf current curve") }

    #[cfg(feature = "fitting")]
    fn update_fitted_curves(&mut self, _fitted: &[FittedModel]) { todo!() }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn ui(
        &mut self,
        _ui: &mut Ui,
        _event_loop: &EventLoopProxy,
        _data: Handle<Measurement>,
        _cache: &Cache,
    ) {
        todo!()
    }
}

impl PlotInspector {
    /// Creates a new inspector for a microfacet area distribution function.
    pub fn new_adf(
        name: String,
        data: Handle<Measurement>,
        cache: Cache,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let mut extra = AreaDistributionExtra::default();
        cache.read(|cache| {
            extra.pre_process(data, cache);
        });
        Self::new_inner(name, data, Some(Box::new(extra)), cache, props, event_loop)
    }

    /// Creates a new inspector for a microfacet masking-shadowing function.
    pub fn new_msf(
        name: String,
        data: Handle<Measurement>,
        cache: Cache,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let mut extra = MaskingShadowingExtra::default();
        cache.read(|cache| {
            extra.pre_process(data, cache);
        });
        Self::new_inner(
            name,
            data,
            Some(Box::new(extra)),
            cache.clone(),
            props,
            event_loop,
        )
    }

    /// Creates a new inspector for a bidirectional scattering distribution
    /// function.
    pub fn new_bsdf(
        name: String,
        data: Handle<Measurement>,
        cache: Cache,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let extra = BsdfPlotExtraData::new(/*view_id*/);
        // extra.pre_process(data, cache);
        Self::new_inner(name, data, Some(Box::new(extra)), cache, props, event_loop)
    }

    pub fn new_sdf(
        name: String,
        data: Handle<Measurement>,
        cache: Cache,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let mut extra = SlopeDistributionExtra::default();
        cache.read(|cache| {
            extra.pre_process(data, cache);
        });
        Self::new_inner(name, data, Some(Box::new(extra)), cache, props, event_loop)
    }

    /// Creates a new inspector with data to be plotted.
    pub fn new<S: Into<String>>(
        name: S,
        cache: Cache,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        Self {
            name: name.into(),
            data_handle: Handle::invalid(),
            cache,
            props,
            legend: Legend::default()
                .text_style(egui::TextStyle::Monospace)
                .background_alpha(1.0)
                .position(Corner::RightTop),
            event_loop,
            variant: None,
            new_curve_kind: CurveKind::None,
            ndf_models: vec![],
            uuid: Uuid::new_v4(),
            // mmsf_models: vec![],
            azimuth_m: rad!(0.0),
        }
    }

    fn new_inner(
        name: String,
        data: Handle<Measurement>,
        extra: Option<Box<dyn VariantData>>,
        cache: Cache,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        Self {
            uuid: Uuid::new_v4(),
            name,
            data_handle: data,
            cache,
            props,
            legend: Legend::default()
                .text_style(egui::TextStyle::Monospace)
                .background_alpha(1.0)
                .position(Corner::RightTop),
            variant: extra,
            new_curve_kind: CurveKind::None,
            ndf_models: vec![],
            event_loop,
            azimuth_m: rad!(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurveKind {
    None,
    TrowbridgeReitzNdf,
    BeckmannNdf,
}

/// Inspector for microfacet distribution function.
pub struct MicrofacetDistributionPlotter {}
pub struct BxdfPlotter {}

impl PlottingWidget for PlotInspector {
    fn uuid(&self) -> Uuid { self.uuid }

    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui) {
        {
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_label("kind")
                    .selected_text(format!("{:?}", self.new_curve_kind))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.new_curve_kind,
                            CurveKind::TrowbridgeReitzNdf,
                            "Trowbridge-Reitz",
                        );
                        ui.selectable_value(
                            &mut self.new_curve_kind,
                            CurveKind::BeckmannNdf,
                            "Beckmann",
                        );
                    });
                if ui.button("Graph").clicked() {
                    match self.new_curve_kind {
                        CurveKind::None => {}
                        CurveKind::BeckmannNdf => {
                            self.ndf_models.push((
                                Box::new(BeckmannDistribution::new(0.5, 0.5)),
                                Uuid::new_v4(),
                            ));
                        }
                        CurveKind::TrowbridgeReitzNdf => self.ndf_models.push((
                            Box::new(TrowbridgeReitzDistribution::new(0.5, 0.5)),
                            Uuid::new_v4(),
                        )),
                    }
                }
            });
        }

        {
            let mut to_be_removed = vec![];
            for (i, (model, uuid)) in self.ndf_models.iter_mut().enumerate() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!(
                        "Model: {}#{}",
                        model.kind().to_str(),
                        &uuid.to_string().as_str()[..6]
                    ));
                    let [mut alpha_x, mut alpha_y] = model.params();
                    let alpha_x_changed = ui
                        .add(
                            egui::Slider::new(&mut alpha_x, 0.00001..=2.0)
                                .drag_value_speed(0.0001)
                                .text("αx"),
                        )
                        .changed();
                    let alpha_y_changed = ui
                        .add(
                            egui::Slider::new(&mut alpha_y, 0.00001..=2.0)
                                .drag_value_speed(0.0001)
                                .text("αy"),
                        )
                        .changed();

                    let mut alpha = alpha_x;
                    let alpha_changed = ui
                        .add(
                            egui::Slider::new(&mut alpha, 0.00001..=2.0)
                                .drag_value_speed(0.0001)
                                .text("α"),
                        )
                        .changed();

                    if alpha_x_changed || alpha_y_changed {
                        model.set_params(&[alpha_x, alpha_y])
                    }

                    if alpha_changed {
                        model.set_params(&[alpha, alpha]);
                    }

                    if ui.button("Remove").clicked() {
                        to_be_removed.push(i);
                    }
                });
            }
            for i in to_be_removed {
                self.ndf_models.remove(i);
            }
        }

        if !self.data_handle.is_valid() {
            ui.label("No data selected!");
            let plot = Plot::new("madf_plot")
                .legend(self.legend.clone())
                .center_x_axis(true)
                .sharp_grid_lines(true)
                .x_grid_spacer(adf_msf_x_angle_spacer)
                .y_grid_spacer(ndf_msf_y_uniform_spacer)
                .x_axis_formatter(|x, _, _| format!("{:.2}°", x.to_degrees()));
            let azimuth_range =
                RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TAU, deg!(1.0).to_radians());
            angle_knob(
                ui,
                true,
                &mut self.azimuth_m,
                azimuth_range.map(|x| x.value()).range_bound_inclusive(),
                azimuth_range.step_size,
                48.0,
                |v| format!("φ = {:>6.2}°", v.to_degrees()),
            );
            plot.show(ui, |plot_ui| {
                for (i, (model, uuid)) in self.ndf_models.iter().enumerate() {
                    let points: Vec<_> = (0..=180)
                        .map(|x| {
                            let theta = x as f64 * std::f64::consts::PI / 180.0
                                - std::f64::consts::PI * 0.5;
                            let current_phi = self.azimuth_m.wrap_to_tau();
                            let phi = if theta > 0.0 {
                                current_phi
                            } else {
                                current_phi.opposite()
                            };
                            let value = model.eval_ndf(theta.cos(), phi.cos() as f64);
                            [theta, value]
                        })
                        .collect();
                    plot_ui.line(
                        Line::new(points)
                            .stroke(egui::epaint::Stroke::new(2.0, LINE_COLORS[i]))
                            .name(format!(
                                "{:?}{:?}#{}",
                                model.isotropy(),
                                model.kind(),
                                &uuid.to_string().as_str()[..6]
                            )),
                    );
                }
            });
        } else {
            {
                let props = self.props.read().unwrap();
                let prop = props.measured.get(&self.data_handle).unwrap();
                #[cfg(feature = "fitting")]
                if let Some(extra) = &mut self.variant {
                    extra.update_fitted_curves(prop.fitted.as_ref());
                }
            }

            match self.measurement_data_kind() {
                MeasurementKind::Bsdf => {
                    todo!("bsdf plot");
                }
                MeasurementKind::Ndf => {
                    if let Some(variant) = &mut self.variant {
                        let zenith = self.cache.read(|cache| {
                            let measurement = cache.get_measurement(self.data_handle).unwrap();
                            measurement
                                .measured
                                .downcast_ref::<MeasuredNdfData>()
                                .unwrap()
                                .zenith_range()
                                .unwrap()
                        });
                        let zenith_bin_width_rad = zenith.step_size.as_f32();
                        variant.ui(ui, &self.event_loop, self.data_handle, &self.cache);
                        if let Some(curve) = variant.current_curve() {
                            let aspect = curve.max_val[0] / curve.max_val[1];
                            let plot = Plot::new("mndf_plot")
                                .legend(self.legend.clone())
                                .data_aspect(aspect as f32 * 0.25)
                                .center_x_axis(true)
                                .sharp_grid_lines(true)
                                .x_grid_spacer(adf_msf_x_angle_spacer)
                                .y_grid_spacer(ndf_msf_y_uniform_spacer)
                                .x_axis_formatter(|x, _, _| format!("{:.2}°", x.to_degrees()))
                                .coordinates_formatter(
                                    Corner::LeftBottom,
                                    CoordinatesFormatter::new(move |p, _| {
                                        let n_bin =
                                            (p.x / zenith_bin_width_rad as f64 + 0.5).floor();
                                        let bin = n_bin * zenith_bin_width_rad.to_degrees() as f64;
                                        let half_bin_width =
                                            zenith_bin_width_rad.to_degrees() as f64 * 0.5;
                                        format!(
                                            "φ: {:.2}° θ: {:.2}°±{half_bin_width:.2}°\nValue: \
                                             {:.2} sr⁻¹",
                                            0.0, bin, p.y
                                        )
                                    }),
                                );
                            plot.show(ui, |plot_ui| {
                                let scale = variant.scale_factor() as f64;
                                let mut color_idx_base = 0;
                                plot_ui.line(
                                    Line::new(
                                        curve
                                            .points
                                            .iter()
                                            .map(|[x, y]| [*x, *y * scale])
                                            .collect::<Vec<_>>(),
                                    )
                                    .stroke(egui::epaint::Stroke::new(
                                        2.0,
                                        LINE_COLORS[color_idx_base],
                                    ))
                                    .name(format!("Measured - ADF (x {:.4})", scale)),
                                );
                                color_idx_base += 1;
                                plot_ui.line(
                                    Line::new(
                                        curve
                                            .points
                                            .iter()
                                            .map(|[x, y]| [*x, *y])
                                            .collect::<Vec<_>>(),
                                    )
                                    .stroke(egui::epaint::Stroke::new(
                                        2.0,
                                        LINE_COLORS[color_idx_base],
                                    ))
                                    .name("Measured - ADF"),
                                );
                                color_idx_base += 1;
                                let extra = variant
                                    .as_any()
                                    .downcast_ref::<AreaDistributionExtra>()
                                    .unwrap();
                                if extra.show_slope_distribution {
                                    plot_ui.line(
                                        Line::new(
                                            curve
                                                .points
                                                .iter()
                                                .map(|[x, y]| [*x, *y * x.cos().powi(4) * scale])
                                                .collect::<Vec<_>>(),
                                        )
                                        .stroke(egui::epaint::Stroke::new(
                                            2.0,
                                            LINE_COLORS[color_idx_base],
                                        ))
                                        .name("Converted - SDF"),
                                    );
                                }
                                color_idx_base += 1;
                                {
                                    for (i, (model, scale, curves)) in
                                        extra.fitted.iter().enumerate()
                                    {
                                        plot_ui.line(
                                            Line::new(
                                                curves[extra.current_azimuth_idx()].points.clone(),
                                            )
                                            .stroke(egui::epaint::Stroke::new(
                                                2.0,
                                                LINE_COLORS
                                                    [(i + color_idx_base) % LINE_COLORS.len()],
                                            ))
                                            .name(
                                                format!(
                                                    "{} (x {:.4}) {}",
                                                    model.kind().to_str(),
                                                    scale,
                                                    model.isotropy().to_string().to_lowercase(),
                                                ),
                                            ),
                                        )
                                    }
                                    color_idx_base += extra.fitted.len() + 1;
                                }
                                for (i, (model, uuid)) in self.ndf_models.iter().enumerate() {
                                    let points: Vec<_> = (0..=180)
                                        .map(|x| {
                                            let theta = x as f64 * std::f64::consts::PI / 180.0
                                                - std::f64::consts::PI * 0.5;
                                            let current_phi = extra.azimuth_m.wrap_to_tau();
                                            let phi = if theta > 0.0 {
                                                current_phi
                                            } else {
                                                current_phi.opposite()
                                            };
                                            let value =
                                                model.eval_ndf(theta.cos(), phi.cos() as f64);
                                            [theta, value]
                                        })
                                        .collect();
                                    plot_ui.line(
                                        Line::new(points)
                                            .stroke(egui::epaint::Stroke::new(
                                                2.0,
                                                LINE_COLORS
                                                    [(i + color_idx_base) % LINE_COLORS.len()],
                                            ))
                                            .name(format!(
                                                "{}#{}",
                                                model.kind().to_str(),
                                                &uuid.to_string().as_str()[..6]
                                            )),
                                    );
                                }
                            });
                        }
                    }
                }
                MeasurementKind::Msf => {
                    if let Some(extra) = &mut self.variant {
                        let variant = extra
                            .as_any_mut()
                            .downcast_mut::<MaskingShadowingExtra>()
                            .unwrap();
                        let zenith = self.cache.read(|cache| {
                            let measurement = cache.get_measurement(self.data_handle).unwrap();
                            measurement
                                .measured
                                .downcast_ref::<MeasuredMsfData>()
                                .unwrap()
                                .params
                                .zenith
                        });
                        let zenith_bin_width_rad = zenith.step_size.value();
                        variant.ui(ui, &self.event_loop, self.data_handle, &self.cache);
                        if let Some(curve) = variant.current_curve() {
                            let aspect = curve.max_val[0] / curve.max_val[1];
                            let plot = Plot::new("plot_msf")
                                .legend(self.legend.clone())
                                .data_aspect(aspect as f32)
                                //.clamp_grid(true)
                                .center_x_axis(true)
                                .sharp_grid_lines(true)
                                .x_grid_spacer(adf_msf_x_angle_spacer)
                                .x_axis_formatter(|x, _, _| format!("{:.2}°", x.to_degrees()))
                                .coordinates_formatter(
                                    Corner::LeftBottom,
                                    CoordinatesFormatter::new(move |p, _| {
                                        let n_bin =
                                            (p.x / zenith_bin_width_rad as f64 + 0.5).floor();
                                        let bin = n_bin * zenith_bin_width_rad.to_degrees() as f64;
                                        let half_bin_width =
                                            zenith_bin_width_rad.to_degrees() as f64 * 0.5;
                                        format!(
                                            "φ: {:.2}° θ: {:.2}°±{half_bin_width:.2}°\nValue: \
                                             {:.2}",
                                            0.0, bin, p.y
                                        )
                                    }),
                                );

                            plot.show(ui, |plot_ui| {
                                plot_ui.line(
                                    Line::new(curve.points.clone())
                                        .stroke(egui::epaint::Stroke::new(
                                            2.0,
                                            egui::Color32::LIGHT_RED,
                                        ))
                                        .name("Microfacet masking shadowing"),
                                );
                                #[cfg(feature = "fitting")]
                                {
                                    let concrete_extra = variant
                                        .as_any()
                                        .downcast_ref::<MaskingShadowingExtra>()
                                        .unwrap();
                                    for (i, (model, curve)) in
                                        concrete_extra.fitted.iter().enumerate()
                                    {
                                        plot_ui.line(
                                            Line::new(curve.points.clone())
                                                .stroke(egui::epaint::Stroke::new(
                                                    2.0,
                                                    LINE_COLORS[(i + 1) % LINE_COLORS.len()],
                                                ))
                                                .name(model.kind().to_str()),
                                        );
                                    }
                                }
                            });
                        }
                    }
                }
                MeasurementKind::Sdf => {
                    if let Some(variant) = &mut self.variant {
                        variant.ui(ui, &self.event_loop, self.data_handle, &self.cache);
                        let extra = variant
                            .as_any()
                            .downcast_ref::<SlopeDistributionExtra>()
                            .unwrap();
                        let zen_step_size = extra.zen_range.step_size;
                        if let Some(curve) = variant.current_curve() {
                            let aspect = curve.max_val[0] / curve.max_val[1];
                            let plot = Plot::new("adf_plot")
                                .legend(self.legend.clone())
                                .data_aspect(aspect as f32 * 0.25)
                                .center_x_axis(true)
                                .sharp_grid_lines(true)
                                .x_grid_spacer(adf_msf_x_angle_spacer)
                                .y_grid_spacer(ndf_msf_y_uniform_spacer)
                                .x_axis_formatter(|x, _, _| format!("{:.2}°", x.to_degrees()))
                                .coordinates_formatter(
                                    Corner::LeftBottom,
                                    CoordinatesFormatter::new(move |p, _| {
                                        let n_bin = (p.x / zen_step_size.as_f64() + 0.5).floor();
                                        let bin = n_bin * zen_step_size.to_degrees().as_f64();
                                        let half_bin_width =
                                            zen_step_size.to_degrees().as_f64() * 0.5;
                                        format!(
                                            "φ: {:.2}° θ: {:.2}°±{half_bin_width:.2}°\nValue: \
                                             {:.2} sr⁻¹",
                                            0.0, bin, p.y
                                        )
                                    }),
                                );
                            plot.show(ui, |plot_ui| {
                                if extra.apply_jacobian {
                                    plot_ui.line(
                                        Line::new(
                                            curve
                                                .points
                                                .iter()
                                                .map(|[x, y]| {
                                                    [*x, *y * math::rcp_f64(x.cos().powi(4))]
                                                })
                                                .collect::<Vec<_>>(),
                                        )
                                        .stroke(egui::epaint::Stroke::new(2.0, LINE_COLORS[1]))
                                        .name("Measured - ADF"),
                                    );
                                } else {
                                    plot_ui.line(
                                        Line::new(
                                            curve
                                                .points
                                                .iter()
                                                .map(|[x, y]| [*x, *y])
                                                .collect::<Vec<_>>(),
                                        )
                                        .stroke(egui::epaint::Stroke::new(2.0, LINE_COLORS[1]))
                                        .name("Measured - ADF"),
                                    );
                                }
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn measurement_data_handle(&self) -> Handle<Measurement> { self.data_handle }

    fn measurement_data_kind(&self) -> MeasurementKind {
        match self.data_handle.variant_id() {
            0 => MeasurementKind::Bsdf,
            1 => MeasurementKind::Ndf,
            2 => MeasurementKind::Msf,
            3 => MeasurementKind::Sdf,
            _ => {
                unreachable!("Invalid measurement data handle variant id")
            }
        }
    }
}

impl Dockable for PlotInspector {
    fn kind(&self) -> WidgetKind { WidgetKind::Plotting }

    fn title(&self) -> WidgetText { self.name().into() }

    fn uuid(&self) -> Uuid { self.uuid }

    fn ui(&mut self, ui: &mut Ui) { PlottingWidget::ui(self, ui); }
}

/// Calculates the ticks for x axis of the ADF or MSF plot.
fn adf_msf_x_angle_spacer(input: GridInput) -> Vec<GridMark> {
    let mut marks = vec![];
    let (min, max) = input.bounds;
    let min = min.floor().to_degrees() as i32;
    let max = max.ceil().to_degrees() as i32;
    for i in min..=max {
        let step_size = if i % 30 == 0 {
            30.0f64.to_radians()
        } else if i % 10 == 0 {
            10.0f64.to_radians()
        } else {
            continue;
        };
        marks.push(GridMark {
            value: (i as f64).to_radians(),
            step_size,
        });
    }
    marks
}

fn ndf_msf_y_uniform_spacer(input: GridInput) -> Vec<GridMark> {
    let mut marks = vec![];
    let (min, max) = input.bounds;
    let min = min.floor();
    let max = max.ceil();
    let step_size = (max - min.max(0.0)) / 10.0;
    for i in 0..10 {
        marks.push(GridMark {
            value: i as f64 * step_size,
            step_size,
        });
    }
    marks
}

fn angle_knob(
    ui: &mut Ui,
    interactive: bool,
    angle: &mut Radians,
    range: RangeInclusive<f32>,
    snap: Radians,
    diameter: f32,
    formatter: impl Fn(f32) -> String,
) -> Response {
    let response = ui.add(
        AngleKnob::new(angle)
            .interactive(interactive)
            .min(Some((*range.start()).into()))
            .max(Some((*range.end()).into()))
            .snap(Some(snap))
            .winding(AngleKnobWinding::CounterClockwise)
            .diameter(diameter)
            .axis_count((Radians::TAU / snap).ceil() as u32),
    );
    ui.label(formatter(angle.value()));
    response
}

#[cfg(debug_assertions)]
fn debug_print_angle_pair(
    initial: Radians,
    range: &RangeByStepSizeInclusive<Radians>,
    ui: &mut Ui,
    text: &str,
) {
    if ui.button(text).clicked() {
        let opposite = initial.opposite();
        println!(
            "initial = {}, index = {} | opposite = {}, index = {}",
            initial.to_degrees(),
            range.index_of(initial),
            opposite.to_degrees(),
            range.index_of(opposite),
        );
    }
}

#[cfg(debug_assertions)]
fn debug_print_angle(
    initial: Radians,
    range: &RangeByStepSizeInclusive<Radians>,
    ui: &mut Ui,
    text: &str,
) {
    if ui.button(text).clicked() {
        let initial = initial.wrap_to_tau();
        println!(
            "angle = {}, index = {}",
            initial.to_degrees(),
            range.index_of(initial),
        );
    }
}

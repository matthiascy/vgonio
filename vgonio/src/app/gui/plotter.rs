mod adf;
mod msf;

pub use adf::*;
pub use msf::*;

use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{
            data::PropertyData,
            docking::{Dockable, WidgetKind},
            event::EventLoopProxy,
            widgets::{AngleKnob, AngleKnobWinding},
        },
    },
    fitting::FittedModel,
    measure::{
        measurement::{MeasurementData, MeasurementKind},
        CollectorScheme,
    },
    RangeByStepSizeInclusive, SphericalPartition,
};
use egui::{plot::*, Align, Context, Response, Ui, Vec2, WidgetText};
use std::{
    any::Any,
    ops::{Deref, RangeInclusive},
    sync::{Arc, RwLock},
};
use uuid::Uuid;
use vgbxdf::{BeckmannDistribution, MicrofacetDistributionModel, TrowbridgeReitzDistribution};
use vgcore::{
    math,
    math::{Handedness, Vec3},
    units::{deg, rad, Radians},
    Isotropy,
};

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

    fn measurement_data_handle(&self) -> Handle<MeasurementData>;

    fn measurement_data_kind(&self) -> MeasurementKind;
}

/// Trait for extra data to be used by the plotting inspector.
pub trait VariantData {
    /// Initialise the extra data.
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache);

    /// Returns the curve to be displayed.
    fn current_curve(&self) -> Option<&Curve>;

    /// Updates the fitted curves according to the given fitted models.
    fn update_fitted_curves(&mut self, fitted: &[FittedModel]);

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn ui(&mut self, ui: &mut Ui, event_loop: &EventLoopProxy, data: Handle<MeasurementData>);
}

pub struct PlotInspector {
    /// Unique ID for the plot widget.
    uuid: Uuid,
    /// Name for the plot.
    name: String,
    /// The handle to the data to be plotted.
    data_handle: Handle<MeasurementData>,
    /// Cache of the application.
    cache: Arc<RwLock<Cache>>,
    /// Inspector properties data might used by the plot.
    props: Arc<RwLock<PropertyData>>,
    /// The legend to be displayed
    legend: Legend,
    /// Variant data for the plot depending on the type of the measurement data.
    variant: Option<Box<dyn VariantData>>,
    new_curve_kind: CurveKind,
    // (model, uuid)
    adf_models: Vec<(Box<dyn MicrofacetDistributionModel>, Uuid)>,
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
    /// Creates a curve from a MADF or MMSF data.
    ///
    /// # Arguments
    ///
    /// * `first_part` - The first part of the data, which is the measured data
    /// for the selected azimuthal angle.
    /// * `opposite` - The second part of the data, which is the measured data
    /// for the opposite azimuthal angle. If `None`, the curve is created from
    /// the first part only.
    fn from_mndf_or_mmsf_data(
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
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache) {
        // TODO: pre-process data
    }

    fn current_curve(&self) -> Option<&Curve> { todo!("bsdf current curve") }

    fn update_fitted_curves(&mut self, fitted: &[FittedModel]) { todo!() }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn ui(&mut self, ui: &mut Ui, event_loop: &EventLoopProxy, data: Handle<MeasurementData>) {
        todo!()
    }
}

impl PlotInspector {
    /// Creates a new inspector for a microfacet area distribution function.
    pub fn new_area_distrib_plot(
        name: String,
        data: Handle<MeasurementData>,
        cache: Arc<RwLock<Cache>>,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let mut extra = AreaDistributionExtra::default();
        {
            let cache = cache.read().unwrap();
            extra.pre_process(data, &cache);
        }
        Self::new_inner(name, data, Some(Box::new(extra)), cache, props, event_loop)
    }

    /// Creates a new inspector for a microfacet masking-shadowing function.
    pub fn new_mmsf(
        name: String,
        data: Handle<MeasurementData>,
        cache: Arc<RwLock<Cache>>,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let mut extra = MaskingShadowingExtra::default();
        {
            let cache = cache.read().unwrap();
            extra.pre_process(data, &cache);
        }
        Self::new_inner(name, data, Some(Box::new(extra)), cache, props, event_loop)
    }

    /// Creates a new inspector for a bidirectional scattering distribution
    /// function.
    pub fn new_bsdf(
        name: String,
        data: Handle<MeasurementData>,
        cache: Arc<RwLock<Cache>>,
        props: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let extra = BsdfPlotExtraData::new(/*view_id*/);
        // extra.pre_process(data, cache);
        Self::new_inner(name, data, Some(Box::new(extra)), cache, props, event_loop)
    }

    /// Creates a new inspector with data to be plotted.
    pub fn new<S: Into<String>>(
        name: S,
        cache: Arc<RwLock<Cache>>,
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
            adf_models: vec![],
            uuid: Uuid::new_v4(),
            // mmsf_models: vec![],
            azimuth_m: rad!(0.0),
        }
    }

    fn new_inner(
        name: String,
        data: Handle<MeasurementData>,
        extra: Option<Box<dyn VariantData>>,
        cache: Arc<RwLock<Cache>>,
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
            adf_models: vec![],
            // mmsf_models: vec![],
            event_loop,
            azimuth_m: rad!(0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurveKind {
    None,
    TrowbridgeReitzADF,
    BeckmannSpizzichinoADF,
}

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
                            CurveKind::TrowbridgeReitzADF,
                            "Trowbridge-Reitz",
                        );
                        ui.selectable_value(
                            &mut self.new_curve_kind,
                            CurveKind::BeckmannSpizzichinoADF,
                            "Beckmann-Spizzichino",
                        );
                    });
                if ui.button("Graph").clicked() {
                    match self.new_curve_kind {
                        CurveKind::None => {}
                        CurveKind::BeckmannSpizzichinoADF => {
                            self.adf_models.push((
                                Box::new(BeckmannDistribution::new(0.5, 0.5)),
                                Uuid::new_v4(),
                            ));
                        }
                        CurveKind::TrowbridgeReitzADF => self.adf_models.push((
                            Box::new(TrowbridgeReitzDistribution::new(0.5, 0.5)),
                            Uuid::new_v4(),
                        )),
                    }
                }
            });
        }

        {
            let mut to_be_removed = vec![];
            for (i, (model, uuid)) in self.adf_models.iter_mut().enumerate() {
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!(
                        "Model: {}#{}",
                        model.kind().to_str(),
                        &uuid.to_string().as_str()[..6]
                    ));
                    let mut alpha_x = model.alpha_x();
                    let mut alpha_y = model.alpha_y();
                    let alpha_x_changed = ui
                        .add(egui::Slider::new(&mut alpha_x, 0.00001..=10.0).text("alpha_x"))
                        .changed();
                    let alpha_y_changed = ui
                        .add(egui::Slider::new(&mut alpha_y, 0.00001..=10.0).text("alpha_y"))
                        .changed();

                    if alpha_x_changed || alpha_y_changed {
                        model.set_alpha_x(alpha_x);
                        model.set_alpha_y(alpha_y);
                    }
                    if ui.button("Remove").clicked() {
                        to_be_removed.push(i);
                    }
                });
            }
            for i in to_be_removed {
                self.adf_models.remove(i);
            }
        }

        if !self.data_handle.is_valid() {
            ui.label("No data selected!");
            let plot = Plot::new("madf_plot")
                .legend(self.legend.clone())
                //.data_aspect(aspect as f32)
                //.clamp_grid(true)
                .center_x_axis(true)
                .sharp_grid_lines(true)
                .x_grid_spacer(adf_msf_x_angle_spacer)
                .y_grid_spacer(ndf_msf_y_uniform_spacer)
                .x_axis_formatter(|x, _| format!("{:.2}°", x.to_degrees()));
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
                for (i, (model, uuid)) in self.adf_models.iter().enumerate() {
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
                            let value = model.eval_adf(theta.cos(), phi.cos() as f64);
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
                if let Some(extra) = &mut self.variant {
                    extra.update_fitted_curves(prop.fitted.as_ref());
                }
            }

            match self.measurement_data_kind() {
                MeasurementKind::Bsdf => {
                    if let Some(extra) = &mut self.variant {
                        let extra = extra
                            .as_any_mut()
                            .downcast_mut::<BsdfPlotExtraData>()
                            .unwrap();
                        let cache = self.cache.read().unwrap();
                        let measurement = cache.get_measurement_data(self.data_handle).unwrap();
                        ui.horizontal(|ui| {
                            ui.label("Plot mode:");
                            ui.selectable_value(&mut extra.mode, BsdfPlotMode::Slice2D, "Slice 2D");
                            ui.selectable_value(&mut extra.mode, BsdfPlotMode::Slice3D, "Slice 3D");
                        });
                        let is_3d = extra.mode == BsdfPlotMode::Slice3D;
                        let bsdf_data = measurement.measured.bsdf_data().unwrap();
                        let zenith_i = bsdf_data.params.emitter.zenith;
                        let azimuth_i = bsdf_data.params.emitter.azimuth;
                        let (zenith_o, azimuth_o) = match bsdf_data.params.collector.scheme {
                            CollectorScheme::Partitioned { partition } => match partition {
                                SphericalPartition::EqualAngle { zenith, azimuth } => {
                                    (zenith, azimuth)
                                }
                                SphericalPartition::EqualArea { zenith, azimuth } => {
                                    (zenith.into(), azimuth)
                                }
                                SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                                    (zenith.into(), azimuth)
                                }
                            },
                            CollectorScheme::SingleRegion {
                                zenith, azimuth, ..
                            } => (zenith, azimuth),
                        };

                        // Incident direction controls
                        ui.allocate_ui_with_layout(
                            Vec2::new(ui.available_width(), 48.0),
                            egui::Layout::left_to_right(Align::Center),
                            |ui| {
                                ui.label("Incident direction: ");
                                let r0 = angle_knob(
                                    ui,
                                    true,
                                    &mut extra.zenith_i,
                                    zenith_i.range_bound_inclusive_f32(),
                                    zenith_i.step_size,
                                    48.0,
                                    |v| format!("θ = {:>6.2}°", v.to_degrees()),
                                )
                                .changed();
                                let r1 = angle_knob(
                                    ui,
                                    true,
                                    &mut extra.azimuth_i,
                                    azimuth_i.range_bound_inclusive_f32(),
                                    azimuth_i.step_size,
                                    48.0,
                                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                                )
                                .changed();
                                #[cfg(debug_assertions)]
                                {
                                    debug_print_angle(
                                        extra.zenith_i,
                                        &zenith_i,
                                        ui,
                                        "debug_print_θ",
                                    );
                                    debug_print_angle_pair(
                                        extra.azimuth_i,
                                        &azimuth_i,
                                        ui,
                                        "debug_print_φ",
                                    );
                                }
                                extra.changed |= r0 || r1;
                            },
                        );

                        if !is_3d {
                            ui.allocate_ui_with_layout(
                                Vec2::new(ui.available_width(), 48.0),
                                egui::Layout::left_to_right(Align::Center),
                                |ui| {
                                    ui.label("Outgoing direction: ");
                                    let r0 = angle_knob(
                                        ui,
                                        true,
                                        &mut extra.azimuth_o,
                                        azimuth_o.range_bound_inclusive_f32(),
                                        azimuth_o.step_size,
                                        48.0,
                                        |v| format!("φ = {:>6.2}°", v.to_degrees()),
                                    )
                                    .changed();
                                    #[cfg(debug_assertions)]
                                    {
                                        debug_print_angle_pair(
                                            extra.azimuth_o,
                                            &azimuth_o,
                                            ui,
                                            "debug_print_φ",
                                        );
                                    }
                                    extra.changed |= r0;
                                },
                            );
                        }

                        let zenith_i_count = zenith_i.step_count_wrapped();
                        let zenith_i_idx = zenith_i.index_of(extra.zenith_i);
                        let azimuth_i_idx = azimuth_i.index_of(extra.azimuth_i);
                        let data_point =
                            &bsdf_data.samples[azimuth_i_idx * zenith_i_count + zenith_i_idx];
                        let spectrum = bsdf_data.params.emitter.spectrum;
                        let wavelengths = spectrum
                            .values()
                            .map(|w| w.value() / 100.0)
                            .collect::<Vec<_>>();
                        let num_rays = bsdf_data.params.emitter.num_rays;

                        if extra.changed {
                            let (zenith_range, azimuth_range) =
                                bsdf_data.params.collector.scheme.ranges();
                            let count = bsdf_data.params.collector.scheme.total_sample_count();
                            debug_assert_eq!(
                                count,
                                zenith_range.step_count_wrapped()
                                    * azimuth_range.step_count_wrapped()
                            );
                            let mut samples_per_wavelength =
                                vec![vec![Vec3::ZERO; count]; wavelengths.len()];
                            let zenith_range_step_count = zenith_range.step_count_wrapped();
                            for (lambda_idx, samples) in
                                samples_per_wavelength.iter_mut().enumerate()
                            {
                                for (azimuth_idx, azimuth) in
                                    azimuth_range.values_wrapped().enumerate()
                                {
                                    for (zenith_idx, zenith) in
                                        zenith_range.values_wrapped().enumerate()
                                    {
                                        let idx =
                                            azimuth_idx * zenith_range_step_count + zenith_idx;
                                        let mut coord = math::spherical_to_cartesian(
                                            1.0,
                                            zenith,
                                            azimuth,
                                            Handedness::RightHandedYUp,
                                        );
                                        coord.y = data_point.data
                                            [azimuth_idx * zenith_range_step_count + zenith_idx]
                                            .0[lambda_idx]
                                            .total_energy;
                                        samples[idx] = coord;
                                    }
                                }
                            }
                            // let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                            //     label: Some(&format!("bsdf_view_buffer_{:?}",
                            // self.controls.view_id)),
                            //     size: std::mem::size_of::<Vec3>() as u64 * count as u64,
                            //     usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            //     mapped_at_creation: false,
                            // });
                            // self.gpu.queue.write_buffer(
                            //     &buffer,
                            //     0,
                            //     bytemuck::cast_slice(&samples_per_wavelength[0]),
                            // );
                            // self.event_loop
                            //     .send_event(VgonioEvent::BsdfViewer(BsdfViewerEvent::UpdateBuffer
                            // {         id: self.controls.view_id,
                            //         buffer: Some(buffer),
                            //         count: samples_per_wavelength[0].len() as u32,
                            //     }))
                            //     .unwrap();
                            extra.changed = false;
                        }

                        // Figure of per wavelength plot, it contains:
                        // - the number of absorbed rays per wavelength
                        // - the number of reflected rays per wavelength
                        // - the number of captured rays per wavelength
                        // - the number of captured energy per wavelength
                        let per_wavelength_plot = Plot::new("per_wavelength_plot")
                            .center_y_axis(false)
                            .data_aspect(1.0)
                            .legend(
                                Legend::default()
                                    .text_style(egui::TextStyle::Monospace)
                                    .background_alpha(1.0)
                                    .position(Corner::RightTop),
                            )
                            .x_grid_spacer(move |input| {
                                let mut marks = vec![];
                                let (min, max) = input.bounds;
                                let min = min as u32;
                                let n = max as u32 - min;
                                for i in 0..=n * 2 {
                                    marks.push(GridMark {
                                        value: i as f64 * 0.5 + min as f64,
                                        step_size: 0.5,
                                    })
                                }
                                marks
                            })
                            .y_grid_spacer(move |input| {
                                let mut marks = vec![];
                                let (min, max) = input.bounds;
                                let min = min as u32 * 10;
                                let max = max as u32 * 10;
                                for i in min..=max {
                                    let step_size = if i % 2 == 0 { 0.2 } else { continue };
                                    marks.push(GridMark {
                                        value: i as f64 * 0.1,
                                        step_size,
                                    })
                                }
                                marks
                            })
                            .x_axis_formatter(|x, _| format!("{:.0}nm", x * 100.0))
                            .y_axis_formatter(move |y, _| format!("{:.0}", y * num_rays as f64))
                            .coordinates_formatter(
                                Corner::LeftBottom,
                                CoordinatesFormatter::new(move |p, _| {
                                    format!("λ = {:.0}nm", p.x * 100.0,)
                                }),
                            )
                            .label_formatter(move |name, value| {
                                if name.starts_with('E') {
                                    format!(
                                        "{}: λ = {:.0}, e = {:.2}%)",
                                        name,
                                        value.x * 100.0,
                                        value.y * 100.0
                                    )
                                } else {
                                    format!(
                                        "{}: λ = {:.0}, n = {:.0})",
                                        name,
                                        value.x * 100.0,
                                        value.y * num_rays as f64
                                    )
                                }
                            })
                            .min_size(Vec2::new(400.0, 240.0));

                        let per_bounce_plot = Plot::new("per_bounce_plot")
                            .center_y_axis(false)
                            .data_aspect(2.5)
                            .legend(
                                Legend::default()
                                    .text_style(egui::TextStyle::Monospace)
                                    .background_alpha(1.0)
                                    .position(Corner::RightTop),
                            )
                            .x_grid_spacer(move |input| {
                                let (min, max) = input.bounds;
                                (min as u32..=max as u32)
                                    .map(move |i| GridMark {
                                        value: i as f64,
                                        step_size: 1.0,
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .x_axis_formatter(|x, _| format!("{}", x))
                            .y_grid_spacer(move |input| {
                                let (min, max) = input.bounds;
                                let mut i = (min.floor().max(0.0) * 10.0) as u32;
                                let mut marks = vec![];
                                while i <= (max.ceil() * 10.0) as u32 {
                                    let step_size = if i % 2 == 0 {
                                        0.2
                                    } else {
                                        i += 1;
                                        continue;
                                    };
                                    marks.push(GridMark {
                                        value: i as f64 / 10.0,
                                        step_size,
                                    });
                                    i += 1;
                                }
                                marks
                            })
                            .min_size(Vec2::new(400.0, 240.0));

                        // Figure of bsdf measurement statistics
                        egui::CollapsingHeader::new("Stats")
                            .default_open(true)
                            .show(ui, |ui| {
                                egui::Grid::new("stats_grid").num_columns(2).show(ui, |ui| {
                                    ui.label("Received rays:");
                                    ui.label(format!("{}", data_point.stats.n_received))
                                        .on_hover_text(
                                            "Number of emitted rays that hit the surface; \
                                             invariant over wavelength",
                                        );
                                    ui.end_row();

                                    ui.label("Per Wavelength: ");
                                    let n_reflected = wavelengths
                                        .iter()
                                        .zip(
                                            data_point
                                                .stats
                                                .n_reflected
                                                .iter()
                                                .map(|n| *n as f64 / num_rays as f64),
                                        )
                                        .map(|(s, n)| [*s as f64, n])
                                        .collect::<Vec<_>>();

                                    let n_absorbed = wavelengths
                                        .iter()
                                        .zip(
                                            data_point
                                                .stats
                                                .n_absorbed
                                                .iter()
                                                .map(|n| *n as f64 / num_rays as f64),
                                        )
                                        .map(|(s, n)| [*s as f64, n])
                                        .collect::<Vec<_>>();

                                    let n_captured = wavelengths
                                        .iter()
                                        .zip(
                                            data_point
                                                .stats
                                                .n_captured
                                                .iter()
                                                .map(|n| *n as f64 / num_rays as f64),
                                        )
                                        .map(|(s, n)| [*s as f64, n])
                                        .collect::<Vec<_>>();

                                    // Calculate the captured energy per wavelength by dividing the
                                    // captured energy by the number of
                                    // captured rays per wavelength
                                    let e_captured = wavelengths
                                        .iter()
                                        .zip(data_point.stats.e_captured.iter())
                                        .zip(data_point.stats.n_captured.iter())
                                        .map(|((l, e), n)| [*l as f64, *e as f64 / *n as f64])
                                        .collect::<Vec<_>>();

                                    per_wavelength_plot.show(ui, |plot_ui| {
                                        plot_ui.line(
                                            Line::new(n_captured.clone())
                                                .name("Nº Captured")
                                                .width(2.0),
                                        );
                                        plot_ui.points(
                                            Points::new(n_captured)
                                                .shape(MarkerShape::Circle)
                                                .radius(4.0)
                                                .name("Nº Captured"),
                                        );

                                        plot_ui.line(
                                            Line::new(n_absorbed.clone())
                                                .name("Nº Absorbed")
                                                .width(2.0),
                                        );
                                        plot_ui.points(
                                            Points::new(n_absorbed)
                                                .shape(MarkerShape::Diamond)
                                                .radius(4.0)
                                                .name("Nº Absorbed"),
                                        );

                                        plot_ui.line(
                                            Line::new(n_reflected.clone())
                                                .name("Nº Reflected")
                                                .width(2.0),
                                        );
                                        plot_ui.points(
                                            Points::new(n_reflected)
                                                .shape(MarkerShape::Square)
                                                .radius(4.0)
                                                .name("Nº Reflected"),
                                        );

                                        plot_ui.line(
                                            Line::new(e_captured.clone())
                                                .name("E Captured")
                                                .width(2.0),
                                        );
                                        plot_ui.points(
                                            Points::new(e_captured)
                                                .shape(MarkerShape::Cross)
                                                .radius(4.0)
                                                .name("E Captured"),
                                        );
                                    });
                                    ui.end_row();

                                    ui.label("Per Bounce:");
                                    let num_rays_bar_charts = data_point
                                        .stats
                                        .num_rays_per_bounce
                                        .iter()
                                        .zip(data_point.stats.n_reflected.iter())
                                        .zip(wavelengths.iter())
                                        .map(|((per_bounce_data, total), lambda)| {
                                            BarChart::new(
                                                per_bounce_data
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(i, n)| {
                                                        // Center the bar on the bounce number
                                                        // and scale the percentage to the range [0,
                                                        // 2]
                                                        Bar::new(
                                                            i as f64 + 0.5,
                                                            (*n as f64 / *total as f64) * 2.0,
                                                        )
                                                        .width(1.0)
                                                    })
                                                    .collect(),
                                            )
                                            .name(format!("Nº of rays, λ = {}", lambda))
                                            .element_formatter(Box::new(|bar, _| -> String {
                                                format!(
                                                    "bounce = {:.0}, number of rays = {:.0}%",
                                                    bar.argument + 0.5,
                                                    (bar.value / 2.0) * 100.0
                                                )
                                            }))
                                        });

                                    let energy_bar_charts = data_point
                                        .stats
                                        .energy_per_bounce
                                        .iter()
                                        .zip(data_point.stats.e_captured.iter())
                                        .zip(wavelengths.iter())
                                        .map(|((energy_per_bounce, total), lambda)| {
                                            BarChart::new(
                                                energy_per_bounce
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(i, e)| {
                                                        // Center the bar on the bounce number
                                                        // and scale the percentage to the range [0,
                                                        // 2]
                                                        Bar::new(
                                                            i as f64 + 0.5,
                                                            (*e as f64 / *total as f64) * 2.0,
                                                        )
                                                        .width(1.0)
                                                    })
                                                    .collect(),
                                            )
                                            .name(format!("Energy, λ = {}", lambda))
                                            .element_formatter(Box::new(|bar, _| -> String {
                                                format!(
                                                    "bounce = {:.0}, energy = {:.0}%",
                                                    bar.argument + 0.5,
                                                    (bar.value / 2.0) * 100.0
                                                )
                                            }))
                                        });

                                    per_bounce_plot.show(ui, |plot_ui| {
                                        for bar_chart in num_rays_bar_charts {
                                            plot_ui.bar_chart(bar_chart);
                                        }
                                        for bar_chart in energy_bar_charts {
                                            plot_ui.bar_chart(bar_chart);
                                        }
                                    });
                                    ui.end_row();
                                });
                            });

                        // Actual BSDF plot
                        let collapsing = egui::CollapsingHeader::new("BSDF");
                        let response = collapsing
                            .show(ui, |ui| {
                                egui::Grid::new("bsdf_grid")
                                    .striped(true)
                                    .num_columns(2)
                                    .show(ui, |ui| {
                                        // ui.label("Energy");
                                        // let response = ui.add(
                                        //     egui::Image::new(self.extra.view_id, [256.0, 256.0])
                                        //         .sense(egui::Sense::click_and_drag()),
                                        // );
                                        // if response.dragged_by(PointerButton::Primary) {
                                        //     let delta_x = response.drag_delta().x;
                                        //     if delta_x != 0.0 {
                                        //         self.event_loop
                                        //
                                        // .send_event(VgonioEvent::BsdfViewer(BsdfViewerEvent::Rotate {
                                        //                 id: self.extra.view_id,
                                        //                 angle: delta_x / 256.0 *
                                        // std::f32::consts::PI,
                                        //             }))
                                        //             .unwrap()
                                        //     }
                                        // }
                                        // ui.end_row();

                                        ui.label("Rays");
                                        ui.end_row();
                                    });
                            })
                            .header_response;

                        // if response.changed() {
                        //     self.event_loop
                        //         .send_event(VgonioEvent::BsdfViewer(BsdfViewerEvent::ToggleView(
                        //             self.extra.view_id,
                        //         )))
                        //         .unwrap()
                        // }
                    }
                }
                MeasurementKind::Adf => {
                    if let Some(variant) = &mut self.variant {
                        let cache = self.cache.read().unwrap();
                        let measurement = cache.get_measurement_data(self.data_handle).unwrap();
                        let zenith = measurement.measured.adf_or_msf_zenith().unwrap();
                        let zenith_bin_width_rad = zenith.step_size.as_f32();
                        variant.ui(ui, &mut self.event_loop, self.data_handle);
                        if let Some(curve) = variant.current_curve() {
                            let aspect = curve.max_val[0] / curve.max_val[1];
                            let plot = Plot::new("mndf_plot")
                                .legend(self.legend.clone())
                                .data_aspect(aspect as f32 * 0.25)
                                .center_x_axis(true)
                                .sharp_grid_lines(true)
                                .x_grid_spacer(adf_msf_x_angle_spacer)
                                .y_grid_spacer(ndf_msf_y_uniform_spacer)
                                .x_axis_formatter(|x, _| format!("{:.2}°", x.to_degrees()))
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
                                plot_ui.line(
                                    Line::new(
                                        curve
                                            .points
                                            .iter()
                                            .map(|[x, y]| [*x, *y * 4.0])
                                            .collect::<Vec<_>>(),
                                    )
                                    .stroke(egui::epaint::Stroke::new(2.0, LINE_COLORS[0]))
                                    .name("Measured - ADF (scaled)"),
                                );
                                let mut color_idx_base = 1;
                                let extra = variant
                                    .as_any()
                                    .downcast_ref::<AreaDistributionExtra>()
                                    .unwrap();
                                {
                                    for (i, (model, curves)) in extra.fitted.iter().enumerate() {
                                        plot_ui.line(
                                            Line::new(
                                                curves[extra.current_azimuth_idx()].points.clone(),
                                            )
                                            .stroke(egui::epaint::Stroke::new(
                                                2.0,
                                                LINE_COLORS[(i + 1) % LINE_COLORS.len()],
                                            ))
                                            .name(model.kind().to_str()),
                                        )
                                    }
                                    color_idx_base += extra.fitted.len() + 1;
                                }
                                for (i, (model, uuid)) in self.adf_models.iter().enumerate() {
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
                                                model.eval_adf(theta.cos(), phi.cos() as f64);
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
                        let cache = self.cache.read().unwrap();
                        let measurement = cache.get_measurement_data(self.data_handle).unwrap();
                        let zenith = measurement.measured.adf_or_msf_zenith().unwrap();
                        let zenith_bin_width_rad = zenith.step_size.value();
                        variant.ui(ui, &mut self.event_loop, self.data_handle);
                        if let Some(curve) = variant.current_curve() {
                            let aspect = curve.max_val[0] / curve.max_val[1];
                            let plot = Plot::new("plot_msf")
                                .legend(self.legend.clone())
                                .data_aspect(aspect as f32)
                                //.clamp_grid(true)
                                .center_x_axis(true)
                                .sharp_grid_lines(true)
                                .x_grid_spacer(adf_msf_x_angle_spacer)
                                .x_axis_formatter(|x, _| format!("{:.2}°", x.to_degrees()))
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
            }
        }
    }

    fn measurement_data_handle(&self) -> Handle<MeasurementData> { self.data_handle }

    fn measurement_data_kind(&self) -> MeasurementKind {
        match self.data_handle.variant_id() {
            0 => MeasurementKind::Bsdf,
            1 => MeasurementKind::Adf,
            2 => MeasurementKind::Msf,
            _ => {
                unreachable!()
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

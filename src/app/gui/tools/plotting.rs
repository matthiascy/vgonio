use crate::{
    app::gui::{
        tools::Tool,
        widgets::{AngleKnob, AngleKnobWinding},
    },
    math::NumericCast,
    measure::{
        measurement::{MeasuredData, MeasurementData},
        CollectorScheme,
    },
    units::{rad, Radians},
    RangeByStepSizeInclusive, SphericalPartition,
};
use egui::{plot::*, Align, Context, TextBuffer, Ui, Vec2};
use std::{any::Any, fmt::format, ops::RangeInclusive, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PlotType {
    Line,
    Bar,
}

pub trait PlottingWidget {
    fn name(&self) -> &str;
    fn ui(&mut self, ui: &mut Ui);
    fn show(&mut self, ctx: &Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| {
                self.ui(ui);
            });
    }
}

pub trait PlottingControls: 'static {}

pub struct PlottingInspector<C: PlottingControls> {
    /// Unique name for the plot.
    name: String,
    /// The data to be plotted
    data: Rc<MeasurementData>,
    /// The legend to be displayed
    legend: Legend,
    /// The type of plot to be displayed
    plot_type: PlotType,
    /// Controlling parameters for the plot.
    controls: C,
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

pub struct BsdfPlottingControls {
    azimuth_i: Radians,
    zenith_i: Radians,
    azimuth_o: Radians,
    mode: BsdfPlotMode,
}

impl Default for BsdfPlottingControls {
    fn default() -> Self {
        Self {
            azimuth_i: rad!(0.0),
            zenith_i: rad!(0.0),
            azimuth_o: rad!(0.0),
            mode: BsdfPlotMode::Slice2D,
        }
    }
}

pub struct MadfPlottingControls {
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    azimuth_m: Radians,
    /// The zenith angle of (facet normal m) the slice to be displayed, in
    /// radians.
    zenith_m: Radians,
}

impl Default for MadfPlottingControls {
    fn default() -> Self {
        Self {
            azimuth_m: rad!(0.0),
            zenith_m: rad!(0.0),
        }
    }
}

pub struct MmsfPlottingControls {
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    azimuth_m: Radians,
    /// The zenith angle of (facet normal m) the slice to be displayed, in
    /// radians.
    zenith_m: Radians,
    /// The azimuthal angle (incident direction i) of the slice to be displayed,
    azimuth_i: Radians,
}

impl Default for MmsfPlottingControls {
    fn default() -> Self {
        Self {
            azimuth_m: rad!(0.0),
            zenith_m: rad!(0.0),
            azimuth_i: rad!(0.0),
        }
    }
}

impl PlottingControls for BsdfPlottingControls {}
impl PlottingControls for MadfPlottingControls {}
impl PlottingControls for MmsfPlottingControls {}

impl<C: PlottingControls> PlottingInspector<C> {
    pub fn new(name: String, data: Rc<MeasurementData>, controls: C) -> Self {
        Self {
            name,
            data,
            legend: Legend::default()
                .text_style(egui::TextStyle::Monospace)
                .background_alpha(1.0)
                .position(Corner::RightTop),
            plot_type: PlotType::Line,
            controls,
        }
    }

    fn angle_knob(
        ui: &mut Ui,
        interactive: bool,
        angle: &mut Radians,
        range: RangeInclusive<f32>,
        snap: Radians,
        diameter: f32,
        formatter: impl Fn(f32) -> String,
    ) {
        ui.add(
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
                range.index_of(opposite.into()),
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

    fn plot_type_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Plot type:");
            ui.selectable_value(&mut self.plot_type, PlotType::Line, "Line");
            ui.selectable_value(&mut self.plot_type, PlotType::Bar, "Bar");
        });
    }
}

impl PlottingWidget for PlottingInspector<MadfPlottingControls> {
    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui) {
        let zenith = self.data.measured.madf_or_mmsf_zenith().unwrap();
        let azimuth = self.data.measured.madf_or_mmsf_azimuth().unwrap();
        let zenith_bin_width_rad = zenith.step_size.cast();
        self.plot_type_ui(ui);
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("microfacet normal: ");
                let mut opposite = self.controls.azimuth_m.wrap_to_tau().opposite();
                Self::angle_knob(
                    ui,
                    false,
                    &mut opposite,
                    azimuth.map(|x| x.value).range_bound_inclusive(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.controls.azimuth_m,
                    azimuth.map(|x| x.value).range_bound_inclusive(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                Self::debug_print_angle_pair(
                    self.controls.azimuth_m,
                    &azimuth,
                    ui,
                    "debug_print_φ_pair",
                );
            },
        );
        let data: Vec<_> = {
            let (starting, opposite) = self.data.adf_data_slice(self.controls.azimuth_m);

            // Data of the opposite azimuthal angle side of the slice, if exists.
            let data_opposite_part = opposite.map(|data| {
                data.iter()
                    .rev()
                    .zip(zenith.values_rev().map(|x| -x))
                    .map(|(y, x)| [x.value as f64, *y as f64])
            });

            let data_starting_part = starting
                .iter()
                .zip(zenith.values())
                .map(|(y, x)| [x.value as f64, *y as f64]);

            match data_opposite_part {
                None => data_starting_part.collect(),
                Some(opposite) => opposite
                    .take(zenith.step_count_wrapped() - 1)
                    .chain(data_starting_part)
                    .collect(),
            }
        };

        let (max_x, max_y) = data.iter().fold((0.01, 0.01), |(max_x, max_y), [x, y]| {
            let val_x = x.abs().max(max_x);
            let val_y = y.max(max_y);
            (val_x, val_y)
        });

        let plot = Plot::new("plotting")
            .legend(self.legend.clone())
            .data_aspect((max_x / max_y) as f32)
            //.clamp_grid(true)
            .center_x_axis(true)
            .sharp_grid_lines(true)
            .x_grid_spacer(|input| {
                let mut marks = vec![];
                let (min, max) = input.bounds;
                let min = min.floor().to_degrees() as i32;
                let max = max.ceil().to_degrees() as i32;
                for i in min..=max {
                    let step_size = if i % 30 == 0 {
                        // 5 degrees
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
            })
            .x_axis_formatter(|x, _| format!("{:.2}°", x.to_degrees()))
            .coordinates_formatter(
                Corner::LeftBottom,
                CoordinatesFormatter::new(move |p, _| {
                    let n_bin = (p.x / zenith_bin_width_rad as f64 + 0.5).floor();
                    let bin = n_bin * zenith_bin_width_rad.to_degrees() as f64;
                    let half_bin_width = zenith_bin_width_rad.to_degrees() as f64 * 0.5;
                    format!(
                        "φ: {:.2}° θ: {:.2}°±{half_bin_width:.2}°\nValue: {:.2} sr⁻¹",
                        0.0, bin, p.y
                    )
                }),
            );

        plot.show(ui, |plot_ui| match self.plot_type {
            PlotType::Line => {
                plot_ui.line(
                    Line::new(data)
                        .stroke(egui::epaint::Stroke::new(2.0, egui::Color32::LIGHT_RED))
                        .name("Microfacet area distribution"),
                );
            }
            PlotType::Bar => {
                plot_ui.bar_chart(
                    BarChart::new(
                        data.iter()
                            .map(|[x, y]| {
                                Bar::new(*x, *y)
                                    .width(zenith_bin_width_rad as f64)
                                    .stroke(egui::epaint::Stroke::new(
                                        1.0,
                                        egui::Color32::LIGHT_RED,
                                    ))
                                    .fill(egui::Color32::from_rgba_unmultiplied(255, 128, 128, 128))
                            })
                            .collect(),
                    )
                    .name("Microfacet area distribution"),
                );
            }
        });
    }
}

impl PlottingWidget for PlottingInspector<MmsfPlottingControls> {
    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui) {
        let zenith = self.data.measured.madf_or_mmsf_zenith().unwrap();
        let azimuth = self.data.measured.madf_or_mmsf_azimuth().unwrap();
        let zenith_bin_width_rad = zenith.step_size.value;
        self.plot_type_ui(ui);
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("microfacet normal: ");
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.controls.zenith_m,
                    zenith.range_bound_inclusive_f32(),
                    zenith.step_size,
                    48.0,
                    |v| format!("θ = {:>6.2}°", v.to_degrees()),
                );
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.controls.azimuth_m,
                    azimuth.range_bound_inclusive_f32(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                {
                    Self::debug_print_angle_pair(
                        self.controls.azimuth_m,
                        &azimuth,
                        ui,
                        "debug_print_φ",
                    );
                    Self::debug_print_angle(self.controls.zenith_m, &zenith, ui, "debug_print_θ");
                }
            },
        );
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("incident direction: ");
                let mut opposite = self.controls.azimuth_i.opposite();
                Self::angle_knob(
                    ui,
                    false,
                    &mut opposite,
                    azimuth.range_bound_inclusive_f32(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.controls.azimuth_i,
                    azimuth.range_bound_inclusive_f32(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                Self::debug_print_angle_pair(
                    self.controls.azimuth_i,
                    &azimuth,
                    ui,
                    "debug_print_φ_pair",
                );
            },
        );
        let data: Vec<_> = {
            let (starting, opposite) = self.data.msf_data_slice(
                self.controls.azimuth_m,
                self.controls.zenith_m,
                self.controls.azimuth_i,
            );
            let data_opposite_part = opposite.map(|data| {
                data.iter()
                    .rev()
                    .zip(zenith.values_rev().map(|v| -v))
                    .map(|(y, x)| [x.value as f64, *y as f64])
            });
            let data_starting_part = starting
                .iter()
                .zip(zenith.values())
                .map(|(y, x)| [x.value as f64, *y as f64]);

            match data_opposite_part {
                None => data_starting_part.collect(),
                Some(opposite) => opposite
                    .take(zenith.step_count_wrapped() - 1)
                    .chain(data_starting_part)
                    .collect(),
            }
        };
        let (max_x, max_y) = data.iter().fold((0.01, 0.01), |(max_x, max_y), [x, y]| {
            let val_x = x.abs().max(max_x);
            let val_y = y.abs().max(max_y);
            (val_x, val_y)
        });
        let plot = Plot::new("plot_msf")
            .legend(self.legend.clone())
            .data_aspect((max_x / max_y) as f32)
            //.clamp_grid(true)
            .center_x_axis(true)
            .sharp_grid_lines(true)
            .x_grid_spacer(|input| {
                let mut marks = vec![];
                let (min, max) = input.bounds;
                let min = min.floor().to_degrees() as i32;
                let max = max.ceil().to_degrees() as i32;
                for i in min..=max {
                    let step_size = if i % 30 == 0 {
                        // 30 degrees
                        30.0f64.to_radians()
                    } else if i % 10 == 0 {
                        // 10 degrees
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
            })
            .x_axis_formatter(|x, _| format!("{:.2}°", x.to_degrees()))
            .coordinates_formatter(
                Corner::LeftBottom,
                CoordinatesFormatter::new(move |p, _| {
                    let n_bin = (p.x / zenith_bin_width_rad as f64 + 0.5).floor();
                    let bin = n_bin * zenith_bin_width_rad.to_degrees() as f64;
                    let half_bin_width = zenith_bin_width_rad.to_degrees() as f64 * 0.5;
                    format!(
                        "φ: {:.2}° θ: {:.2}°±{half_bin_width:.2}°\nValue: {:.2}",
                        0.0, bin, p.y
                    )
                }),
            );

        plot.show(ui, |plot_ui| match self.plot_type {
            PlotType::Line => {
                plot_ui.line(
                    Line::new(data)
                        .stroke(egui::epaint::Stroke::new(2.0, egui::Color32::LIGHT_RED))
                        .name("Microfacet masking shadowing"),
                );
            }
            PlotType::Bar => {
                plot_ui.bar_chart(
                    BarChart::new(
                        data.iter()
                            .map(|[x, y]| {
                                Bar::new(*x, *y)
                                    .width(zenith_bin_width_rad as f64)
                                    .stroke(egui::epaint::Stroke::new(
                                        1.0,
                                        egui::Color32::LIGHT_RED,
                                    ))
                                    .fill(egui::Color32::from_rgba_unmultiplied(255, 128, 128, 128))
                            })
                            .collect(),
                    )
                    .name("Microfacet masking shadowing"),
                );
            }
        });
    }
}

impl PlottingWidget for PlottingInspector<BsdfPlottingControls> {
    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Plot mode:");
            ui.selectable_value(&mut self.controls.mode, BsdfPlotMode::Slice2D, "Slice 2D");
            ui.selectable_value(&mut self.controls.mode, BsdfPlotMode::Slice3D, "Slice 3D");
        });
        let is_3d = self.controls.mode == BsdfPlotMode::Slice3D;

        if !is_3d {
            self.plot_type_ui(ui);
        }

        let bsdf_data = self.data.measured.bsdf_data().unwrap();
        let zenith_i = bsdf_data.params.emitter.zenith;
        let azimuth_i = bsdf_data.params.emitter.azimuth;
        let (zenith_o, azimuth_o) = match bsdf_data.params.collector.scheme {
            CollectorScheme::Partitioned { partition, .. } => match partition {
                SphericalPartition::EqualAngle { zenith, azimuth } => (zenith, azimuth),
                SphericalPartition::EqualArea { zenith, azimuth } => (zenith.into(), azimuth),
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
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.controls.zenith_i,
                    zenith_i.range_bound_inclusive_f32(),
                    zenith_i.step_size,
                    48.0,
                    |v| format!("θ = {:>6.2}°", v.to_degrees()),
                );
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.controls.azimuth_i,
                    azimuth_i.range_bound_inclusive_f32(),
                    azimuth_i.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                {
                    Self::debug_print_angle(self.controls.zenith_i, &zenith_i, ui, "debug_print_θ");
                    Self::debug_print_angle_pair(
                        self.controls.azimuth_i,
                        &azimuth_i,
                        ui,
                        "debug_print_φ",
                    );
                }
            },
        );

        if !is_3d {
            ui.allocate_ui_with_layout(
                Vec2::new(ui.available_width(), 48.0),
                egui::Layout::left_to_right(Align::Center),
                |ui| {
                    ui.label("Outgoing direction: ");
                    Self::angle_knob(
                        ui,
                        true,
                        &mut self.controls.azimuth_o,
                        azimuth_o.range_bound_inclusive_f32(),
                        azimuth_o.step_size,
                        48.0,
                        |v| format!("φ = {:>6.2}°", v.to_degrees()),
                    );
                    #[cfg(debug_assertions)]
                    {
                        Self::debug_print_angle_pair(
                            self.controls.azimuth_o,
                            &azimuth_o,
                            ui,
                            "debug_print_φ",
                        );
                    }
                },
            );
        }

        let zenith_i_count = zenith_i.step_count_wrapped();
        let azimuth_i_idx = azimuth_i.index_of(self.controls.azimuth_i);
        let bsdf_data_point = &bsdf_data.samples[azimuth_i_idx * zenith_i_count];
        let spectrum = bsdf_data.params.emitter.spectrum;
        let wavelengths = spectrum
            .values()
            .map(|w| w.value / 100.0)
            .collect::<Vec<_>>();
        let num_rays = bsdf_data.params.emitter.num_rays;
        let max_bounces = bsdf_data.params.emitter.max_bounces;

        let per_wavelength_plot = Plot::new("num_rays_per_wavelength_plot")
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
                let max = max as u32;
                for i in min..=max {
                    let step_size = if i % 2 == 0 {
                        2.0
                    } else if i % 1 == 0 {
                        1.0
                    } else {
                        continue;
                    };
                    marks.push(GridMark {
                        value: i as f64,
                        step_size,
                    })
                }
                marks
            })
            .x_axis_formatter(|x, _| format!("{:.0}nm", x * 100.0))
            .y_axis_formatter(move |y, _| format!("{:.0}", y * num_rays as f64))
            .coordinates_formatter(
                Corner::LeftBottom,
                CoordinatesFormatter::new(move |p, _| format!("λ = {:.0}nm", p.x * 100.0,)),
            )
            .label_formatter(move |name, value| {
                format!(
                    "{}: λ = {:.0}, n = {:.0})",
                    name,
                    value.x * 100.0,
                    value.y * num_rays as f64
                )
            })
            .min_size(Vec2::new(400.0, 220.0));

        let per_bounce_plot = Plot::new("num_rays_per_bounce")
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
                    .into_iter()
                    .map(move |i| GridMark {
                        value: i as f64,
                        step_size: 1.0,
                    })
                    .collect::<Vec<_>>()
            })
            .x_axis_formatter(|x, r| format!("{}", x))
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
            .min_size(Vec2::new(400.0, 220.0));

        egui::CollapsingHeader::new("Stats")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("stats_grid").num_columns(2).show(ui, |ui| {
                    ui.label("Received rays:");
                    ui.label(format!("{}", bsdf_data_point.stats.n_received))
                        .on_hover_text(
                            "Number of emitted rays that hit the surface; invariant over \
                             wavelength",
                        );
                    ui.end_row();

                    ui.label("Ray Count: ");
                    let n_reflected = wavelengths
                        .iter()
                        .zip(
                            bsdf_data_point
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
                            bsdf_data_point
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
                            bsdf_data_point
                                .stats
                                .n_captured
                                .iter()
                                .map(|n| *n as f64 / num_rays as f64),
                        )
                        .map(|(s, n)| [*s as f64, n])
                        .collect::<Vec<_>>();

                    per_wavelength_plot.show(ui, |plot_ui| {
                        plot_ui.line(Line::new(n_captured.clone()).name("Captured").width(2.0));
                        plot_ui.points(
                            Points::new(n_captured)
                                .shape(MarkerShape::Cross)
                                .radius(4.0)
                                .name("Captured"),
                        );

                        plot_ui.line(Line::new(n_absorbed.clone()).name("Absorbed").width(2.0));
                        plot_ui.points(
                            Points::new(n_absorbed)
                                .shape(MarkerShape::Diamond)
                                .radius(4.0)
                                .name("Absorbed"),
                        );

                        plot_ui.line(Line::new(n_reflected.clone()).name("Reflected").width(2.0));
                        plot_ui.points(
                            Points::new(n_reflected)
                                .shape(MarkerShape::Square)
                                .radius(4.0)
                                .name("Reflected"),
                        );
                    });
                    ui.end_row();

                    ui.label("Ray Bounces:");
                    let bar_charts = bsdf_data_point
                        .stats
                        .num_rays_per_bounce
                        .iter()
                        .zip(bsdf_data_point.stats.n_reflected.iter())
                        .zip(wavelengths.iter())
                        .map(|((per_bounce_data, total), lambda)| {
                            BarChart::new(
                                per_bounce_data
                                    .iter()
                                    .enumerate()
                                    .map(|(i, n)| {
                                        // Center the bar on the bounce number
                                        // and scale the percentage to the range [0, 2]
                                        Bar::new(i as f64 + 0.5, (*n as f64 / *total as f64) * 2.0)
                                            .width(1.0)
                                    })
                                    .collect(),
                            )
                            .name(format!("λ = {}", lambda))
                            .element_formatter(Box::new(
                                |bar, chart| -> String {
                                    format!("{:.0}%", (bar.value / 2.0) * 100.0)
                                },
                            ))
                        });

                    per_bounce_plot.show(ui, |plot_ui| {
                        for bar_chart in bar_charts {
                            plot_ui.bar_chart(bar_chart);
                        }
                    });
                    ui.end_row();

                    ui.label("Ray Energy:");
                    ui.end_row();
                });
            });
    }
}

impl<C: PlottingControls> Tool for PlottingInspector<C>
where
    PlottingInspector<C>: PlottingWidget,
{
    fn name(&self) -> &str { self.name.as_str() }

    fn show(&mut self, ctx: &Context, open: &mut bool) {
        <Self as PlottingWidget>::show(self, ctx, open);
    }

    fn ui(&mut self, ui: &mut Ui) { <Self as PlottingWidget>::ui(self, ui); }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

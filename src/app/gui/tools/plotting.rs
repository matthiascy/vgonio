use crate::{
    app::gui::{
        tools::Tool,
        widgets::{AngleKnob, AngleKnobWinding},
        Plottable, PlottingMode,
    },
    math,
    math::NumericCast,
    measure::measurement::MeasurementData,
    units::{rad, Radians},
    RangeByStepSizeInclusive,
};
use egui::{plot::*, Align, Context, Ui, Vec2, Widget, WidgetText};
use std::{any::Any, f32::consts::TAU, ops::RangeInclusive, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PlotType {
    Line,
    Bar,
}

pub struct PlottingInspector {
    /// Unique name for the plot.
    pub name: String,
    /// The data to be plotted
    pub data: Rc<dyn Plottable>,
    /// The legend to be displayed
    pub legend: Legend,
    /// The type of plot to be displayed
    plot_type: PlotType,
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    azimuth_m: Radians,
    /// The zenith angle of (facet normal m) the slice to be displayed, in
    /// radians.
    zenith_m: Radians,
    /// The azimuthal angle (incident direction i) of the slice to be displayed,
    azimuth_i: Radians,
}

impl PlottingInspector {
    pub fn new(name: String, data: Rc<dyn Plottable>) -> Self {
        Self {
            name,
            data,
            legend: Legend::default()
                .text_style(egui::TextStyle::Monospace)
                .background_alpha(1.0)
                .position(Corner::RightTop),
            plot_type: PlotType::Line,
            azimuth_m: rad!(0.0),
            zenith_m: rad!(0.0),
            azimuth_i: rad!(0.0),
        }
    }

    fn angle_slider(
        value: &mut f32,
        range: RangeInclusive<f32>,
        step: f64,
        text: impl Into<WidgetText>,
    ) -> impl Widget + '_ {
        egui::Slider::new(value, range)
            .clamp_to_range(true)
            .step_by(step)
            .custom_formatter(|x, _| format!("{:.2}°", x.to_degrees()))
            .text(text)
    }

    fn angle_knob(
        ui: &mut Ui,
        interactive: bool,
        value: &mut f32,
        range: RangeInclusive<f32>,
        snap: f32,
        diameter: f32,
        formatter: impl Fn(f32) -> String,
    ) {
        ui.add(
            AngleKnob::new(value)
                .interactive(interactive)
                .min(Some(*range.start()))
                .max(Some(*range.end()))
                .snap(Some(snap))
                .winding(AngleKnobWinding::CounterClockwise)
                .diameter(diameter)
                .axis_count((TAU / snap).ceil() as u32),
        );
        ui.label(formatter(*value));
    }

    #[cfg(debug_assertions)]
    fn debug_print_angle_pair(
        initial: Radians,
        range: &RangeByStepSizeInclusive<Radians>,
        ui: &mut Ui,
        text: &str,
    ) {
        if ui.button(text).clicked() {
            let opposite = math::calculate_opposite_angle(initial.value);
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
            let initial = rad!(math::wrap_angle_to_tau_exclusive(initial.value));
            println!(
                "angle = {}, index = {}",
                initial.to_degrees(),
                range.index_of(initial),
            );
        }
    }
}

impl Tool for PlottingInspector {
    fn name(&self) -> &str { self.name.as_str() }

    fn show(&mut self, ctx: &Context, open: &mut bool) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| self.ui(ui));
    }

    fn ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Plot type:");
            ui.selectable_value(&mut self.plot_type, PlotType::Line, "Line");
            ui.selectable_value(&mut self.plot_type, PlotType::Bar, "Bar");
        });

        match self.data.mode() {
            PlottingMode::Adf => {
                let measured = self
                    .data
                    .as_any()
                    .downcast_ref::<MeasurementData>()
                    .unwrap();
                let zenith_bin_width_rad = measured.zenith.step_size().cast();
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 48.0),
                    egui::Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.label("microfacet normal: ");
                        let mut opposite = math::calculate_opposite_angle(self.azimuth_m.value);
                        PlottingInspector::angle_knob(
                            ui,
                            false,
                            &mut opposite,
                            measured.azimuth.map(|x| x.value).range_bound_inclusive(),
                            measured.azimuth.step_size().value,
                            48.0,
                            |v| format!("φ = {:>6.2}°", v.to_degrees()),
                        );
                        PlottingInspector::angle_knob(
                            ui,
                            true,
                            &mut self.azimuth_m.value,
                            measured.azimuth.map(|x| x.value).range_bound_inclusive(),
                            measured.azimuth.step_size().value,
                            48.0,
                            |v| format!("φ = {:>6.2}°", v.to_degrees()),
                        );
                        #[cfg(debug_assertions)]
                        Self::debug_print_angle_pair(
                            self.azimuth_m,
                            &measured.azimuth,
                            ui,
                            "debug_print_φ_pair",
                        );
                    },
                );
                let data: Vec<_> = {
                    let (starting, opposite) = measured.adf_data_slice(self.azimuth_m.value);

                    // Data of the opposite azimuthal angle side of the slice, if exists.
                    let data_opposite_part = opposite.map(|data| {
                        data.iter()
                            .rev()
                            .zip(measured.zenith.values_rev().map(|x| -x))
                            .map(|(y, x)| [x.value as f64, *y as f64])
                    });

                    let data_starting_part = starting
                        .iter()
                        .zip(measured.zenith.values())
                        .map(|(y, x)| [x.value as f64, *y as f64]);

                    match data_opposite_part {
                        None => data_starting_part.collect(),
                        Some(opposite) => opposite
                            .take(measured.zenith.step_count() - 1)
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
                                            .fill(egui::Color32::from_rgba_unmultiplied(
                                                255, 128, 128, 128,
                                            ))
                                    })
                                    .collect(),
                            )
                            .name("Microfacet area distribution"),
                        );
                    }
                });
            }

            PlottingMode::None => {
                ui.label("No data to plot");
            }
            PlottingMode::Bsdf => {
                ui.label("Bidirectional Scattering Distribution Function");
            }
            PlottingMode::Msf => {
                let measured = self
                    .data
                    .as_any()
                    .downcast_ref::<MeasurementData>()
                    .unwrap();
                let zenith_bin_width_rad = measured.zenith.step_size().value;
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 48.0),
                    egui::Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.label("microfacet normal: ");
                        PlottingInspector::angle_knob(
                            ui,
                            true,
                            &mut self.zenith_m.value,
                            measured.zenith.range_bound_inclusive_f32(),
                            measured.zenith.step_size().value,
                            48.0,
                            |v| format!("θ = {:>6.2}°", v.to_degrees()),
                        );
                        PlottingInspector::angle_knob(
                            ui,
                            true,
                            &mut self.azimuth_m.value,
                            measured.azimuth.range_bound_inclusive_f32(),
                            measured.azimuth.step_size().value,
                            48.0,
                            |v| format!("φ = {:>6.2}°", v.to_degrees()),
                        );
                        #[cfg(debug_assertions)]
                        Self::debug_print_angle_pair(
                            self.azimuth_m,
                            &measured.azimuth,
                            ui,
                            "debug_print_φ",
                        );
                        #[cfg(debug_assertions)]
                        Self::debug_print_angle(
                            self.zenith_m,
                            &measured.zenith,
                            ui,
                            "debug_print_θ",
                        );
                    },
                );
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 48.0),
                    egui::Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.label("incident direction: ");
                        let mut opposite = math::calculate_opposite_angle(self.azimuth_i.value);
                        PlottingInspector::angle_knob(
                            ui,
                            false,
                            &mut opposite,
                            measured.azimuth.range_bound_inclusive_f32(),
                            measured.azimuth.step_size().value,
                            48.0,
                            |v| format!("φ = {:>6.2}°", v.to_degrees()),
                        );
                        PlottingInspector::angle_knob(
                            ui,
                            true,
                            &mut self.azimuth_i.value,
                            measured.azimuth.range_bound_inclusive_f32(),
                            measured.azimuth.step_size().value,
                            48.0,
                            |v| format!("φ = {:>6.2}°", v.to_degrees()),
                        );
                        #[cfg(debug_assertions)]
                        Self::debug_print_angle_pair(
                            self.azimuth_i,
                            &measured.azimuth,
                            ui,
                            "debug_print_φ_pair",
                        );
                    },
                );
                let data: Vec<_> = {
                    let (starting, opposite) = measured.msf_data_slice(
                        self.azimuth_m.value,
                        self.zenith_m.value,
                        self.azimuth_i.value,
                    );
                    let data_opposite_part = opposite.map(|data| {
                        data.iter()
                            .rev()
                            .zip(measured.zenith.values_rev().map(|v| -v))
                            .map(|(y, x)| [x.value as f64, *y as f64])
                    });
                    let data_starting_part = starting
                        .iter()
                        .zip(measured.zenith.values())
                        .map(|(y, x)| [x.value as f64, *y as f64]);

                    match data_opposite_part {
                        None => data_starting_part.collect(),
                        Some(opposite) => opposite
                            .take(measured.zenith.step_count() as usize - 1)
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
                                            .fill(egui::Color32::from_rgba_unmultiplied(
                                                255, 128, 128, 128,
                                            ))
                                    })
                                    .collect(),
                            )
                            .name("Microfacet masking shadowing"),
                        );
                    }
                });
            }
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn as_any(&self) -> &dyn Any { self }
}

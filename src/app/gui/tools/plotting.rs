use crate::{
    app::gui::{tools::Tool, Plottable, PlottingMode},
    measure::measurement::MeasurementData,
};
use egui::{plot::*, text::LayoutJob, Context, Response, Ui, Widget, WidgetText};
use std::{
    any::Any,
    iter::{Map, Rev, Zip},
    ops::RangeInclusive,
    rc::Rc,
    slice::Iter,
};

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
    /// The azimuthal angle of the slice to be displayed, in radians.
    microfacet_normal_azimuth: f32,
    microfacet_normal_zenith: f32,

    incident_direction_azimuth: f32,
    incident_direction_zenith: f32,
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
            microfacet_normal_azimuth: 0.0,
            microfacet_normal_zenith: 0.0,
            incident_direction_azimuth: 0.0,
            incident_direction_zenith: 0.0,
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
                let zenith_bin_width_rad = measured.zenith.bin_width;

                ui.horizontal(|ui| {
                    ui.label("microfacet normal - φ: ");
                    ui.add(PlottingInspector::angle_slider(
                        &mut self.microfacet_normal_azimuth,
                        measured.azimuth.range_inclusive(),
                        measured.azimuth.bin_width as _,
                        "",
                    ));
                });

                let data: Vec<_> = {
                    let (starting, opposite) =
                        measured.adf_data_slice(self.microfacet_normal_azimuth);

                    // Data of the opposite azimuthal angle side of the slice, if exists.
                    let data_opposite_part = opposite.map(|data| {
                        data.iter()
                            .rev()
                            .zip(measured.zenith.rev_negative_angles())
                            .map(|(y, x)| [x as f64, *y as f64])
                    });

                    let data_starting_part = starting
                        .iter()
                        .zip(measured.zenith.angles())
                        .map(|(y, x)| [x as f64, *y as f64]);

                    match data_opposite_part {
                        None => data_starting_part.collect(),
                        Some(opposite) => opposite
                            .take(measured.zenith.bin_count as usize - 1)
                            .chain(data_starting_part)
                            .collect(),
                    }
                };

                let (max_x, max_y) = data.iter().fold((0.0, 0.0), |(max_x, max_y), [x, y]| {
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
                        CoordinatesFormatter::new(move |p, b| {
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
                let zenith_bin_width_rad = measured.zenith.bin_width;
                ui.horizontal(|ui| {
                    ui.label("microfacet normal: ");
                    ui.vertical(|ui| {
                        ui.add(PlottingInspector::angle_slider(
                            &mut self.microfacet_normal_azimuth,
                            measured.azimuth.range_inclusive(),
                            measured.azimuth.bin_width as _,
                            "φ",
                        ));
                        ui.add(PlottingInspector::angle_slider(
                            &mut self.microfacet_normal_zenith,
                            measured.zenith.range_inclusive(),
                            measured.zenith.bin_width as _,
                            "θ",
                        ));
                    });
                });
                ui.horizontal(|ui| {
                    ui.label("incident direction:  ");
                    ui.add(
                        egui::Slider::new(
                            &mut self.incident_direction_azimuth,
                            measured.azimuth.start..=measured.azimuth.end,
                        )
                        .clamp_to_range(true)
                        .step_by(measured.azimuth.bin_width as _)
                        .custom_formatter(|x, _| format!("{:.2}°", x.to_degrees()))
                        .text("φ"),
                    );
                });

                let data: Vec<_> = {
                    let (starting, opposite) = measured.msf_data_slice(
                        self.microfacet_normal_azimuth,
                        self.microfacet_normal_zenith,
                        self.incident_direction_azimuth,
                    );
                    let data_opposite_part = opposite.map(|data| {
                        data.iter()
                            .rev()
                            .zip(measured.zenith.rev_negative_angles())
                            .map(|(y, x)| [x as f64, *y as f64])
                    });
                    let data_starting_part = starting
                        .iter()
                        .zip(measured.zenith.angles())
                        .map(|(y, x)| [x as f64, *y as f64]);

                    match data_opposite_part {
                        None => data_starting_part.collect(),
                        Some(opposite) => opposite
                            .take(measured.zenith.bin_count as usize - 1)
                            .chain(data_starting_part)
                            .collect(),
                    }
                };
                let (max_x, max_y) = data.iter().fold((0.0, 0.0), |(max_x, max_y), [x, y]| {
                    let val_x = x.abs().max(max_x);
                    let val_y = y.max(max_y);
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
                        CoordinatesFormatter::new(move |p, b| {
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

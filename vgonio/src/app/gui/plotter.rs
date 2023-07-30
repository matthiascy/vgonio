use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{
            event::EventLoopProxy,
            widgets::{AngleKnob, AngleKnobWinding},
        },
    },
    measure::{
        measurement::{MeasurementData, MeasurementKind},
        CollectorScheme,
    },
    RangeByStepSizeInclusive, SphericalPartition,
};
use egui::{plot::*, Align, Context, Response, Ui, Vec2};
use std::ops::RangeInclusive;
use uuid::Uuid;
use vgcore::{
    math,
    math::{Handedness, Vec3},
    units::{rad, Radians},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PlotType {
    Line,
    Bar,
}

pub trait PlottingWidget {
    fn uuid(&self) -> Uuid;
    fn name(&self) -> &str;
    fn ui(&mut self, ui: &mut Ui, cache: &Cache);
    fn show(&mut self, ctx: &Context, open: &mut bool, cache: &Cache) {
        egui::Window::new(self.name())
            .open(open)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::new([false, true])
                    .min_scrolled_height(640.0)
                    .show(ui, |ui| {
                        self.ui(ui, cache);
                    });
            });
    }
    fn data_handle(&self) -> Handle<MeasurementData>;

    fn data_kind(&self) -> MeasurementKind;
}

/// Trait for extra data to be used by the plotting inspector.
pub trait ExtraData: 'static {
    /// Initialise the extra data.
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache);
}

pub struct PlotInspector<C: ExtraData> {
    /// Unique ID for the plot widget.
    uuid: Uuid,
    /// Unique name for the plot.
    name: String,
    /// The handle to the data to be plotted.
    data: Handle<MeasurementData>,
    /// The legend to be displayed
    legend: Legend,
    /// The type of plot to be displayed
    plot_type: PlotType,
    /// Extra data, including controls parameters and the pre-processed data.
    extra: C,
    /// The event loop.
    event_loop: EventLoopProxy,
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

pub struct MadfPlotExtraData {
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    azimuth_m: Radians,
    /// The zenith angle of (facet normal m) the slice to be displayed, in
    /// radians.
    zenith_m: Radians,
}

impl Default for MadfPlotExtraData {
    fn default() -> Self {
        Self {
            azimuth_m: rad!(0.0),
            zenith_m: rad!(0.0),
        }
    }
}

pub struct MmsfPlotExtraData {
    /// The azimuthal angle (facet normal m) of the slice to be displayed, in
    /// radians.
    azimuth_m: Radians,
    /// The zenith angle of (facet normal m) the slice to be displayed, in
    /// radians.
    zenith_m: Radians,
    /// The azimuthal angle (incident direction i) of the slice to be displayed,
    azimuth_i: Radians,
}

impl Default for MmsfPlotExtraData {
    fn default() -> Self {
        Self {
            azimuth_m: rad!(0.0),
            zenith_m: rad!(0.0),
            azimuth_i: rad!(0.0),
        }
    }
}

impl ExtraData for BsdfPlotExtraData {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache) {
        // TODO: pre-process data
    }
}
impl ExtraData for MadfPlotExtraData {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache) {
        // TODO: pre-process data
    }
}
impl ExtraData for MmsfPlotExtraData {
    fn pre_process(&mut self, data: Handle<MeasurementData>, cache: &Cache) {
        // TODO: pre-process data
    }
}

impl<C: ExtraData> PlotInspector<C> {
    pub fn new(
        name: String,
        data: Handle<MeasurementData>,
        mut extra: C,
        cache: &Cache,
        event_loop: EventLoopProxy,
    ) -> Self {
        extra.pre_process(data, cache);
        Self {
            name,
            data,
            legend: Legend::default()
                .text_style(egui::TextStyle::Monospace)
                .background_alpha(1.0)
                .position(Corner::RightTop),
            plot_type: PlotType::Line,
            event_loop,
            extra,
            uuid: Uuid::new_v4(),
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

    fn plot_type_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Plot type:");
            ui.selectable_value(&mut self.plot_type, PlotType::Line, "Line");
            ui.selectable_value(&mut self.plot_type, PlotType::Bar, "Bar");
        });
    }
}

impl PlottingWidget for PlotInspector<MadfPlotExtraData> {
    fn uuid(&self) -> Uuid { self.uuid }

    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui, cache: &Cache) {
        let measurement = &cache.get_measurement_data(self.data).unwrap();
        let zenith = measurement.measured.madf_or_mmsf_zenith().unwrap();
        let azimuth = measurement.measured.madf_or_mmsf_azimuth().unwrap();
        let zenith_bin_width_rad = zenith.step_size.as_f32();
        self.plot_type_ui(ui);
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("microfacet normal: ");
                let mut opposite = self.extra.azimuth_m.wrap_to_tau().opposite();
                Self::angle_knob(
                    ui,
                    false,
                    &mut opposite,
                    azimuth.map(|x| x.value()).range_bound_inclusive(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.extra.azimuth_m,
                    azimuth.map(|x| x.value()).range_bound_inclusive(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                Self::debug_print_angle_pair(
                    self.extra.azimuth_m,
                    &azimuth,
                    ui,
                    "debug_print_φ_pair",
                );
            },
        );
        let data: Vec<_> = {
            let (starting, opposite) = measurement.adf_data_slice(self.extra.azimuth_m);

            // Data of the opposite azimuthal angle side of the slice, if exists.
            let data_opposite_part = opposite.map(|data| {
                data.iter()
                    .rev()
                    .zip(zenith.values_rev().map(|x| -x))
                    .map(|(y, x)| [x.as_f64(), *y as f64])
            });

            let data_starting_part = starting
                .iter()
                .zip(zenith.values())
                .map(|(y, x)| [x.as_f64(), *y as f64]);

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

    fn data_handle(&self) -> Handle<MeasurementData> { self.data }

    fn data_kind(&self) -> MeasurementKind { MeasurementKind::Madf }
}

impl PlottingWidget for PlotInspector<MmsfPlotExtraData> {
    fn uuid(&self) -> Uuid { self.uuid }

    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui, cache: &Cache) {
        let measurement = cache.get_measurement_data(self.data).unwrap();
        let zenith = measurement.measured.madf_or_mmsf_zenith().unwrap();
        let azimuth = measurement.measured.madf_or_mmsf_azimuth().unwrap();
        let zenith_bin_width_rad = zenith.step_size.value();
        self.plot_type_ui(ui);
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("microfacet normal: ");
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.extra.zenith_m,
                    zenith.range_bound_inclusive_f32(),
                    zenith.step_size,
                    48.0,
                    |v| format!("θ = {:>6.2}°", v.to_degrees()),
                );
                Self::angle_knob(
                    ui,
                    true,
                    &mut self.extra.azimuth_m,
                    azimuth.range_bound_inclusive_f32(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                {
                    Self::debug_print_angle_pair(
                        self.extra.azimuth_m,
                        &azimuth,
                        ui,
                        "debug_print_φ",
                    );
                    Self::debug_print_angle(self.extra.zenith_m, &zenith, ui, "debug_print_θ");
                }
            },
        );
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 48.0),
            egui::Layout::left_to_right(Align::Center),
            |ui| {
                ui.label("incident direction: ");
                let mut opposite = self.extra.azimuth_i.opposite();
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
                    &mut self.extra.azimuth_i,
                    azimuth.range_bound_inclusive_f32(),
                    azimuth.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                );
                #[cfg(debug_assertions)]
                Self::debug_print_angle_pair(
                    self.extra.azimuth_i,
                    &azimuth,
                    ui,
                    "debug_print_φ_pair",
                );
            },
        );
        let data: Vec<_> = {
            let (starting, opposite) = measurement.msf_data_slice(
                self.extra.azimuth_m,
                self.extra.zenith_m,
                self.extra.azimuth_i,
            );
            let data_opposite_part = opposite.map(|data| {
                data.iter()
                    .rev()
                    .zip(zenith.values_rev().map(|v| -v))
                    .map(|(y, x)| [x.as_f64(), *y as f64])
            });
            let data_starting_part = starting
                .iter()
                .zip(zenith.values())
                .map(|(y, x)| [x.as_f64(), *y as f64]);

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

    fn data_handle(&self) -> Handle<MeasurementData> { self.data }

    fn data_kind(&self) -> MeasurementKind { MeasurementKind::Mmsf }
}

impl PlottingWidget for PlotInspector<BsdfPlotExtraData> {
    fn uuid(&self) -> uuidUuid { self.uuid }

    fn name(&self) -> &str { self.name.as_str() }

    fn ui(&mut self, ui: &mut Ui, cache: &Cache) {
        let measurement = cache.get_measurement_data(self.data).unwrap();
        ui.horizontal(|ui| {
            ui.label("Plot mode:");
            ui.selectable_value(&mut self.extra.mode, BsdfPlotMode::Slice2D, "Slice 2D");
            ui.selectable_value(&mut self.extra.mode, BsdfPlotMode::Slice3D, "Slice 3D");
        });
        let is_3d = self.extra.mode == BsdfPlotMode::Slice3D;
        let bsdf_data = measurement.measured.bsdf_data().unwrap();
        let zenith_i = bsdf_data.params.emitter.zenith;
        let azimuth_i = bsdf_data.params.emitter.azimuth;
        let (zenith_o, azimuth_o) = match bsdf_data.params.collector.scheme {
            CollectorScheme::Partitioned { partition } => match partition {
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
                let r0 = Self::angle_knob(
                    ui,
                    true,
                    &mut self.extra.zenith_i,
                    zenith_i.range_bound_inclusive_f32(),
                    zenith_i.step_size,
                    48.0,
                    |v| format!("θ = {:>6.2}°", v.to_degrees()),
                )
                .changed();
                let r1 = Self::angle_knob(
                    ui,
                    true,
                    &mut self.extra.azimuth_i,
                    azimuth_i.range_bound_inclusive_f32(),
                    azimuth_i.step_size,
                    48.0,
                    |v| format!("φ = {:>6.2}°", v.to_degrees()),
                )
                .changed();
                #[cfg(debug_assertions)]
                {
                    Self::debug_print_angle(self.extra.zenith_i, &zenith_i, ui, "debug_print_θ");
                    Self::debug_print_angle_pair(
                        self.extra.azimuth_i,
                        &azimuth_i,
                        ui,
                        "debug_print_φ",
                    );
                }
                self.extra.changed |= r0 || r1;
            },
        );

        if !is_3d {
            ui.allocate_ui_with_layout(
                Vec2::new(ui.available_width(), 48.0),
                egui::Layout::left_to_right(Align::Center),
                |ui| {
                    ui.label("Outgoing direction: ");
                    let r0 = Self::angle_knob(
                        ui,
                        true,
                        &mut self.extra.azimuth_o,
                        azimuth_o.range_bound_inclusive_f32(),
                        azimuth_o.step_size,
                        48.0,
                        |v| format!("φ = {:>6.2}°", v.to_degrees()),
                    )
                    .changed();
                    #[cfg(debug_assertions)]
                    {
                        Self::debug_print_angle_pair(
                            self.extra.azimuth_o,
                            &azimuth_o,
                            ui,
                            "debug_print_φ",
                        );
                    }
                    self.extra.changed |= r0;
                },
            );
        }

        let zenith_i_count = zenith_i.step_count_wrapped();
        let zenith_i_idx = zenith_i.index_of(self.extra.zenith_i);
        let azimuth_i_idx = azimuth_i.index_of(self.extra.azimuth_i);
        let data_point = &bsdf_data.samples[azimuth_i_idx * zenith_i_count + zenith_i_idx];
        let spectrum = bsdf_data.params.emitter.spectrum;
        let wavelengths = spectrum
            .values()
            .map(|w| w.value() / 100.0)
            .collect::<Vec<_>>();
        let num_rays = bsdf_data.params.emitter.num_rays;

        if self.extra.changed {
            let (zenith_range, azimuth_range) = bsdf_data.params.collector.scheme.ranges();
            let count = bsdf_data.params.collector.scheme.total_sample_count();
            debug_assert_eq!(
                count,
                zenith_range.step_count_wrapped() * azimuth_range.step_count_wrapped()
            );
            let mut samples_per_wavelength = vec![vec![Vec3::ZERO; count]; wavelengths.len()];
            let zenith_range_step_count = zenith_range.step_count_wrapped();
            for (lambda_idx, samples) in samples_per_wavelength.iter_mut().enumerate() {
                for (azimuth_idx, azimuth) in azimuth_range.values_wrapped().enumerate() {
                    for (zenith_idx, zenith) in zenith_range.values_wrapped().enumerate() {
                        let idx = azimuth_idx * zenith_range_step_count + zenith_idx;
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
            //     label: Some(&format!("bsdf_view_buffer_{:?}", self.controls.view_id)),
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
            //     .send_event(VgonioEvent::BsdfViewer(BsdfViewerEvent::UpdateBuffer {
            //         id: self.controls.view_id,
            //         buffer: Some(buffer),
            //         count: samples_per_wavelength[0].len() as u32,
            //     }))
            //     .unwrap();
            self.extra.changed = false;
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
                CoordinatesFormatter::new(move |p, _| format!("λ = {:.0}nm", p.x * 100.0,)),
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
                            "Number of emitted rays that hit the surface; invariant over \
                             wavelength",
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

                    // Calculate the captured energy per wavelength by dividing the captured energy
                    // by the number of captured rays per wavelength
                    let e_captured = wavelengths
                        .iter()
                        .zip(data_point.stats.e_captured.iter())
                        .zip(data_point.stats.n_captured.iter())
                        .map(|((l, e), n)| [*l as f64, *e as f64 / *n as f64])
                        .collect::<Vec<_>>();

                    per_wavelength_plot.show(ui, |plot_ui| {
                        plot_ui.line(Line::new(n_captured.clone()).name("Nº Captured").width(2.0));
                        plot_ui.points(
                            Points::new(n_captured)
                                .shape(MarkerShape::Circle)
                                .radius(4.0)
                                .name("Nº Captured"),
                        );

                        plot_ui.line(Line::new(n_absorbed.clone()).name("Nº Absorbed").width(2.0));
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

                        plot_ui.line(Line::new(e_captured.clone()).name("E Captured").width(2.0));
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
                                        // and scale the percentage to the range [0, 2]
                                        Bar::new(i as f64 + 0.5, (*n as f64 / *total as f64) * 2.0)
                                            .width(1.0)
                                    })
                                    .collect(),
                            )
                            .name(format!("Nº of rays, λ = {}", lambda))
                            .element_formatter(Box::new(
                                |bar, _| -> String {
                                    format!(
                                        "bounce = {:.0}, number of rays = {:.0}%",
                                        bar.argument + 0.5,
                                        (bar.value / 2.0) * 100.0
                                    )
                                },
                            ))
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
                                        // and scale the percentage to the range [0, 2]
                                        Bar::new(i as f64 + 0.5, (*e as f64 / *total as f64) * 2.0)
                                            .width(1.0)
                                    })
                                    .collect(),
                            )
                            .name(format!("Energy, λ = {}", lambda))
                            .element_formatter(Box::new(
                                |bar, _| -> String {
                                    format!(
                                        "bounce = {:.0}, energy = {:.0}%",
                                        bar.argument + 0.5,
                                        (bar.value / 2.0) * 100.0
                                    )
                                },
                            ))
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
                        //             .send_event(VgonioEvent::BsdfViewer(BsdfViewerEvent::Rotate {
                        //                 id: self.extra.view_id,
                        //                 angle: delta_x / 256.0 * std::f32::consts::PI,
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

    fn data_handle(&self) -> Handle<MeasurementData> { self.data }

    fn data_kind(&self) -> MeasurementKind { MeasurementKind::Bsdf }
}

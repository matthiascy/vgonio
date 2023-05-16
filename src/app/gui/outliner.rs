use crate::{
    app::{
        cache::{Cache, Handle},
        gui::tools::{PlottingInspector, Tool},
    },
    measure::measurement::{MeasurementData, MeasurementDataSource, MeasurementKind},
    msurf::MicroSurface,
    units::LengthUnit,
};
use std::{collections::HashMap, rc::Weak};

/// States of one item in the outliner.
#[derive(Clone, Debug)]
pub struct PerMicroSurfaceState {
    /// The name of the micro surface.
    pub name: String,
    /// Whether the micro surface is visible.
    pub visible: bool,
    /// The scale factor of the micro surface.
    pub scale: f32,
    /// The length unit of the micro surface.
    pub unit: LengthUnit,
    /// The lowest value of the micro surface.
    pub min: f32,
    /// The highest value of the micro surface.
    pub max: f32,
    /// Offset along the Y axis without scaling.
    pub y_offset: f32,
}

/// Outliner is a widget that displays the scene graph of the current scene.
///
/// It will reads the micro-surfaces from the cache and display them in a tree
/// structure. The user can toggle the visibility of the micro surfaces.
pub struct Outliner {
    /// States of the micro surfaces, indexed by their ids.
    surfaces: HashMap<Handle<MicroSurface>, (SurfaceCollapsableHeader, PerMicroSurfaceState)>,
    /// States of the measured data.
    measurements: Vec<(Weak<MeasurementData>, MeasuredDataCollapsableHeader)>,
    /// Plotting inspectors, linked to the measurement data they are inspecting.
    plotting_inspectors: Vec<(Weak<MeasurementData>, PlottingInspector)>,
}

impl Default for Outliner {
    fn default() -> Self { Self::new() }
}

impl Outliner {
    /// Creates a new outliner.
    pub fn new() -> Self {
        Self {
            surfaces: HashMap::new(),
            measurements: Default::default(),
            plotting_inspectors: vec![],
        }
    }

    /// Returns an iterator over all the visible micro surfaces.
    pub fn visible_surfaces(&self) -> Vec<(&Handle<MicroSurface>, &PerMicroSurfaceState)> {
        self.surfaces
            .iter()
            .filter(|(_, (_, s))| s.visible)
            .map(|(id, (_, s))| (id, s))
            .collect()
    }

    pub fn any_visible_surfaces(&self) -> bool { self.surfaces.iter().any(|(_, (_, s))| s.visible) }

    /// Updates the list of micro surfaces.
    pub fn update_surfaces(&mut self, surfs: &[Handle<MicroSurface>], cache: &Cache) {
        for hdl in surfs {
            if let std::collections::hash_map::Entry::Vacant(e) = self.surfaces.entry(*hdl) {
                let record = cache.get_micro_surface_record(*hdl).unwrap();
                let surf = cache.get_micro_surface(*e.key()).unwrap();
                e.insert((
                    SurfaceCollapsableHeader { selected: false },
                    PerMicroSurfaceState {
                        name: record.name().to_string(),
                        visible: false,
                        scale: 1.0,
                        unit: surf.unit,
                        min: surf.min,
                        max: surf.max,
                        y_offset: 0.0,
                    },
                ));
            }
        }
    }

    /// Updates the list of measurement data.
    pub fn update_measurement_data(
        &mut self,
        measurements: &[Handle<MeasurementData>],
        cache: &Cache,
    ) {
        for meas in measurements {
            let data = cache.get_measurement_data(*meas).unwrap();
            if !self.measurements.iter().any(|(d, _)| d.ptr_eq(&data)) {
                self.measurements.push((
                    data,
                    MeasuredDataCollapsableHeader {
                        selected: false,
                        show_plot: false,
                    },
                ));
            }
        }
    }
}

pub struct SurfaceCollapsableHeader {
    selected: bool,
}

impl SurfaceCollapsableHeader {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut PerMicroSurfaceState) {
        let id = ui.make_persistent_id(&state.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    ui.horizontal(|ui| {
                        if ui.selectable_label(self.selected, &state.name).clicked() {
                            self.selected = !self.selected;
                        }
                    })
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut state.visible, "");
                })
            })
            .body(|ui| {
                // Scale
                egui::Grid::new("surface_collapsable_header_grid")
                    .num_columns(3)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new("Min:"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.min, state.unit)));
                        ui.end_row();
                        ui.add(egui::Label::new("Max:"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.max, state.unit)));
                        ui.end_row();
                        ui.add(egui::Label::new(format!("Y Offset ({}):", state.unit)))
                            .on_hover_text(
                                "Offset along the Y axis without scaling. (Visual only - does not \
                                 affect the actual surface)",
                            );
                        ui.add(
                            egui::Slider::new(&mut state.y_offset, -100.0..=100.0)
                                .trailing_fill(true),
                        );
                        ui.end_row();
                        ui.add(egui::Label::new(""));
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .add(egui::Button::new("Median"))
                                .on_hover_text(
                                    "Sets the Y offset to median value of the surface heights",
                                )
                                .clicked()
                            {
                                state.y_offset = -(state.min + state.max) * 0.5;
                            }
                            if ui
                                .add(egui::Button::new("Ground"))
                                .on_hover_text(
                                    "Adjusts its position so that the minimum height value is at \
                                     the ground level.",
                                )
                                .clicked()
                            {
                                state.y_offset = -state.min;
                            }
                        });
                        ui.end_row();
                        ui.add(egui::Label::new("Scale:")).on_hover_text(
                            "Scales the surface visually. Doest not affect the actual surface.",
                        );
                        ui.add(egui::Slider::new(&mut state.scale, 0.05..=1.5).trailing_fill(true));
                    });
            });
    }
}

struct MeasuredDataCollapsableHeader {
    selected: bool,
    show_plot: bool,
}

impl MeasuredDataCollapsableHeader {
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        data: Weak<MeasurementData>,
        plots: &mut Vec<(Weak<MeasurementData>, PlottingInspector)>,
    ) {
        let meas_data = data.upgrade().unwrap();
        let id = ui.make_persistent_id(&meas_data.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(self.selected, &meas_data.name)
                            .clicked()
                        {
                            self.selected = !self.selected;
                        }
                    })
                });
            })
            .body(|ui| {
                egui::Grid::new("measurement_data_body")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Type:");
                        ui.label(format!("{}", meas_data.kind));
                        ui.end_row();
                        ui.label("Source:");
                        match meas_data.source {
                            MeasurementDataSource::Loaded(_) => {
                                ui.label("loaded");
                            }
                            MeasurementDataSource::Measured(_) => {
                                ui.label("measured");
                            }
                        }
                        ui.end_row();

                        if meas_data.kind == MeasurementKind::MicrofacetAreaDistribution {
                            ui.label("θ:");
                            ui.label(format!(
                                "{:.2}° ~ {:.2}°, every {:.2}°",
                                meas_data.zenith.start.to_degrees(),
                                meas_data.zenith.end.to_degrees(),
                                meas_data.zenith.bin_width.to_degrees(),
                            ));
                            ui.end_row();
                            ui.label("φ:");
                            ui.label(format!(
                                "{:.2}° ~ {:.2}°, every {:.2}°",
                                meas_data.azimuth.start.to_degrees(),
                                meas_data.azimuth.end.to_degrees(),
                                meas_data.azimuth.bin_width.to_degrees(),
                            ));
                            ui.end_row();
                        }
                    });
                ui.add_space(5.0);
                if ui.button("Plot").clicked() {
                    self.show_plot = true;
                    if !plots.iter_mut().any(|p| p.0.ptr_eq(&data)) {
                        plots.push((
                            data.clone(),
                            PlottingInspector::new(meas_data.name.clone(), meas_data.clone()),
                        ));
                    }
                }
            });
    }
}

// GUI related functions
impl Outliner {
    /// Creates the ui for the outliner.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("MicroSurfaces")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for (_, (hdr, state)) in self.surfaces.iter_mut() {
                        hdr.ui(ui, state);
                    }
                });
            });
        egui::CollapsingHeader::new("Measurements")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for (data, hdr) in self.measurements.iter_mut() {
                        hdr.ui(ui, data.clone(), &mut self.plotting_inspectors);
                    }
                })
            });

        for (data, plot) in self.plotting_inspectors.iter_mut() {
            let open = &mut self
                .measurements
                .iter_mut()
                .find(|(d, _)| d.ptr_eq(data))
                .unwrap()
                .1
                .show_plot;
            plot.show(ui.ctx(), open);
        }
    }
}

use crate::{
    app::{
        cache::{Cache, Handle},
        gfx::GpuContext,
        gui::{
            bsdf_viewer::BsdfViewer,
            docking::Dockable,
            tools::{
                BsdfPlottingControls, MadfPlottingControls, MmsfPlottingControls,
                PlottingInspector, PlottingWidget,
            },
            VgonioEventLoop,
        },
    },
    measure::measurement::{MeasuredData, MeasurementData, MeasurementDataSource, MeasurementKind},
};
use egui::{Ui, WidgetText};
use egui_dock::NodeIndex;
use std::{
    any::Any,
    collections::HashMap,
    rc::Weak,
    sync::{Arc, RwLock},
};
use vgcore::units::LengthUnit;
use vgsurf::MicroSurface;

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
    /// The offset along the y-axis.
    pub height_offset: f32,
    /// Size of the micro-surface.
    pub size: (u32, u32),
}

/// Outliner is a widget that displays the scene graph of the current scene.
///
/// It will reads the micro-surfaces from the cache and display them in a tree
/// structure. The user can toggle the visibility of the micro surfaces.
pub struct Outliner {
    /// The unique id of the outliner.
    uuid: uuid::Uuid,
    gpu_ctx: Arc<GpuContext>,
    event_loop: VgonioEventLoop,
    bsdf_viewer: Arc<RwLock<BsdfViewer>>,
    /// States of the micro surfaces, indexed by their ids.
    surfaces: HashMap<Handle<MicroSurface>, (SurfaceCollapsableHeader, PerMicroSurfaceState)>,
    /// States of the measured data.
    measurements: Vec<(Weak<MeasurementData>, MeasuredDataCollapsableHeader)>,
    /// Plotting inspectors, linked to the measurement data they are inspecting.
    plotting_inspectors: Vec<(Weak<MeasurementData>, Box<dyn PlottingWidget>)>,
}

impl Outliner {
    /// Creates a new outliner.
    pub fn new(
        gpu_ctx: Arc<GpuContext>,
        bsdf_viewer: Arc<RwLock<BsdfViewer>>,
        event_loop: VgonioEventLoop,
    ) -> Self {
        log::info!("Creating outliner");
        Self {
            gpu_ctx,
            event_loop,
            bsdf_viewer,
            surfaces: HashMap::new(),
            measurements: Default::default(),
            plotting_inspectors: vec![],
            uuid: uuid::Uuid::new_v4(),
        }
    }

    pub fn surfaces(
        &self,
    ) -> &HashMap<Handle<MicroSurface>, (SurfaceCollapsableHeader, PerMicroSurfaceState)> {
        &self.surfaces
    }

    /// Returns an iterator over all the visible micro surfaces.
    pub fn visible_surfaces(&self) -> Vec<(Handle<MicroSurface>, PerMicroSurfaceState)> {
        self.surfaces
            .iter()
            .filter(|(_, (_, s))| s.visible)
            .map(|(id, (_, s))| (*id, s.clone()))
            .collect()
    }

    pub fn any_visible_surfaces(&self) -> bool { self.surfaces.iter().any(|(_, (_, s))| s.visible) }

    /// Updates the list of micro surfaces.
    pub fn update_surfaces(&mut self, surfs: &[Handle<MicroSurface>], cache: &Cache) {
        for hdl in surfs {
            if let std::collections::hash_map::Entry::Vacant(e) = self.surfaces.entry(*hdl) {
                let record = cache.get_micro_surface_record(*hdl).unwrap();
                let surf = cache.get_micro_surface(*e.key()).unwrap();
                let mesh = cache.get_micro_surface_mesh(record.mesh).unwrap();
                e.insert((
                    SurfaceCollapsableHeader { selected: false },
                    PerMicroSurfaceState {
                        name: record.name().to_string(),
                        visible: false,
                        scale: 1.0,
                        unit: surf.unit,
                        min: surf.min,
                        max: surf.max,
                        height_offset: mesh.height_offset,
                        size: (surf.rows as u32, surf.cols as u32),
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

fn header_content(ui: &mut egui::Ui, name: &str, selected: &mut bool) {
    ui.vertical_centered_justified(|ui| {
        ui.horizontal(|ui| {
            if ui.selectable_label(*selected, name).clicked() {
                *selected = !*selected;
            }
        })
    });
}

pub struct SurfaceCollapsableHeader {
    selected: bool,
}

impl SurfaceCollapsableHeader {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut PerMicroSurfaceState) {
        let id = ui.make_persistent_id(&state.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                header_content(ui, &state.name, &mut self.selected);
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
                        ui.add(egui::Label::new("Size:"));
                        ui.add(egui::Label::new(format!(
                            "{} x {}",
                            state.size.0, state.size.1
                        )));
                        ui.end_row();

                        ui.add(egui::Label::new("Min:"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.min, state.unit)));
                        ui.end_row();

                        ui.add(egui::Label::new("Max:"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.max, state.unit)));
                        ui.end_row();

                        ui.add(egui::Label::new("Scale:")).on_hover_text(
                            "Scales the surface visually. Doest not affect the actual surface.",
                        );
                        ui.add(
                            egui::Slider::new(&mut state.scale, 0.005..=1.5).trailing_fill(true),
                        );
                        ui.end_row();
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
        plots: &mut Vec<(Weak<MeasurementData>, Box<dyn PlottingWidget>)>,
        bsdf_viewer: Arc<RwLock<BsdfViewer>>,
        gpu: Arc<GpuContext>,
        event_loop: VgonioEventLoop,
    ) {
        let measured = data.upgrade().unwrap();
        let id = ui.make_persistent_id(&measured.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                header_content(ui, &measured.name, &mut self.selected);
            })
            .body(|ui| {
                let measurement_kind = measured.kind();
                egui::Grid::new("measurement_data_body")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Type:");
                        ui.label(format!("{}", measurement_kind));
                        ui.end_row();
                        ui.label("Source:");
                        match measured.source {
                            MeasurementDataSource::Loaded(_) => {
                                ui.label("loaded");
                            }
                            MeasurementDataSource::Measured(_) => {
                                ui.label("measured");
                            }
                        }
                        ui.end_row();

                        if measurement_kind == MeasurementKind::MicrofacetAreaDistribution
                            || measurement_kind == MeasurementKind::MicrofacetMaskingShadowing
                        {
                            let zenith = measured.measured.madf_or_mmsf_zenith().unwrap();
                            let azimuth = measured.measured.madf_or_mmsf_azimuth().unwrap();
                            ui.label("θ:");
                            ui.label(format!(
                                "{:.2}° ~ {:.2}°, every {:.2}°",
                                zenith.start.to_degrees(),
                                zenith.stop.to_degrees(),
                                zenith.step_size.to_degrees(),
                            ));
                            ui.end_row();
                            #[cfg(debug_assertions)]
                            {
                                ui.label("θ bins:");
                                ui.label(format!("{}", zenith.step_count_wrapped()));
                                ui.end_row()
                            }
                            ui.label("φ:");
                            ui.label(format!(
                                "{:.2}° ~ {:.2}°, every {:.2}°",
                                azimuth.start.to_degrees(),
                                azimuth.stop.to_degrees(),
                                azimuth.step_size.to_degrees(),
                            ));
                            ui.end_row();
                            #[cfg(debug_assertions)]
                            {
                                ui.label("φ bins:");
                                ui.label(format!("{}", azimuth.step_count_wrapped()));
                                ui.end_row()
                            }
                        }
                    });
                ui.add_space(5.0);
                if ui.button("Plot").clicked() {
                    self.show_plot = true;
                    if !plots.iter_mut().any(|p| p.0.ptr_eq(&data)) {
                        match &measured.measured {
                            MeasuredData::Madf(_) => {
                                plots.push((
                                    data.clone(),
                                    Box::new(PlottingInspector::new(
                                        measured.name.clone(),
                                        measured.clone(),
                                        MadfPlottingControls::default(),
                                        gpu,
                                        event_loop,
                                    )),
                                ));
                            }
                            MeasuredData::Mmsf(_) => {
                                plots.push((
                                    data.clone(),
                                    Box::new(PlottingInspector::new(
                                        measured.name.clone(),
                                        measured.clone(),
                                        MmsfPlottingControls::default(),
                                        gpu,
                                        event_loop,
                                    )),
                                ));
                            }
                            MeasuredData::Bsdf(_) => {
                                plots.push((
                                    data.clone(),
                                    Box::new(PlottingInspector::new(
                                        measured.name.clone(),
                                        measured.clone(),
                                        BsdfPlottingControls::new(
                                            bsdf_viewer.write().unwrap().create_new_view(),
                                        ),
                                        gpu,
                                        event_loop,
                                    )),
                                ));
                            }
                        }
                    }
                }
            });
    }
}

// GUI related functions
impl Outliner {
    /// Creates the ui for the outliner.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Outliner");
        ui.separator();
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
                        hdr.ui(
                            ui,
                            data.clone(),
                            &mut self.plotting_inspectors,
                            self.bsdf_viewer.clone(),
                            self.gpu_ctx.clone(),
                            self.event_loop.clone(),
                        );
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

impl Dockable for Outliner {
    fn title(&self) -> WidgetText { WidgetText::from("Outliner") }

    fn ui(&mut self, ui: &mut Ui) { self.ui(ui); }

    fn uuid(&self) -> uuid::Uuid { self.uuid }
}

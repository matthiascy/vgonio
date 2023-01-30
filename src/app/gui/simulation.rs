use crate::{
    app::{
        cache::Cache,
        gui::{gizmo::VgonioGizmo, VgonioEvent},
    },
    measure::measurement::{BsdfMeasurement, Measurement},
};
use egui_gizmo::{GizmoMode, GizmoOrientation};
use glam::Mat4;
use std::{
    cell::RefCell,
    fmt::{Display, Formatter},
    path::Path,
    sync::Arc,
};
use winit::event_loop::EventLoopProxy;

/// Helper enum used in GUI to specify the radius of the emitter/detector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum RadiusMode {
    Fixed,
    Auto,
}

/// Helper enum used in GUI to specify the spherical partition mode of the
/// emitter/detector.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum PartitionMode {
    Area,
    ProjectedArea,
    Angle,
}

impl Display for PartitionMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PartitionMode::Area => "Equal Area",
                PartitionMode::ProjectedArea => "Equal Projected Area",
                PartitionMode::Angle => "Equal Angle",
            }
        )
    }
}

/// Helper struct used in GUI to specify the spherical partition parameters of
/// the emitter/detector.
#[derive(Debug, Copy, Clone)]
struct Partition {
    mode: PartitionMode,
    zenith: (f32, f32, f32),
    azimuth: (f32, f32, f32),
}

// TODO(yang): refactor this because of the change in measurement description.
/// A panel for configuration of the simulation parameters.
pub struct SimulationPanel {
    /// Measurement description.
    desc: Measurement,

    /// Selected surface index.
    selected_surface_index: usize,

    /// Emitter radius mode.
    emitter_radius_mode: RadiusMode,

    /// Emitter radius.
    emitter_radius: f32,

    // /// Emitter partition.
    // emitter_partition: Partition,
    /// Collector radius mode.
    collector_radius_mode: RadiusMode,

    /// Collector radius.
    collector_radius: f32,

    /// Collector partition.
    collector_partition: Partition,

    simulation_started: bool,

    simulation_progress: f32,

    cache: Arc<RefCell<Cache>>,
}

impl SimulationPanel {
    pub fn new(cache: Arc<RefCell<Cache>>) -> Self {
        Self {
            desc: Measurement {
                kind: crate::measure::measurement::MeasurementKind::Bsdf(BsdfMeasurement::default()),
                surfaces: vec![],
            },
            selected_surface_index: 0,
            emitter_radius_mode: RadiusMode::Auto,
            emitter_radius: 0.0,
            collector_radius_mode: RadiusMode::Auto,
            collector_radius: 0.0,
            collector_partition: Partition {
                mode: PartitionMode::Area,
                zenith: (0.0, 0.0, 0.0),
                azimuth: (0.0, 0.0, 0.0),
            },
            simulation_started: false,
            simulation_progress: 0.0,
            cache,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        egui::Window::new("Simulation Panel")
            .open(open)
            .title_bar(true)
            .collapsible(true)
            .resizable(false)
            .show(ctx, |ui| {
                self.ui(ui);
            });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // TODO:
        // egui::Grid::new("simulation_pane_grid")
        //     .num_columns(2)
        //     .show(ui, |ui| {
        //         ui.add(egui::Label::new("measurement"));
        //         egui::ComboBox::from_id_source("measurement_choice")
        //             .selected_text(format!("{:?}", self.desc.kind))
        //             .show_ui(ui, |ui| {
        //                 ui.selectable_value(
        //                     &mut self.desc.kind,
        //                     MeasurementKind::Bsdf(BsdfKind::InPlaneBrdf),
        //                     "in-plane brdf",
        //                 );
        //                 ui.selectable_value(&mut self.desc.kind,
        // MeasurementKind::Ndf, "ndf");             });
        //         ui.end_row();
        //
        //         ui.add(egui::Label::new("incident medium"));
        //         egui::ComboBox::from_id_source("incident_medium_choice")
        //             .selected_text(format!("{:?}",
        // self.desc.incident_medium))             .show_ui(ui, |ui| {
        //                 ui.selectable_value(&mut self.desc.incident_medium,
        // Medium::Air, "Air");                 ui.selectable_value(
        //                     &mut self.desc.incident_medium,
        //                     Medium::Copper,
        //                     "Copper",
        //                 );
        //                 ui.selectable_value(
        //                     &mut self.desc.incident_medium,
        //                     Medium::Aluminium,
        //                     "Aluminium",
        //                 );
        //                 ui.selectable_value(
        //                     &mut self.desc.incident_medium,
        //                     Medium::Vacuum,
        //                     "Vacuum",
        //                 );
        //             });
        //         ui.end_row();
        //
        //         ui.add(egui::Label::new("transmitted medium"));
        //         egui::ComboBox::from_id_source("transmitted_medium_choice")
        //             .selected_text(format!("{:?}",
        // self.desc.transmitted_medium))             .show_ui(ui, |ui|
        // {                 ui.selectable_value(&mut
        // self.desc.transmitted_medium, Medium::Air, "Air");
        //                 ui.selectable_value(
        //                     &mut self.desc.transmitted_medium,
        //                     Medium::Copper,
        //                     "Copper",
        //                 );
        //                 ui.selectable_value(
        //                     &mut self.desc.transmitted_medium,
        //                     Medium::Aluminium,
        //                     "Aluminium",
        //                 );
        //                 ui.selectable_value(
        //                     &mut self.desc.transmitted_medium,
        //                     Medium::Vacuum,
        //                     "Vacuum",
        //                 );
        //             });
        //         ui.end_row();
        //
        //         ui.add(egui::Label::new("surface"));
        //         if self.desc.surfaces.is_empty() {
        //             ui.add(egui::Label::new("no surface loaded!"));
        //         } else {
        //             egui::ComboBox::from_id_source("surface_choice")
        //                 .selected_text(format!(
        //                     "{:?}",
        //                     self.desc.surfaces[self.selected_surface_index]
        //                         .file_name()
        //                         .unwrap()
        //                 ))
        //                 .show_ui(ui, |ui| {
        //                     for (i, surface) in
        // self.desc.surfaces.iter().enumerate() {
        // ui.selectable_value(                             &mut
        // self.selected_surface_index,                             i,
        //                             self.desc.surfaces[i].to_str().unwrap(),
        //                         );
        //                     }
        //                 });
        //         }
        //         ui.end_row();
        //
        //         // ui.add(egui::Label::new("tracing method"));
        //         // ui.selectable_value(
        //         //     &mut self.desc.tracing_method,
        //         //     RayTracingMethod::Standard,
        //         //     "standard",
        //         // );
        //         // ui.selectable_value(
        //         //     &mut self.desc.tracing_method,
        //         //     RayTracingMethod::Grid,
        //         //     "  grid  ",
        //         // );
        //         // ui.end_row();
        //     });
        //
        // egui::CollapsingHeader::new("emitter")
        //     .default_open(true)
        //     .show(ui, |ui| {
        //         egui::Grid::new("emitter_grid")
        //             .num_columns(2)
        //             .show(ui, |ui| {
        //                 ui.label("num rays");
        //                 ui.add(egui::DragValue::new(&mut
        // self.desc.emitter.num_rays));                 ui.end_row();
        //
        //                 ui.label("max bounces");
        //                 ui.add(egui::DragValue::new(&mut
        // self.desc.emitter.max_bounces));
        // ui.end_row();
        //
        //                 ui.label("radius");
        //                 ui.horizontal(|ui| {
        //                     ui.add(egui::DragValue::new(&mut
        // self.emitter_radius));
        // ui.selectable_value(                         &mut
        // self.emitter_radius_mode,
        // RadiusMode::Auto,                         "auto",
        //                     );
        //                     ui.selectable_value(
        //                         &mut self.emitter_radius_mode,
        //                         RadiusMode::Fixed,
        //                         "fixed",
        //                     );
        //                 });
        //                 ui.end_row();
        //
        //                 ui.label("spectrum (nm)");
        //                 ui.horizontal(|ui| {
        //                     ui.add(
        //                         egui::DragValue::new(&mut
        // self.desc.emitter.spectrum.start.value)
        // .prefix("start: ")
        // .clamp_range(380.0..=780.0),                     );
        //                     ui.add(
        //                         egui::DragValue::new(&mut
        // self.desc.emitter.spectrum.stop.value)
        // .prefix("stop: ")
        // .clamp_range(self.desc.emitter.spectrum.start..=780.0),
        //                     );
        //                     ui.add(
        //                         egui::DragValue::new(&mut
        // self.desc.emitter.spectrum.step_size.value)
        // .prefix("step: ")
        // .clamp_range(0.1..=400.0),                     );
        //                 });
        //
        //                 ui.end_row();
        //
        //                 ui.label("zenith (θ)");
        //                 ui.horizontal(|ui| {
        //                     ui.add(input(
        //                         &mut self.desc.emitter.zenith.start,
        //                         "start: ",
        //                         Some(0.0..=360.0),
        //                     ));
        //                     ui.add(input(
        //                         &mut self.desc.emitter.zenith.stop,
        //                         "stop: ",
        //                         Some(self.desc.emitter.zenith.start..=360.0),
        //                     ));
        //                     ui.add(input(
        //                         &mut self.desc.emitter.zenith.step_size,
        //                         "step: ",
        //                         Some(1.0..=360.0),
        //                     ));
        //                 });
        //                 ui.end_row();
        //
        //                 ui.label("azimuth (φ)");
        //                 ui.horizontal(|ui| {
        //                     ui.add(input(
        //                         &mut self.desc.emitter.azimuth.start,
        //                         "start: ",
        //                         Some(0.0..=360.0),
        //                     ));
        //                     ui.add(input(
        //                         &mut self.desc.emitter.azimuth.stop,
        //                         "stop: ",
        //
        // Some(self.desc.emitter.azimuth.start..=360.0),
        // ));                     ui.add(input(
        //                         &mut self.desc.emitter.azimuth.step_size,
        //                         "step: ",
        //                         Some(1.0..=360.0),
        //                     ));
        //                 });
        //                 ui.end_row();
        //             });
        //     });
        //
        // egui::CollapsingHeader::new("collector")
        //     .default_open(true)
        //     .show(ui, |ui| {
        //         egui::Grid::new("collector_grid")
        //             .num_columns(2)
        //             .show(ui, |ui| {
        //                 ui.label("radius");
        //                 ui.horizontal(|ui| {
        //                     ui.add(egui::DragValue::new(&mut
        // self.collector_radius));
        // ui.selectable_value(                         &mut
        // self.collector_radius_mode,
        // RadiusMode::Auto,                         "auto",
        //                     );
        //                     ui.selectable_value(
        //                         &mut self.collector_radius_mode,
        //                         RadiusMode::Fixed,
        //                         "fixed",
        //                     );
        //                 });
        //                 ui.end_row();
        //
        //                 ui.label("shape");
        //
        // egui::ComboBox::from_id_source("collector_shape_choice")
        //                     .selected_text(format!("{:?}",
        // self.desc.collector.shape))                     .show_ui(ui,
        // |ui| {                         ui.selectable_value(
        //                             &mut self.desc.collector.shape,
        //                             SphericalDomain::WholeSphere,
        //                             "Whole hemisphere",
        //                         );
        //                         ui.selectable_value(
        //                             &mut self.desc.collector.shape,
        //                             SphericalDomain::UpperHemisphere,
        //                             "Upper hemisphere",
        //                         );
        //                         ui.selectable_value(
        //                             &mut self.desc.collector.shape,
        //                             SphericalDomain::LowerHemisphere,
        //                             "Lower hemisphere",
        //                         );
        //                     });
        //                 ui.end_row();
        //
        //                 ui.label("partition");
        //
        // egui::ComboBox::from_id_source("collector_partition_mode")
        //                     .selected_text(format!("{}",
        // self.collector_partition.mode))
        // .show_ui(ui, |ui| {
        // ui.selectable_value(                             &mut
        // self.collector_partition.mode,
        // PartitionMode::Angle,                             "Equal
        // Angle",                         );
        //                         ui.selectable_value(
        //                             &mut self.collector_partition.mode,
        //                             PartitionMode::Area,
        //                             "Equal Area",
        //                         );
        //                         ui.selectable_value(
        //                             &mut self.collector_partition.mode,
        //                             PartitionMode::ProjectedArea,
        //                             "Equal Projected Area",
        //                         );
        //                     });
        //                 ui.end_row();
        //
        //                 ui.label("        zenith (θ)");
        //                 ui.horizontal(|ui| {
        //                     ui.add(input(
        //                         &mut self.collector_partition.zenith.0,
        //                         "start: ",
        //                         Some(0.0..=360.0),
        //                     ));
        //                     ui.add(input(
        //                         &mut self.collector_partition.zenith.1,
        //                         "stop: ",
        //
        // Some(self.collector_partition.zenith.0..=360.0),
        // ));
        //
        //                     if self.collector_partition.mode ==
        // PartitionMode::Angle {                         ui.add(input(
        //                             &mut self.collector_partition.zenith.2,
        //                             "step: ",
        //                             Some(0.0..=360.0),
        //                         ));
        //                     } else {
        //                         ui.add(input(
        //                             &mut self.collector_partition.zenith.2,
        //                             "count: ",
        //                             None,
        //                         ));
        //                     }
        //                 });
        //                 ui.end_row();
        //
        //                 ui.label("        azimuth (φ)");
        //                 ui.horizontal(|ui| {
        //                     ui.add(input(
        //                         &mut self.collector_partition.azimuth.0,
        //                         "start: ",
        //                         Some(0.0..=360.0),
        //                     ));
        //                     ui.add(input(
        //                         &mut self.collector_partition.azimuth.1,
        //                         "stop: ",
        //
        // Some(self.collector_partition.azimuth.0..=360.0),
        // ));                     ui.add(input(
        //                         &mut self.collector_partition.azimuth.2,
        //                         "step: ",
        //                         Some(0.0..=360.0),
        //                     ));
        //                 });
        //                 ui.end_row();
        //             })
        //     });
        //
        // ui.separator();
        // ui.horizontal_wrapped(|ui| {
        //     if ui.button("Simulate").clicked() {
        //         self.simulation_started = true;
        //         // match self.desc.measurement_kind {
        //         //     MeasurementKind::Bsdf(kind) => {
        //         //         match self.desc.tracing_method {
        //         //             RayTracingMethod::Standard => {
        //         //                 bsdf::measure_bsdf_embree_rt(&self.desc, )
        //         //             }
        //         //             RayTracingMethod::Grid => {}
        //         //         }
        //         //     }
        //         //     MeasurementKind::Ndf => {
        //         //         todo!()
        //         //     }
        //         // }
        //         todo!()
        //     }
        //
        //     ui.add(
        //         egui::ProgressBar::new(self.simulation_progress)
        //             .show_percentage()
        //             .animate(self.simulation_started),
        //     );
        // });
    }
}

/// Workspace for the measurement.
pub struct SimulationWorkspace {
    view_gizmo_opened: bool,

    simulation_pane_opened: bool,

    /// Whether the visual grid (ground) is visible.
    visual_grid_enabled: bool,

    surface_visible: bool,

    /// The gizmo used to indicate the camera's orientation.
    view_gizmo: VgonioGizmo,

    /// The simulation parameters.
    pub(crate) simulation_panel: SimulationPanel,

    /// The scale factor of loaded micro-surface in the scene.
    surface_scale_factor: f32,

    /// Event loop proxy of user defined events. See [`UserEvent`].
    event_loop: EventLoopProxy<VgonioEvent>,
}

impl SimulationWorkspace {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>, cache: Arc<RefCell<Cache>>) -> Self {
        Self {
            view_gizmo_opened: false,
            simulation_pane_opened: false,
            view_gizmo: VgonioGizmo::new(GizmoMode::Translate, GizmoOrientation::Global),
            simulation_panel: SimulationPanel::new(cache),
            visual_grid_enabled: true,
            surface_scale_factor: 1.0,
            event_loop,
            surface_visible: true,
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        self.ui(ctx);
        self.view_gizmo.show(ctx, &mut self.view_gizmo_opened);
        self.simulation_panel
            .show(ctx, &mut self.simulation_pane_opened);
    }

    pub fn update_surface_list(&mut self, list: &Vec<&Path>) {
        if !list.is_empty() {
            for &surface in list {
                let buf = surface.to_owned();
                if !self.simulation_panel.desc.surfaces.contains(&buf) {
                    self.simulation_panel.desc.surfaces.push(buf);
                }
            }
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.view_gizmo.update_matrices(model, view, proj)
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::Area::new("view_controls")
            .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
            .show(ctx, |ui| {
                egui::Grid::new("controls_grid")
                    .num_columns(4)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Scale factor:");
                        {
                            let res = ui.add(egui::Slider::new(
                                &mut self.surface_scale_factor,
                                0.05..=1.2,
                            ));
                            if res.changed()
                                && self
                                    .event_loop
                                    .send_event(VgonioEvent::UpdateSurfaceScaleFactor(
                                        self.surface_scale_factor,
                                    ))
                                    .is_err()
                            {
                                log::warn!("[EVENT] Failed to send SetScaleFactor event");
                            }
                        }

                        ui.label("Visual Grid");
                        {
                            let res = ui.add(super::widgets::toggle(&mut self.visual_grid_enabled));
                            if res.changed()
                                && self.event_loop.send_event(VgonioEvent::ToggleGrid).is_err()
                            {
                                log::warn!("[EVENT] Failed to send ToggleGrid event");
                            }
                        }

                        ui.label("Surface Visibility");
                        {
                            let res = ui.add(super::widgets::toggle(&mut self.surface_visible));
                            if res.changed()
                                && self
                                    .event_loop
                                    .send_event(VgonioEvent::ToggleSurfaceVisibility)
                                    .is_err()
                            {
                                log::warn!("[EVENT] Failed to send ToggleSurfaceVisibility event");
                            }
                        }
                        ui.end_row();
                    });
            });

        egui::TopBottomPanel::bottom("simulation_menu_bar").show(ctx, |ui| {
            egui::trace!(ui);
            egui::menu::bar(ui, |ui| {
                if ui.button("Simulation Pane").clicked() {
                    self.simulation_pane_opened = !self.simulation_pane_opened;
                }
            });
        });
    }
}

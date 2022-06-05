use crate::acq::bxdf::BxdfKind;
use crate::acq::desc::{MeasurementDesc, MeasurementKind};
use crate::acq::util::SphericalShape;
use crate::acq::Medium;
use crate::app::gui::gizmo::VgonioGizmo;
use crate::app::gui::ui::Workspace;
use crate::app::gui::{UserEvent, VisualDebugTool};
use egui_gizmo::{GizmoMode, GizmoOrientation};
use glam::Mat4;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::sync::Arc;
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

/// A panel for configuration of the simulation parameters.
struct SimulationPanel {
    /// Measurement description.
    measurement_desc: MeasurementDesc,

    /// Selected surface index.
    selected_surface_index: usize,

    /// Emitter radius mode.
    emitter_radius_mode: RadiusMode,

    /// Emitter radius.
    emitter_radius: f32,

    /// Emitter partition.
    emitter_partition: Partition,

    /// Collector radius mode.
    collector_radius_mode: RadiusMode,

    /// Collector radius.
    collector_radius: f32,

    /// Collector partition.
    collector_partition: Partition,

    simulation_started: bool,

    simulation_progress: f32,
}

impl SimulationPanel {
    pub fn new() -> Self {
        Self {
            measurement_desc: MeasurementDesc::default(),
            selected_surface_index: 0,
            emitter_radius_mode: RadiusMode::Auto,
            emitter_radius: 0.0,
            emitter_partition: Partition {
                mode: PartitionMode::Area,
                zenith: (0.0, 0.0, 0.0),
                azimuth: (0.0, 0.0, 0.0),
            },
            collector_radius_mode: RadiusMode::Auto,
            collector_radius: 0.0,
            collector_partition: Partition {
                mode: PartitionMode::Area,
                zenith: (0.0, 0.0, 0.0),
                azimuth: (0.0, 0.0, 0.0),
            },
            simulation_started: false,
            simulation_progress: 0.0,
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
        egui::Grid::new("simulation_pane_grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.add(egui::Label::new("measurement"));
                egui::ComboBox::from_id_source("measurement_choice")
                    .selected_text(format!("{:?}", self.measurement_desc.measurement_kind))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.measurement_desc.measurement_kind,
                            MeasurementKind::Bxdf {
                                kind: BxdfKind::InPlane,
                            },
                            "InPlane BRDF",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.measurement_kind,
                            MeasurementKind::Ndf,
                            "NDF",
                        );
                    });
                ui.end_row();

                ui.add(egui::Label::new("incident medium"));
                egui::ComboBox::from_id_source("incident_medium_choice")
                    .selected_text(format!("{:?}", self.measurement_desc.incident_medium))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.measurement_desc.incident_medium,
                            Medium::Air,
                            "Air",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.incident_medium,
                            Medium::Copper,
                            "Copper",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.incident_medium,
                            Medium::Aluminium,
                            "Aluminium",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.incident_medium,
                            Medium::Vacuum,
                            "Vacuum",
                        );
                    });
                ui.end_row();

                ui.add(egui::Label::new("transmitted medium"));
                egui::ComboBox::from_id_source("transmitted_medium_choice")
                    .selected_text(format!("{:?}", self.measurement_desc.transmitted_medium))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.measurement_desc.transmitted_medium,
                            Medium::Air,
                            "Air",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.transmitted_medium,
                            Medium::Copper,
                            "Copper",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.transmitted_medium,
                            Medium::Aluminium,
                            "Aluminium",
                        );
                        ui.selectable_value(
                            &mut self.measurement_desc.transmitted_medium,
                            Medium::Vacuum,
                            "Vacuum",
                        );
                    });
                ui.end_row();

                ui.add(egui::Label::new("surface"));
                if self.measurement_desc.surfaces.is_empty() {
                    ui.add(egui::Label::new("no surface loaded!"));
                } else {
                    egui::ComboBox::from_id_source("surface_choice")
                        .selected_text(format!(
                            "{:?}",
                            self.measurement_desc.surfaces[self.selected_surface_index]
                                .file_name()
                                .unwrap()
                        ))
                        .show_ui(ui, |ui| {
                            for (i, surface) in self.measurement_desc.surfaces.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.selected_surface_index,
                                    i,
                                    self.measurement_desc.surfaces[i].to_str().unwrap(),
                                );
                            }
                        });
                }
                ui.end_row();
            });

        egui::CollapsingHeader::new("emitter")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("emitter_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("num rays");
                        ui.add(egui::DragValue::new(
                            &mut self.measurement_desc.emitter.num_rays,
                        ));
                        ui.end_row();

                        ui.label("max bounces");
                        ui.add(egui::DragValue::new(
                            &mut self.measurement_desc.emitter.max_bounces,
                        ));
                        ui.end_row();

                        ui.label("radius");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut self.emitter_radius));
                            ui.selectable_value(
                                &mut self.emitter_radius_mode,
                                RadiusMode::Auto,
                                "auto",
                            );
                            ui.selectable_value(
                                &mut self.emitter_radius_mode,
                                RadiusMode::Fixed,
                                "fixed",
                            );
                        });
                        ui.end_row();

                        ui.label("spectrum");
                        ui.horizontal(|ui| {
                            ui.label("start");
                            ui.add(egui::DragValue::new(
                                &mut self.measurement_desc.emitter.spectrum.start,
                            ));
                            ui.label("stop");
                            ui.add(egui::DragValue::new(
                                &mut self.measurement_desc.emitter.spectrum.stop,
                            ));
                            ui.label("step");
                            ui.add(egui::DragValue::new(
                                &mut self.measurement_desc.emitter.spectrum.step,
                            ));
                        });
                        ui.end_row();

                        ui.label("partition");
                        egui::ComboBox::from_id_source("emitter_partition_mode")
                            .selected_text(format!("{}", self.emitter_partition.mode))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.emitter_partition.mode,
                                    PartitionMode::Angle,
                                    "Equal Angle",
                                );
                                ui.selectable_value(
                                    &mut self.emitter_partition.mode,
                                    PartitionMode::Area,
                                    "Equal Area",
                                );
                                ui.selectable_value(
                                    &mut self.emitter_partition.mode,
                                    PartitionMode::ProjectedArea,
                                    "Equal Projected Area",
                                );
                            });
                        ui.end_row();

                        ui.label("        zenith (θ)");
                        ui.horizontal(|ui| {
                            ui.label("start");
                            ui.add(egui::DragValue::new(&mut self.emitter_partition.zenith.0));
                            ui.label("stop");
                            ui.add(egui::DragValue::new(&mut self.emitter_partition.zenith.1));

                            if self.emitter_partition.mode == PartitionMode::Angle {
                                ui.label("step");
                                ui.add(egui::DragValue::new(&mut self.emitter_partition.zenith.2));
                            } else {
                                ui.label("count");
                                ui.add(egui::DragValue::new(&mut self.emitter_partition.zenith.2));
                            }
                        });
                        ui.end_row();

                        ui.label("        azimuth (φ)");
                        ui.horizontal(|ui| {
                            ui.label("start");
                            ui.add(egui::DragValue::new(&mut self.emitter_partition.azimuth.0));
                            ui.label("stop");
                            ui.add(egui::DragValue::new(&mut self.emitter_partition.azimuth.1));
                            ui.label("step");
                            ui.add(egui::DragValue::new(&mut self.emitter_partition.azimuth.2));
                        });
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("collector")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("collector_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("radius");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut self.collector_radius));
                            ui.selectable_value(
                                &mut self.collector_radius_mode,
                                RadiusMode::Auto,
                                "auto",
                            );
                            ui.selectable_value(
                                &mut self.collector_radius_mode,
                                RadiusMode::Fixed,
                                "fixed",
                            );
                        });
                        ui.end_row();

                        ui.label("shape");
                        egui::ComboBox::from_id_source("collector_shape_choice")
                            .selected_text(format!("{:?}", self.measurement_desc.collector.shape))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.measurement_desc.collector.shape,
                                    SphericalShape::WholeSphere,
                                    "Whole hemisphere",
                                );
                                ui.selectable_value(
                                    &mut self.measurement_desc.collector.shape,
                                    SphericalShape::UpperHemisphere,
                                    "Upper hemisphere",
                                );
                                ui.selectable_value(
                                    &mut self.measurement_desc.collector.shape,
                                    SphericalShape::LowerHemisphere,
                                    "Lower hemisphere",
                                );
                            });
                        ui.end_row();

                        ui.label("partition");
                        egui::ComboBox::from_id_source("collector_partition_mode")
                            .selected_text(format!("{}", self.collector_partition.mode))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.collector_partition.mode,
                                    PartitionMode::Angle,
                                    "Equal Angle",
                                );
                                ui.selectable_value(
                                    &mut self.collector_partition.mode,
                                    PartitionMode::Area,
                                    "Equal Area",
                                );
                                ui.selectable_value(
                                    &mut self.collector_partition.mode,
                                    PartitionMode::ProjectedArea,
                                    "Equal Projected Area",
                                );
                            });
                        ui.end_row();

                        ui.label("        zenith (θ)");
                        ui.horizontal(|ui| {
                            ui.label("start");
                            ui.add(egui::DragValue::new(&mut self.collector_partition.zenith.0));
                            ui.label("stop");
                            ui.add(egui::DragValue::new(&mut self.collector_partition.zenith.1));

                            if self.collector_partition.mode == PartitionMode::Angle {
                                ui.label("step");
                                ui.add(egui::DragValue::new(
                                    &mut self.collector_partition.zenith.2,
                                ));
                            } else {
                                ui.label("count");
                                ui.add(egui::DragValue::new(
                                    &mut self.collector_partition.zenith.2,
                                ));
                            }
                        });
                        ui.end_row();

                        ui.label("        azimuth (φ)");
                        ui.horizontal(|ui| {
                            ui.label("start");
                            ui.add(egui::DragValue::new(
                                &mut self.collector_partition.azimuth.0,
                            ));
                            ui.label("stop");
                            ui.add(egui::DragValue::new(
                                &mut self.collector_partition.azimuth.1,
                            ));
                            ui.label("step");
                            ui.add(egui::DragValue::new(
                                &mut self.collector_partition.azimuth.2,
                            ));
                        });
                        ui.end_row();
                    })
            });

        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui.button("Simulate").clicked() {
                self.simulation_started = true;
                println!("TODO: Simulating");
            }

            ui.add(
                egui::ProgressBar::new(self.simulation_progress)
                    .show_percentage()
                    .animate(self.simulation_started),
            );
        });
    }
}

/// Workspace for the measurement.
pub struct SimulationWorkspace {
    view_gizmo_opened: bool,

    visual_debugger_opened: bool,

    simulation_pane_opened: bool,

    /// Whether the visual grid (ground) is visible.
    visual_grid_enabled: bool,

    /// The gizmo used to indicate the camera's orientation.
    view_gizmo: VgonioGizmo,

    /// The visual debugger.
    pub(crate) visual_debug_tool: VisualDebugTool,

    /// The simulation parameters.
    pub(crate) simulation_panel: SimulationPanel,

    /// The scale factor of loaded micro-surface in the scene.
    surface_scale_factor: f32,

    /// Event loop proxy of user defined events. See [`UserEvent`].
    event_loop: Arc<EventLoopProxy<UserEvent>>,
}

impl Workspace for SimulationWorkspace {
    fn name(&self) -> &str {
        "Simulation"
    }

    fn show(&mut self, ctx: &egui::Context) {
        self.ui(ctx);
        self.view_gizmo.show(ctx, &mut self.view_gizmo_opened);
        self.visual_debug_tool
            .show(ctx, &mut self.visual_debugger_opened);
        self.simulation_panel
            .show(ctx, &mut self.simulation_pane_opened);
    }
}

impl SimulationWorkspace {
    pub fn new(event_loop: Arc<EventLoopProxy<UserEvent>>) -> Self {
        Self {
            view_gizmo_opened: false,
            visual_debugger_opened: false,
            simulation_pane_opened: false,
            view_gizmo: VgonioGizmo::new(GizmoMode::Translate, GizmoOrientation::Global),
            visual_debug_tool: VisualDebugTool::new(event_loop.clone()),
            simulation_panel: SimulationPanel::new(),
            visual_grid_enabled: true,
            surface_scale_factor: 1.0,
            event_loop,
        }
    }

    pub fn update_surface_list(&mut self, list: &Vec<PathBuf>) {
        if !list.is_empty() {
            for surface in list {
                if !self
                    .simulation_panel
                    .measurement_desc
                    .surfaces
                    .contains(surface)
                {
                    self.simulation_panel
                        .measurement_desc
                        .surfaces
                        .push(surface.clone());
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
                                    .send_event(UserEvent::UpdateSurfaceScaleFactor(
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
                                && self.event_loop.send_event(UserEvent::ToggleGrid).is_err()
                            {
                                log::warn!("[EVENT] Failed to send ToggleGrid event");
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
                if ui.button("Visual Debugger").clicked() {
                    self.visual_debugger_opened = !self.visual_debugger_opened;
                }
            });
        });
    }
}

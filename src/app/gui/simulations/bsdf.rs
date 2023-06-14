use crate::{
    app::gui::{misc, simulations::SurfaceSelector, MeasureEvent, VgonioEvent, VgonioEventLoop},
    measure::{
        bsdf::BsdfKind,
        emitter::RegionShape,
        measurement::{BsdfMeasurementParams, Radius},
        Collector, CollectorScheme, Emitter,
    },
    units::{deg, mm, rad, Radians},
    Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalDomain,
    SphericalPartition,
};
use std::hash::Hash;
use winit::event_loop::EventLoopProxy;

impl BsdfKind {
    /// Creates the UI for selecting the BSDF kind.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_source(id_source)
            .selected_text(format!("{}", self))
            .show_ui(ui, |ui| {
                ui.selectable_value(self, BsdfKind::Brdf, "BRDF");
                ui.selectable_value(self, BsdfKind::Btdf, "BTDF");
                ui.selectable_value(self, BsdfKind::Bssdf, "BSSDF");
                ui.selectable_value(self, BsdfKind::Bssrdf, "BSSRDF");
                ui.selectable_value(self, BsdfKind::Bssrdf, "BSSRDF");
            });
    }
}

impl Medium {
    /// Creates the UI for selecting the medium.
    pub fn selectable_ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        egui::ComboBox::from_id_source(id_source)
            .selected_text(format!("{:?}", self))
            .show_ui(ui, |ui| {
                ui.selectable_value(self, Medium::Air, "Air");
                ui.selectable_value(self, Medium::Copper, "Copper");
                ui.selectable_value(self, Medium::Aluminium, "Aluminium");
                ui.selectable_value(self, Medium::Vacuum, "Vacuum");
            });
    }
}

impl Radius {
    /// Creates the UI radius input.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let is_auto = self.is_auto();
        ui.horizontal(|ui| {
            if ui
                .selectable_label(is_auto, "Auto")
                .on_hover_text(
                    "Automatically set the distance according to the dimensions of the surface.",
                )
                .clicked()
            {
                *self = Radius::Auto(mm!(0.0));
            }

            let fixed_response = ui.selectable_label(!is_auto, "Fixed");

            if !is_auto {
                ui.add(
                    egui::DragValue::new(&mut self.value_mut().value)
                        .speed(1.0)
                        .suffix("mm")
                        .clamp_range(0.0..=f32::INFINITY),
                );
            }

            if fixed_response.clicked() {
                *self = Radius::Fixed(mm!(self.value().value));
            }
        });
    }
}

impl RegionShape {
    /// Creates the UI for parameterizing the region shape.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.columns(1, |uis| {
            let ui = &mut uis[0];
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(self.is_spherical_cap(), "Cap")
                    .on_hover_text("The emitter is a spherical cap.")
                    .clicked()
                {
                    *self = RegionShape::default_spherical_cap();
                }

                if ui
                    .selectable_label(self.is_spherical_rect(), "Rect")
                    .on_hover_text("The emitter is a spherical shell.")
                    .clicked()
                {
                    *self = RegionShape::default_spherical_rect();
                }

                if ui
                    .selectable_label(self.is_disk(), "Disk")
                    .on_hover_text("The emitter is a disk.")
                    .clicked()
                {
                    *self = RegionShape::Disk {
                        radius: Default::default(),
                    }
                }
            });
            ui.end_row();

            if self.is_spherical_cap() {
                let zenith = self.cap_zenith_mut().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Zenith range θ: ");
                    ui.add(misc::drag_angle(zenith, "start: "));
                });
            }

            if self.is_spherical_rect() {
                let zenith = self.rect_zenith_mut().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Zenith range θ: ");
                    ui.add(misc::drag_angle(zenith.0, "start: "));
                    ui.add(misc::drag_angle(zenith.1, "end: "));
                });
                ui.end_row();

                let azimuth = self.rect_azimuth_mut().unwrap();
                ui.horizontal(|ui| {
                    ui.label("Azimuth range φ: ");
                    ui.add(misc::drag_angle(azimuth.0, "start: "));
                    ui.add(misc::drag_angle(azimuth.1, "end: "));
                });
            }
            ui.end_row();
        });
    }
}

impl Emitter {
    /// Creates the UI for parameterizing the emitter.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Emitter")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("emitter_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Number of rays: ");
                        ui.add(
                            egui::DragValue::new(&mut self.num_rays)
                                .speed(1.0)
                                .clamp_range(1..=100_000_000),
                        );
                        ui.end_row();

                        ui.label("Max. bounces: ");
                        ui.add(
                            egui::DragValue::new(&mut self.max_bounces)
                                .speed(1.0)
                                .clamp_range(1..=100),
                        );
                        ui.end_row();

                        ui.label("Distance:")
                            .on_hover_text("Distance from emitter to the surface.");
                        self.radius.ui(ui);
                        ui.end_row();

                        ui.label("Azimuthal range φ: ");
                        self.azimuth.ui(ui);
                        ui.end_row();

                        ui.label("Zenith range θ: ");
                        self.zenith.ui(ui);
                        ui.end_row();

                        ui.label("Region shape: ");
                        self.shape.ui(ui);
                        ui.end_row();

                        ui.label("Wavelength range: ");
                        self.spectrum.ui(ui);
                        ui.end_row();
                    });
            });
    }
}

impl SphericalDomain {
    /// Creates the UI for parameterizing the spherical domain.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .selectable_label(*self == SphericalDomain::Upper, "Upper")
                .on_hover_text("The upper hemisphere.")
                .clicked()
            {
                *self = SphericalDomain::Upper;
            }

            if ui
                .selectable_label(*self == SphericalDomain::Lower, "Lower")
                .on_hover_text("The lower hemisphere.")
                .clicked()
            {
                *self = SphericalDomain::Lower;
            }

            if ui
                .selectable_label(*self == SphericalDomain::Whole, "Whole")
                .on_hover_text("The whole sphere.")
                .clicked()
            {
                *self = SphericalDomain::Whole;
            }
        });
    }
}

impl CollectorScheme {
    /// Creates the UI for parameterizing the collector scheme.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Collector Scheme")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("collector_scheme_grid")
                    .striped(true)
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Type: ");
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(self.is_partitioned(), "Partitioned")
                                .on_hover_text("Partition the collector into sub-regions.")
                                .clicked()
                            {
                                *self = CollectorScheme::default_partition();
                            }

                            if ui
                                .selectable_label(self.is_single_region(), "Single region")
                                .on_hover_text("Use a single region for the collector.")
                                .clicked()
                            {
                                *self = CollectorScheme::default_single_region();
                            }
                        });
                        ui.end_row();

                        ui.label("Domain: ");
                        self.domain_mut().ui(ui);
                        ui.end_row();

                        if self.is_partitioned() {
                            ui.label("Partitioning: ");
                            let partition = self.partition_mut().unwrap();
                            ui.horizontal(|ui| {
                                let is_equal_angle = partition.is_equal_angle();
                                let is_equal_area = partition.is_equal_area();
                                let is_equal_projected_area = partition.is_equal_projected_area();
                                if ui.selectable_label(is_equal_angle, "Equal angle").clicked() {
                                    *partition = SphericalPartition::EqualAngle {
                                        zenith: RangeByStepSizeInclusive::zero_to_half_pi(
                                            Radians::from_degrees(5.0),
                                        ),
                                        azimuth: RangeByStepSizeInclusive::zero_to_tau(
                                            Radians::from_degrees(15.0),
                                        ),
                                    };
                                }

                                if ui.selectable_label(is_equal_area, "Equal area").clicked() {
                                    *partition = SphericalPartition::EqualArea {
                                        zenith: RangeByStepCountInclusive::new(
                                            Radians::from_degrees(0.0),
                                            Radians::from_degrees(90.0),
                                            17,
                                        ),
                                        azimuth: RangeByStepSizeInclusive::zero_to_tau(
                                            Radians::from_degrees(15.0),
                                        ),
                                    };
                                }

                                if ui
                                    .selectable_label(
                                        is_equal_projected_area,
                                        "Equal projected area",
                                    )
                                    .clicked()
                                {
                                    *partition = SphericalPartition::EqualProjectedArea {
                                        zenith: RangeByStepCountInclusive::new(
                                            Radians::from_degrees(0.0),
                                            Radians::from_degrees(90.0),
                                            17,
                                        ),
                                        azimuth: RangeByStepSizeInclusive::zero_to_tau(
                                            Radians::from_degrees(15.0),
                                        ),
                                    };
                                }
                            });
                            ui.end_row();

                            match partition {
                                SphericalPartition::EqualAngle { zenith, azimuth } => {
                                    ui.label("Azimuthal range φ: ");
                                    azimuth.ui(ui);
                                    ui.end_row();

                                    ui.label("Zenith range θ: ");
                                    zenith.ui(ui);
                                }
                                SphericalPartition::EqualArea { zenith, azimuth } => {
                                    ui.label("Azimuthal range φ: ");
                                    azimuth.ui(ui);
                                    ui.end_row();

                                    ui.label("Zenith range θ: ");
                                    zenith.ui(ui);
                                }
                                SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                                    ui.label("Azimuthal range φ: ");
                                    azimuth.ui(ui);
                                    ui.end_row();

                                    ui.label("Zenith range θ: ");
                                    zenith.ui(ui);
                                }
                            }
                        } else {
                            ui.label("Region Shape: ");
                            self.shape_mut().unwrap().ui(ui);
                            ui.end_row();

                            ui.label("Azimuthal range φ: ");
                            self.azimuth_mut().unwrap().ui(ui);
                            ui.end_row();

                            ui.label("Zenith range θ: ");
                            self.zenith_mut().unwrap().ui(ui);
                        }
                        ui.end_row();
                    });
            });
    }
}

impl Collector {
    /// Creates the UI for parameterizing the collector.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Collector")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("collector_grid")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Distance:")
                            .on_hover_text("Distance from collector to the surface.");
                        self.radius.ui(ui);
                        ui.end_row();
                    });
                self.scheme.ui(ui);
            });
    }
}

pub struct BsdfSimulation {
    pub params: BsdfMeasurementParams,
    pub(crate) selector: SurfaceSelector,
    event_loop: VgonioEventLoop,
}

impl BsdfSimulation {
    /// Creates a new BSDF simulation UI.
    pub fn new(event_loop: VgonioEventLoop) -> Self {
        Self {
            params: BsdfMeasurementParams::default(),
            selector: Default::default(),
            event_loop,
        }
    }

    /// UI for BSDF simulation parameters.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("bsdf_sim_grid")
            .striped(true)
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("BSDF type:");
                self.params.kind.selectable_ui("bsdf_kind_choice", ui);
                ui.end_row();

                ui.label("Incident medium:");
                self.params
                    .incident_medium
                    .selectable_ui("incident_medium_choice", ui);
                ui.end_row();

                ui.label("Surface medium:");
                self.params
                    .transmitted_medium
                    .selectable_ui("transmitted_medium_choice", ui);
                ui.end_row();

                ui.label("Micro-surfaces: ");
                self.selector.ui("micro_surface_selector", ui);
                ui.end_row();
            });

        self.params.emitter.ui(ui);
        self.params.collector.ui(ui);
        if ui.button("Simulate").clicked() {
            self.event_loop
                .send_event(VgonioEvent::Measure(MeasureEvent::Bsdf {
                    params: self.params,
                    surfaces: self.selector.selected.clone().into_iter().collect(),
                }))
                .unwrap();
        }
    }
}

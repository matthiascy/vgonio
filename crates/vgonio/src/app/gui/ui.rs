#[cfg(feature = "fitting")]
use crate::fitting::{
    FittedModel, FittingProblem, FittingProblemKind, MicrofacetBrdfFittingProblem,
    MicrofacetDistributionFittingProblem,
};
use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{
            data::PropertyData,
            event::{EventLoopProxy, OutlinerEvent, SurfaceViewerEvent, VgonioEvent},
            file_drop::FileDragDrop,
            // gizmo::NavigationGizmo,
            icons,
            measurement::MeasurementDialog,
            notify::{NotifyKind, NotifySystem},
            outliner::OutlinerItem,
            plotter::{PlotInspector, PlottingWidget},
            state::GuiRenderer,
            theme::ThemeKind,
            tools::{SamplingInspector, Scratch, Tools},
            DebuggingInspector,
        },
        Config,
    },
    measure::Measurement,
};
use crate::{
    fitting::{FittingReport, MfdFittingData},
    measure::{
        bsdf::{MeasuredBsdfData, L0},
        mfd::{MeasuredMsfData, MeasuredNdfData},
    },
};
use base::{
    io::{CompressionScheme, FileEncoding},
    range::RangeByStepSizeInclusive,
    MeasurementKind,
};
use bxdf::brdf::BxdfFamily;
use gfxkit::context::GpuContext;
use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};
use surf::MicroSurface;

use super::{docking::DockSpace, event::EventResponse};

/// Implementation of the GUI for vgonio application.
pub struct VgonioGui {
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy,

    gpu_ctx: Arc<GpuContext>,

    cache: Cache,

    // pub simulation_workspace: SimulationWorkspace, // TODO: make private, simplify access
    /// The drag and drop state.
    drag_drop: FileDragDrop,

    // Gizmo inside the viewport for navigating the scene.
    // navigator: NavigationGizmo,
    /// Notification system.
    notif: NotifySystem,

    /// Plotters are windows that contain plots.
    plotters: Vec<(bool, Box<dyn PlottingWidget>)>,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    /// New measurement dialog.
    measurement: MeasurementDialog,

    pub properties: Arc<RwLock<PropertyData>>,

    /// Docking system for the UI.
    pub(crate) dock_space: DockSpace,
}

impl VgonioGui {
    pub fn new(
        event_loop: EventLoopProxy,
        config: Arc<Config>,
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        // bsdf_viewer: Arc<RwLock<BsdfViewer>>,
        cache: Cache,
    ) -> Self {
        log::info!("Initializing UI ...");
        let properties = Arc::new(RwLock::new(PropertyData::new()));
        Self {
            config,
            event_loop: event_loop.clone(),
            tools: Tools::new(event_loop.clone(), gpu.clone(), gui.clone()),
            cache: cache.clone(),
            drag_drop: FileDragDrop::new(event_loop.clone()),
            // navigator: NavigationGizmo::new(GizmoOrientation::Global),
            measurement: MeasurementDialog::new(
                event_loop.clone(),
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                cache.clone(),
            ),
            plotters: vec![],
            dock_space: DockSpace::default_layout(
                gui.clone(),
                cache,
                properties.clone(),
                event_loop,
            ),
            properties,
            gpu_ctx: gpu,
            notif: NotifySystem::new(),
        }
    }

    /// Handles a user event.
    ///
    /// Returns [`EventResponse::Ignored`] if the event was not handled,
    /// otherwise returns [`EventResponse::Handled`].
    pub fn on_user_event(&mut self, event: VgonioEvent) -> EventResponse {
        match &event {
            VgonioEvent::OpenFiles(paths) => {
                self.on_open_files(paths);
                EventResponse::Handled
            }
            VgonioEvent::Notify { kind, text, time } => {
                self.notif.notify(*kind, text.clone(), *time as f64);
                EventResponse::Handled
            }
            VgonioEvent::SurfaceViewer(SurfaceViewerEvent::Create { uuid, .. }) => {
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                self.measurement.update_surface_viewers(&[*uuid]);
                EventResponse::Ignored(event)
            }
            VgonioEvent::Outliner(outliner_event) => match outliner_event {
                OutlinerEvent::SelectItem(item) => {
                    let mut properties = self.properties.write().unwrap();
                    properties.on_item_selected(*item);
                    EventResponse::Handled
                }
                OutlinerEvent::RemoveItem(item) => {
                    let mut properties = self.properties.write().unwrap();
                    self.cache.write(|cache| match item {
                        OutlinerItem::MicroSurface(handle) => {
                            properties.surfaces.remove(handle);
                            cache.unload_micro_surface(*handle).unwrap();
                        }
                        _ => {
                            todo!("Remove measured data from cache")
                        }
                    });
                    EventResponse::Handled
                }
            },
            VgonioEvent::ExportMeasurement(meas) => {
                use rfd::AsyncFileDialog;
                let dir = std::env::current_dir().unwrap();
                let task = AsyncFileDialog::new().set_directory(dir).save_file();
                let event_loop = self.event_loop.clone();
                let hdl = std::thread::spawn(move || pollster::block_on(task)).join();
                if let Ok(Some(hdl)) = hdl {
                    self.cache.read(|cache| {
                        let measured = cache.get_measurement(*meas).unwrap();
                        crate::io::write_single_measured_data_to_file(
                            measured,
                            FileEncoding::Binary,
                            CompressionScheme::Zlib,
                            Some(512),
                            hdl.inner(),
                        )
                        .map_err(|err| {
                            log::error!("Failed to write measured data to file: {}", err);
                        })
                        .unwrap_or_default();
                        event_loop.send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: format!("Export measurement to {:?}", hdl.path()),
                            time: 3.0,
                        });
                    });
                }
                EventResponse::Handled
            }
            VgonioEvent::Graphing {
                kind,
                data,
                independent,
            } => {
                let plotter_win_idx = self
                    .plotters
                    .iter()
                    .position(|(_, p)| p.measurement_data_handle() == *data);

                if plotter_win_idx.is_some() && *independent {
                    self.plotters[plotter_win_idx.unwrap()].0 = true;
                } else {
                    let prop = self.properties.read().unwrap();
                    let plotter = match kind {
                        MeasurementKind::Bsdf => Box::new(PlotInspector::new_bsdf(
                            prop.measured.get(data).unwrap().name.clone(),
                            *data,
                            self.cache.clone(),
                            self.properties.clone(),
                            self.event_loop.clone(),
                        )),
                        MeasurementKind::Ndf => Box::new(PlotInspector::new_adf(
                            prop.measured.get(data).unwrap().name.clone(),
                            *data,
                            self.cache.clone(),
                            self.properties.clone(),
                            self.event_loop.clone(),
                        )),
                        MeasurementKind::Msf => Box::new(PlotInspector::new_msf(
                            prop.measured.get(data).unwrap().name.clone(),
                            *data,
                            self.cache.clone(),
                            self.properties.clone(),
                            self.event_loop.clone(),
                        )),
                        MeasurementKind::Sdf => Box::new(PlotInspector::new_sdf(
                            prop.measured.get(data).unwrap().name.clone(),
                            *data,
                            self.cache.clone(),
                            self.properties.clone(),
                            self.event_loop.clone(),
                        )),
                        _ => {
                            unreachable!("Unsupported measurement kind: {:?}", kind)
                        }
                    };
                    if *independent {
                        self.plotters.push((true, plotter));
                    } else {
                        self.dock_space.add_existing_widget(plotter)
                    }
                }
                EventResponse::Handled
            }
            #[cfg(feature = "fitting")]
            VgonioEvent::Fitting { kind, data, scale } => {
                match kind {
                    FittingProblemKind::Bxdf {
                        family,
                        distro,
                        isotropy,
                    } => {
                        match family {
                            BxdfFamily::Microfacet => {
                                let report = self.cache.read(|cache| {
                                    let measured_brdf_data = cache
                                        .get_measurement(*data)
                                        .unwrap()
                                        .measured
                                        .downcast_ref::<MeasuredBsdfData>()
                                        .unwrap();
                                    let problem = MicrofacetBrdfFittingProblem::new(
                                        measured_brdf_data.brdf_at(L0).unwrap(),
                                        distro.unwrap(),
                                        RangeByStepSizeInclusive::new(0.001, 1.0, 0.01),
                                        L0,
                                        &cache.iors,
                                    );
                                    problem.lsq_lm_fit(*isotropy)
                                });
                                report.print_fitting_report();
                                // TODO: update the fitted models
                            }
                            _ => unimplemented!("Fitting BxDF family: {:?}", family),
                        }
                    }
                    FittingProblemKind::Mfd { model, isotropy } => {
                        let mut prop = self.properties.write().unwrap();
                        let fitted = &mut prop.measured.get_mut(data).unwrap().fitted;
                        if fitted.contains(kind, Some(*scale), *isotropy) {
                            log::debug!("Already fitted, skipping");
                            return EventResponse::Handled;
                        }
                        let report = self.cache.read(|cache| {
                            let measurement = cache.get_measurement(*data).unwrap();
                            log::debug!(
                                "Fitting MFD with {:?} through {:?}",
                                model,
                                measurement.measured.kind()
                            );
                            let data = match measurement.measured.kind() {
                                MeasurementKind::Msf => MfdFittingData::Msf(
                                    measurement
                                        .measured
                                        .downcast_ref::<MeasuredMsfData>()
                                        .unwrap(),
                                ),
                                MeasurementKind::Ndf => MfdFittingData::Ndf(
                                    measurement
                                        .measured
                                        .downcast_ref::<MeasuredNdfData>()
                                        .unwrap(),
                                ),
                                _ => {
                                    log::error!(
                                        "Measurement kind not supported for fitting NDF or MSF."
                                    );
                                    return FittingReport::empty();
                                }
                            };
                            let problem =
                                MicrofacetDistributionFittingProblem::new(data, *model, *scale);
                            problem.lsq_lm_fit(*isotropy)
                        });
                        report.print_fitting_report();
                        if let Some(model) = report.best_model() {
                            fitted.push(FittedModel::Ndf(model.clone_box(), *scale));
                        } else {
                            log::warn!("No model fitted");
                        }
                    }
                }
                EventResponse::Handled
            }
            VgonioEvent::SmoothSurface { .. } => EventResponse::Handled,
            _ => EventResponse::Ignored(event),
        }
    }

    // pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4)
    // {     self.navigator.update_matrices(model, view, proj);
    // }

    pub fn show(&mut self, ctx: &egui::Context, theme_kind: ThemeKind) {
        egui::TopBottomPanel::top("vgonio_top_panel")
            .exact_height(28.0)
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    self.main_menu(ui, theme_kind);
                });
            });

        self.dock_space.show(ctx, self.properties.clone());
        self.tools.show(ctx);
        self.drag_drop.show(ctx);
        // self.navigator.show(ctx);
        self.measurement.show(ctx);
        self.notif.show(ctx);
        for (is_open, plotter) in &mut self.plotters {
            plotter.show(ctx, is_open);
        }
    }

    pub fn on_open_files(&mut self, files: &[rfd::FileHandle]) {
        log::info!("Process UI opening files: {:?}", files);
        let (surfaces, measurements) = self.open_files(files);
        self.cache.read(|cache| {
            let mut properties = self.properties.write().unwrap();
            properties.update_surfaces(&surfaces, cache);
            properties.update_measurement_data(&measurements, cache);
            self.measurement.update_surface_selector(&surfaces, cache);
        });
        self.event_loop.send_event(VgonioEvent::SurfaceViewer(
            SurfaceViewerEvent::UpdateSurfaceList { surfaces },
        ));
    }

    fn main_menu(&mut self, ui: &mut egui::Ui, kind: ThemeKind) {
        ui.set_height(28.0);
        let icon = match kind {
            ThemeKind::Dark => icons::VGONIO_MENU_DARK,
            ThemeKind::Light => icons::VGONIO_MENU_LIGHT,
        };

        ui.menu_image_button(icon.as_image(), |ui| {
            if ui.button("About").clicked() {
                self.event_loop.send_event(VgonioEvent::Notify {
                    kind: NotifyKind::Info,
                    text: "TODO: about".to_string(),
                    time: 0.0,
                });
            }

            ui.menu_button("New", |ui| {
                if ui.button("Measurement").clicked() {
                    self.measurement.open();
                }
                if ui.button("Micro-surface").clicked() {
                    self.event_loop.send_event(VgonioEvent::Notify {
                        kind: NotifyKind::Info,
                        text: "TODO: new height field".to_string(),
                        time: 3.0,
                    });
                }
            });
            if ui.button("Open...").clicked() {
                use rfd::AsyncFileDialog;
                let dir = self
                    .config
                    .user_data_dir()
                    .unwrap_or_else(|| self.config.sys_data_dir());
                let task = AsyncFileDialog::new().set_directory(dir).pick_files();
                let event_loop = self.event_loop.clone();
                std::thread::spawn(move || {
                    pollster::block_on(async {
                        let file_handles = task.await;
                        if let Some(hds) = file_handles {
                            event_loop.send_event(VgonioEvent::OpenFiles(hds));
                        }
                    })
                });
            }
            ui.menu_button("Recent...", |ui| {
                for i in 0..10 {
                    if ui.button(format!("item {i}")).clicked() {
                        self.event_loop.send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: format!("TODO: open recent item {i}"),
                            time: 3.0,
                        });
                    }
                }
            });

            ui.add_space(6.0);

            {
                if ui.button("Save...").clicked() {
                    self.event_loop.send_event(VgonioEvent::Notify {
                        kind: NotifyKind::Info,
                        text: "TODO: save".into(),
                        time: 3.0,
                    });
                }
            }

            ui.menu_button("Edit", |ui| {
                {
                    if ui.button("     Undo").clicked() {
                        self.event_loop.send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: "TODO: undo".into(),
                            time: 3.0,
                        });
                    }
                    if ui.button("     Redo").clicked() {
                        self.event_loop.send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: "TODO: redo".into(),
                            time: 3.0,
                        });
                    }
                }

                ui.separator();

                if ui.button("     Reset windows").clicked() {
                    ui.ctx().memory_mut(|mem| mem.reset_areas());
                    ui.close_menu();
                }

                // TODO: per surface viewer instance
                // {
                //     ui.horizontal_wrapped(|ui| {
                //         ui.label("     Visual grid");
                //         ui.add_space(5.0);
                //         ui.add(ToggleSwitch::new(visual_grid_visible));
                //     });
                // }

                ui.separator();

                if ui.button("\u{2699} Preferences").clicked() {
                    self.event_loop.send_event(VgonioEvent::Notify {
                        kind: NotifyKind::Info,
                        text: "TODO: open preferences window".into(),
                        time: 3.0,
                    });
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui.button("\u{1F4D8} Console").clicked() {
                    println!("TODO: open console window");
                }
                if ui.button("Scratch").clicked() {
                    self.tools.toggle::<Scratch>();
                }
                if ui.button("\u{1F41B} Debugging").clicked() {
                    self.tools.toggle::<DebuggingInspector>();
                }
                if ui.button("\u{1F3B2} Sampling").clicked() {
                    self.tools.toggle::<SamplingInspector>();
                }
            });
            ui.menu_button("Theme", |ui| {
                if ui.button("☀ Light").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::UpdateThemeKind(ThemeKind::Light));
                }
                if ui.button("🌙 Dark").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::UpdateThemeKind(ThemeKind::Dark));
                }
            });

            ui.add_space(6.0);
            ui.hyperlink_to("Help", "https://github.com/matthiascy/vgonio");
            ui.add_space(6.0);
            #[cfg(not(target_arch = "wasm32"))]
            {
                if ui.button("Quit").clicked() {
                    self.event_loop.send_event(VgonioEvent::Quit);
                }
            }
        });
    }

    fn open_files(
        &mut self,
        files: &[rfd::FileHandle],
    ) -> (Vec<Handle<MicroSurface>>, Vec<Handle<Measurement>>) {
        let mut surfaces = vec![];
        let mut measurements = vec![];
        // TODO: handle other file types
        for file in files {
            let path: PathBuf = file.into();
            let ext = match path.extension() {
                None => None,
                Some(s) => s.to_str().map(|s| s.to_lowercase()),
            };

            if let Some(ext) = ext {
                match ext.as_str() {
                    "vgmo" => {
                        // Micro-surface measurement data
                        log::debug!("Opening micro-surface measurement output: {:?}", path);
                        self.cache.write(|cache| {
                            match cache.load_micro_surface_measurement(&self.config, &path) {
                                Ok(hdl) => {
                                    measurements.push(hdl);
                                }
                                Err(e) => {
                                    log::error!(
                                        "Failed to load micro surface measurement: {:?}",
                                        e
                                    );
                                }
                            }
                        })
                    }
                    "vgms" | "txt" | "os3d" => {
                        // Micro-surface profile
                        log::debug!("Opening micro-surface profile: {:?}", path);
                        self.cache.write(|cache| {
                            match cache.load_micro_surface(&self.config, &path) {
                                Ok((surf, _)) => {
                                    let _ = cache
                                        .create_micro_surface_renderable_mesh(&self.gpu_ctx, surf)
                                        .unwrap();
                                    surfaces.push(surf)
                                }
                                Err(e) => {
                                    log::error!("Failed to load micro surface: {:?}", e);
                                }
                            }
                        })
                    }
                    "spd" => {
                        todo!()
                    }
                    "csv" => {
                        todo!()
                    }
                    _ => {}
                }
            } else {
                log::warn!("File {:?} has no extension, ignoring", path);
            }
        }
        (surfaces, measurements)
    }
}

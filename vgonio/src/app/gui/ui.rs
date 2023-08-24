use crate::{
    app::{
        cache::{Cache, Handle},
        gfx::GpuContext,
        gui::{
            data::PropertyData,
            event::{EventLoopProxy, OutlinerEvent, SurfaceViewerEvent, VgonioEvent},
            file_drop::FileDragDrop,
            gizmo::NavigationGizmo,
            icons,
            notify::{NotifyKind, NotifySystem},
            outliner::OutlinerItem,
            plotter::{PlotInspector, PlottingWidget},
            simulations::Simulations,
            state::GuiRenderer,
            theme::ThemeKind,
            tools::{SamplingInspector, Scratch, Tools},
            DebuggingInspector,
        },
        Config,
    },
    fitting::{
        AreaDistributionFittingProblem, BeckmannSpizzichinoAnisotropicNDF, BeckmannSpizzichinoMSF,
        BeckmannSpizzichinoNDF, FittedModel, FittingProblem, Isotropy, MicrofacetModelFamily,
        MmsfFittingProblem, ReflectionModelFamily, TrowbridgeReitzAnisotropicNDF,
        TrowbridgeReitzMSF, TrowbridgeReitzNDF,
    },
    measure::measurement::{MeasurementData, MeasurementKind},
};
use egui::NumExt;
use egui_gizmo::GizmoOrientation;
use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};
use vgcore::math::{Mat4, Vec3};
use vgsurf::MicroSurface;

use super::{docking::DockSpace, event::EventResponse};

/// Implementation of the GUI for vgonio application.
pub struct VgonioGui {
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy,

    gpu_ctx: Arc<GpuContext>,

    cache: Arc<RwLock<Cache>>,

    // pub simulation_workspace: SimulationWorkspace, // TODO: make private, simplify access
    /// The drag and drop state.
    drag_drop: FileDragDrop,

    /// Gizmo inside the viewport for navigating the scene.
    navigator: NavigationGizmo,

    /// Notification system.
    notif: NotifySystem,

    /// Plotters are windows that contain plots.
    plotters: Vec<(bool, Box<dyn PlottingWidget>)>,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    pub simulations: Simulations,
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
        cache: Arc<RwLock<Cache>>,
    ) -> Self {
        log::info!("Initializing UI ...");
        let properties = Arc::new(RwLock::new(PropertyData::new()));

        Self {
            config,
            event_loop: event_loop.clone(),
            tools: Tools::new(event_loop.clone(), gpu.clone(), gui.clone(), cache.clone()),
            cache: cache.clone(),
            drag_drop: FileDragDrop::new(event_loop.clone()),
            navigator: NavigationGizmo::new(GizmoOrientation::Global),
            simulations: Simulations::new(event_loop.clone()),
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
                self.tools
                    .get_tool_mut::<DebuggingInspector>()
                    .unwrap()
                    .update_surface_viewers(&[*uuid]);
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
                    let mut cache = self.cache.write().unwrap();
                    match item {
                        OutlinerItem::MicroSurface(handle) => {
                            properties.surfaces.remove(handle);
                            cache.unload_micro_surface(*handle).unwrap();
                        }
                        _ => {
                            todo!("Remove measured data from cache")
                        }
                    }
                    EventResponse::Handled
                }
            },
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
                        MeasurementKind::Mndf => Box::new(PlotInspector::new_area_distrib_plot(
                            prop.measured.get(data).unwrap().name.clone(),
                            *data,
                            self.cache.clone(),
                            self.properties.clone(),
                            self.event_loop.clone(),
                        )),
                        MeasurementKind::Mmsf => Box::new(PlotInspector::new_mmsf(
                            prop.measured.get(data).unwrap().name.clone(),
                            *data,
                            self.cache.clone(),
                            self.properties.clone(),
                            self.event_loop.clone(),
                        )),
                    };
                    if *independent {
                        self.plotters.push((true, plotter));
                    } else {
                        self.dock_space.add_existing_widget(plotter)
                    }
                }
                EventResponse::Handled
            }
            #[cfg(feature = "scaled-ndf-fitting")]
            VgonioEvent::Fitting {
                kind,
                family,
                data,
                isotropy,
                scaled,
            } => {
                match kind {
                    MeasurementKind::Bsdf => {}
                    MeasurementKind::Mndf => {
                        let mut prop = self.properties.write().unwrap();
                        let fitted = &mut prop.measured.get_mut(data).unwrap().fitted;
                        if !fitted.contains(*family, *isotropy, *scaled) {
                            let cache = self.cache.read().unwrap();
                            let measurement = cache.get_measurement_data(*data).unwrap();
                            let data = measurement
                                .measured
                                .madf_data()
                                .expect("Measurement has no MNDF data.");
                            let problem = AreaDistributionFittingProblem::new(
                                data, *family, *isotropy, *scaled,
                            );
                            let report = problem.lsq_lm_fit();
                            report.print_fitting_report();

                            fitted.push(FittedModel::Mndf(report.best_model().clone_box()));
                        }
                    }
                    MeasurementKind::Mmsf => {
                        let mut prop = self.properties.write().unwrap();
                        let fitted = &mut prop.measured.get_mut(data).unwrap().fitted;
                        if !fitted.contains(*family, *isotropy, *scaled) {
                            let cache = self.cache.read().unwrap();
                            let measurement = cache.get_measurement_data(*data).unwrap();
                            let data = measurement
                                .measured
                                .mmsf_data()
                                .expect("Measurement has no MMSF data.");
                            let problem = match family {
                                ReflectionModelFamily::Microfacet(m) => match m {
                                    MicrofacetModelFamily::TrowbridgeReitz => {
                                        MmsfFittingProblem::new(
                                            data,
                                            TrowbridgeReitzMSF { width: 0.05 },
                                            Vec3::Y,
                                        )
                                    }
                                    MicrofacetModelFamily::BeckmannSpizzichino => {
                                        MmsfFittingProblem::new(
                                            data,
                                            BeckmannSpizzichinoMSF { alpha: 0.05 },
                                            Vec3::Y,
                                        )
                                    }
                                },
                            };
                            let report = problem.lsq_lm_fit();
                            report.print_fitting_report();
                            fitted.push(FittedModel::Mmsf(report.best_model().clone_box()));
                        }
                    }
                }
                EventResponse::Handled
            }
            #[cfg(not(feature = "scaled-ndf-fitting"))]
            VgonioEvent::Fitting {
                kind,
                family,
                data,
                isotropy: isotropic,
            } => {
                match kind {
                    MeasurementKind::Bsdf => {}
                    MeasurementKind::Mndf => {
                        let mut prop = self.properties.write().unwrap();
                        let fitted = &mut prop.measured.get_mut(data).unwrap().fitted;
                        if !fitted.contains(*family, *isotropic) {
                            let cache = self.cache.read().unwrap();
                            let measurement = cache.get_measurement_data(*data).unwrap();
                            let data = measurement
                                .measured
                                .madf_data()
                                .expect("Measurement has no MADF data.");
                            let problem = match family {
                                ReflectionModelFamily::Microfacet(m) => match m {
                                    MicrofacetModelFamily::TrowbridgeReitz => {
                                        if *isotropic {
                                            AreaDistributionFittingProblem::new(
                                                data,
                                                TrowbridgeReitzNDF::default(),
                                            )
                                        } else {
                                            AreaDistributionFittingProblem::new(
                                                data,
                                                TrowbridgeReitzAnisotropicNDF::default(),
                                            )
                                        }
                                    }
                                    MicrofacetModelFamily::BeckmannSpizzichino => {
                                        if *isotropic {
                                            AreaDistributionFittingProblem::new(
                                                data,
                                                BeckmannSpizzichinoNDF::default(),
                                            )
                                        } else {
                                            AreaDistributionFittingProblem::new(
                                                data,
                                                BeckmannSpizzichinoAnisotropicNDF::default(),
                                            )
                                        }
                                    }
                                },
                            };
                            let (result, report) = problem.lsq_lm_fit();
                            log::info!("Report: {:?}", report);
                            log::info!("Result: {:?}", result);
                            fitted.push(FittedModel::Mndf(result));
                        }
                    }
                    MeasurementKind::Mmsf => {
                        let mut prop = self.properties.write().unwrap();
                        let fitted = &mut prop.measured.get_mut(data).unwrap().fitted;
                        if fitted.contains(*family, *isotropic) {
                            let cache = self.cache.read().unwrap();
                            let measurement = cache.get_measurement_data(*data).unwrap();
                            let data = measurement
                                .measured
                                .mmsf_data()
                                .expect("Measurement has no MMSF data.");
                            let problem = match family {
                                ReflectionModelFamily::Microfacet(m) => match m {
                                    MicrofacetModelFamily::TrowbridgeReitz => {
                                        MmsfFittingProblem::new(
                                            data,
                                            TrowbridgeReitzMSF { width: 0.05 },
                                            Vec3::Y,
                                        )
                                    }
                                    MicrofacetModelFamily::BeckmannSpizzichino => {
                                        MmsfFittingProblem::new(
                                            data,
                                            BeckmannSpizzichinoMSF { alpha: 0.05 },
                                            Vec3::Y,
                                        )
                                    }
                                },
                            };
                            let (result, report) = problem.lsq_lm_fit();
                            log::info!("Report: {:?}", report);
                            log::info!("Result: {:?}", result);
                            fitted.push(FittedModel::Mmsf(result));
                        }
                    }
                }
                EventResponse::Handled
            }
            _ => EventResponse::Ignored(event),
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.navigator.update_matrices(model, view, proj);
    }

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
        self.navigator.show(ctx);
        self.simulations.show_all(ctx);
        self.notif.show(ctx);
        for (is_open, plotter) in &mut self.plotters {
            plotter.show(ctx, is_open);
        }
    }

    pub fn on_open_files(&mut self, files: &[rfd::FileHandle]) {
        log::info!("Process UI opening files: {:?}", files);
        let (surfaces, measurements) = self.open_files(files);
        let cache = self.cache.read().unwrap();
        let mut properties = self.properties.write().unwrap();
        properties.update_surfaces(&surfaces, &cache);
        properties.update_measurement_data(&measurements, &cache);
        self.tools
            .get_tool_mut::<DebuggingInspector>()
            .unwrap()
            .update_surfaces(&surfaces);
        self.simulations.update_loaded_surfaces(&surfaces, &cache);
        self.event_loop
            .send_event(VgonioEvent::SurfaceViewer(
                SurfaceViewerEvent::UpdateSurfaceList { surfaces },
            ))
            .unwrap();
    }

    fn main_menu(&mut self, ui: &mut egui::Ui, kind: ThemeKind) {
        ui.set_height(28.0);
        let icon_image = match kind {
            ThemeKind::Dark => icons::get_icon_image("vgonio_menu_dark").unwrap(),
            ThemeKind::Light => icons::get_icon_image("vgonio_menu_light").unwrap(),
        };
        let desired_icon_height = (ui.max_rect().height() - 4.0).at_most(28.0);
        let image_size = icon_image.size_vec2() * (desired_icon_height / icon_image.size_vec2().y);
        let texture_id = icon_image.texture_id(ui.ctx());

        ui.menu_image_button(texture_id, image_size, |ui| {
            if ui.button("About").clicked() {
                self.event_loop
                    .send_event(VgonioEvent::Notify {
                        kind: NotifyKind::Info,
                        text: "TODO: about".to_string(),
                        time: 0.0,
                    })
                    .unwrap();
            }

            ui.menu_button("New", |ui| {
                ui.menu_button("Measurement", |ui| {
                    if ui.button("BSDF").clicked() {
                        self.simulations.open_bsdf_sim();
                    }
                    if ui.button("NDF").clicked() {
                        self.simulations.open_madf_sim();
                    }
                    if ui.button("Masking/Shadowing").clicked() {
                        self.simulations.open_mmsf_sim();
                    }
                });
                if ui.button("Micro-surface").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: "TODO: new height field".to_string(),
                            time: 3.0,
                        })
                        .unwrap();
                }
            });
            if ui.button("Open...").clicked() {
                use rfd::AsyncFileDialog;
                let dir = self
                    .config
                    .user_data_dir()
                    .unwrap_or_else(|| self.config.sys_data_dir());
                let task = AsyncFileDialog::new().set_directory(dir).pick_files();
                let event_loop_proxy = self.event_loop.clone();
                std::thread::spawn(move || {
                    pollster::block_on(async {
                        let file_handles = task.await;
                        if let Some(hds) = file_handles {
                            if event_loop_proxy
                                .send_event(VgonioEvent::OpenFiles(hds))
                                .is_err()
                            {
                                log::warn!("[EVENT] Failed to send OpenFiles event");
                            }
                        }
                    })
                });
            }
            ui.menu_button("Recent...", |ui| {
                for i in 0..10 {
                    if ui.button(format!("item {i}")).clicked() {
                        self.event_loop
                            .send_event(VgonioEvent::Notify {
                                kind: NotifyKind::Info,
                                text: format!("TODO: open recent item {i}"),
                                time: 3.0,
                            })
                            .unwrap();
                    }
                }
            });

            ui.add_space(6.0);

            {
                if ui.button("Save...").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: "TODO: save".into(),
                            time: 3.0,
                        })
                        .unwrap();
                }
            }

            ui.menu_button("Edit", |ui| {
                {
                    if ui.button("     Undo").clicked() {
                        self.event_loop
                            .send_event(VgonioEvent::Notify {
                                kind: NotifyKind::Info,
                                text: "TODO: undo".into(),
                                time: 3.0,
                            })
                            .unwrap();
                    }
                    if ui.button("     Redo").clicked() {
                        self.event_loop
                            .send_event(VgonioEvent::Notify {
                                kind: NotifyKind::Info,
                                text: "TODO: redo".into(),
                                time: 3.0,
                            })
                            .unwrap();
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
                    self.event_loop
                        .send_event(VgonioEvent::Notify {
                            kind: NotifyKind::Info,
                            text: "TODO: open preferences window".into(),
                            time: 3.0,
                        })
                        .unwrap();
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
                        .send_event(VgonioEvent::UpdateThemeKind(ThemeKind::Light))
                        .unwrap();
                }
                if ui.button("🌙 Dark").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::UpdateThemeKind(ThemeKind::Dark))
                        .unwrap();
                }
            });

            ui.add_space(6.0);
            ui.hyperlink_to("Help", "https://github.com/matthiascy/vgonio");
            ui.add_space(6.0);
            #[cfg(not(target_arch = "wasm32"))]
            {
                if ui.button("Quit").clicked()
                    && self.event_loop.send_event(VgonioEvent::Quit).is_err()
                {
                    log::warn!("[EVENT] Failed to send Quit event.");
                }
            }
        });
    }

    fn open_files(
        &mut self,
        files: &[rfd::FileHandle],
    ) -> (Vec<Handle<MicroSurface>>, Vec<Handle<MeasurementData>>) {
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
                        match self
                            .cache
                            .write()
                            .unwrap()
                            .load_micro_surface_measurement(&self.config, &path)
                        {
                            Ok(hdl) => {
                                measurements.push(hdl);
                            }
                            Err(e) => {
                                log::error!("Failed to load micro surface measurement: {:?}", e);
                            }
                        }
                    }
                    "vgms" | "txt" | "os3d" => {
                        // Micro-surface profile
                        log::debug!("Opening micro-surface profile: {:?}", path);
                        let mut locked_cache = self.cache.write().unwrap();
                        match locked_cache.load_micro_surface(&self.config, &path) {
                            Ok((surf, _)) => {
                                let _ = locked_cache
                                    .create_micro_surface_renderable_mesh(
                                        &self.gpu_ctx.device,
                                        surf,
                                    )
                                    .unwrap();
                                surfaces.push(surf)
                            }
                            Err(e) => {
                                log::error!("Failed to load micro surface: {:?}", e);
                            }
                        }
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

fn icon_toggle_button(
    ui: &mut egui::Ui,
    icon_name: &'static str,
    selected: &mut bool,
) -> egui::Response {
    let size_points = egui::Vec2::splat(16.0);
    let image = icons::get_icon_image(icon_name).unwrap();
    // let image = self.icon_image(icon);
    let tex_id = image.texture_id(ui.ctx());
    let tint = if *selected {
        ui.visuals().widgets.inactive.fg_stroke.color
    } else {
        egui::Color32::from_gray(100)
    };
    let mut response = ui.add(egui::ImageButton::new(tex_id, size_points).tint(tint));
    if response.clicked() {
        *selected = !*selected;
        response.mark_changed()
    }
    response
}

use super::{
    docking::{Dockable, DockableString, Tab, TabKind, TabViewer},
    state::GuiRenderer,
    VgonioEvent,
};
use crate::app::{
    cache::{Cache, Handle},
    gfx::GpuContext,
    gui::{
        bsdf_viewer::BsdfViewer,
        file_drag_drop::FileDragDrop,
        gizmo::NavigationGizmo,
        icons::Icon,
        outliner::Outliner,
        simulations::Simulations,
        state::InputState,
        surf_viewer::SurfViewer,
        theme::{Theme, ThemeState, ThemeVisuals},
        tools::{SamplingInspector, Scratch, Tools},
        widgets::ToggleSwitch,
        DebuggingInspector, DockingTabsTree, VgonioEventLoop,
    },
    Config,
};
use egui::{Id, NumExt, Ui, Widget, WidgetText};
use egui_dock::{Node, NodeIndex, TabIndex};
use egui_extras::RetainedImage;
use egui_gizmo::GizmoOrientation;
use egui_toast::ToastKind;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::{Debug, Formatter},
    ops::{Deref, Index},
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};
use vgcore::{error::VgonioError, math::Mat4};
use vgsurf::MicroSurface;

#[derive(Default)]
pub struct ImageCache {
    images: HashMap<&'static str, Arc<RetainedImage>>,
}

impl ImageCache {
    pub fn get_or_insert(
        &mut self,
        id: &'static str,
        bytes: Option<&'static [u8]>,
    ) -> Option<Arc<RetainedImage>> {
        match bytes {
            None => self.images.get(id).cloned(),
            Some(b) => Some(
                self.images
                    .entry(id)
                    .or_insert_with(|| {
                        let image = load_image_from_bytes(b)
                            .unwrap_or_else(|err| panic!("Failed to load image {id:?}: {}", err));
                        let retained = RetainedImage::from_color_image(id, image);
                        Arc::new(retained)
                    })
                    .clone(),
            ),
        }
    }
}

fn load_image_from_bytes(bytes: &[u8]) -> Result<egui::ColorImage, VgonioError> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| {
            VgonioError::new(format!("Failed to load image \"{}\"from bytes", err), None)
        })?
        .into_rgba8();
    let size = [image.width() as _, image.height() as _];
    let pixels = image.as_flat_samples();
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        size,
        pixels.as_slice(),
    ))
}

/// Implementation of the GUI for vgonio application.
pub struct VgonioUi {
    gpu: Arc<GpuContext>,
    gui: Arc<RwLock<GuiRenderer>>,

    cache: Arc<RwLock<Cache>>,

    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Event loop proxy for sending user defined events.
    event_loop: VgonioEventLoop,

    theme: ThemeState,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    // pub simulation_workspace: SimulationWorkspace, // TODO: make private, simplify access
    /// The drag and drop state.
    drag_drop: FileDragDrop,

    /// Gizmo inside the viewport for navigating the scene.
    navigator: NavigationGizmo,

    /// Outliner of the scene.
    outliner: Arc<RwLock<Outliner>>,

    image_cache: Arc<Mutex<ImageCache>>,

    /// Visibility of the visual grid.
    pub visual_grid_enabled: bool,

    pub right_panel_expanded: bool,
    pub left_panel_expanded: bool,

    pub simulations: Simulations,

    pub id_counter: u32,
}

impl VgonioUi {
    pub fn new(
        event_loop: VgonioEventLoop,
        config: Arc<Config>,
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        outliner: Arc<RwLock<Outliner>>,
        cache: Arc<RwLock<Cache>>,
    ) -> Self {
        log::info!("Initializing UI");
        Self {
            gpu: gpu.clone(),
            gui: gui.clone(),
            cache: cache.clone(),
            config,
            event_loop: event_loop.clone(),
            tools: Tools::new(
                event_loop.clone(),
                gpu.clone(),
                &mut gui.write().unwrap(),
                cache,
            ),
            // simulation_workspace: SimulationWorkspace::new(event_loop.clone(), cache.clone()),
            drag_drop: FileDragDrop::new(event_loop.clone()),
            theme: ThemeState::default(),
            navigator: NavigationGizmo::new(GizmoOrientation::Global),
            outliner,
            image_cache: Arc::new(Mutex::new(Default::default())),
            visual_grid_enabled: true,
            right_panel_expanded: true,
            left_panel_expanded: false,
            simulations: Simulations::new(event_loop),
            id_counter: 0,
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.navigator.update_matrices(model, view, proj);
    }

    pub fn show(&mut self, ctx: &egui::Context, tabs: &mut DockingTabsTree<Tab>) {
        self.theme.update(ctx);
        egui::TopBottomPanel::top("vgonio_top_panel")
            .exact_height(28.0)
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    self.vgonio_menu(ui);
                });
            });

        if self.right_panel_expanded {
            egui::SidePanel::right("vgonio_right_panel")
                .resizable(true)
                .min_width(300.0)
                .default_width(460.0)
                .width_range(200.0..=500.0)
                .show(ctx, |ui| self.outliner.write().unwrap().ui(ui));
        }

        // New docking system
        let mut added_nodes = Vec::new();
        egui_dock::DockArea::new(tabs)
            .show_add_buttons(true)
            .show_add_popup(true)
            .show(
                ctx,
                &mut TabViewer {
                    to_be_added: &mut added_nodes,
                },
            );

        added_nodes.drain(..).for_each(|to_be_added| {
            // Focus the node that we want to add the tab to
            tabs.set_focused_node(to_be_added.parent);
            // Allocate a new index for the tab
            let index = NodeIndex(tabs.num_tabs());
            // Add the tab
            let dockable: Box<dyn Dockable> = match to_be_added.kind {
                TabKind::SurfViewer => Box::new(SurfViewer::new(
                    self.gpu.clone(),
                    self.gui.clone(),
                    512,
                    512,
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                    self.cache.clone(),
                    self.outliner.clone(),
                    self.event_loop.clone(),
                )),
                // TabKind::Outliner => Box::new(Outliner::new(
                //     self.outliner.clone(),
                //     self.event_loop.clone(),
                //     self.cache.clone(),
                // )),
                _ => Box::new(DockableString::new(String::from("Hello world"))),
            };
            tabs.push_to_focused_leaf(Tab {
                kind: to_be_added.kind,
                index,
                dockable,
            });
        });

        self.tools.show(ctx);
        self.drag_drop.show(ctx);
        // self.navigator.show(ctx);
        // self.simulations.show_all(ctx);
    }

    pub fn set_theme(&mut self, theme: Theme) { self.theme.set(theme); }

    pub fn current_theme_visuals(&self) -> &ThemeVisuals { self.theme.current_theme_visuals() }

    pub fn update_loaded_surfaces(&mut self, surfaces: &[Handle<MicroSurface>], cache: &Cache) {
        self.tools
            .get_tool_mut::<DebuggingInspector>()
            .unwrap()
            .update_surfaces(surfaces);
        self.simulations.update_loaded_surfaces(surfaces, cache);
    }
}

impl VgonioUi {
    fn icon_image(&self, icon: &Icon) -> Arc<RetainedImage> {
        self.image_cache
            .lock()
            .unwrap()
            .get_or_insert(icon.id, Some(icon.bytes))
            .unwrap()
    }

    fn icon_toggle_button(
        &self,
        ui: &mut egui::Ui,
        icon: &Icon,
        selected: &mut bool,
    ) -> egui::Response {
        let size_points = egui::Vec2::splat(16.0);
        let image = self.icon_image(icon);
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

    fn vgonio_menu(&mut self, ui: &mut egui::Ui) {
        ui.set_height(28.0);
        let icon_image = if self.theme.current_theme() == Theme::Dark {
            self.icon_image(&Icon::VGONIO_MENU_DARK)
        } else {
            self.icon_image(&Icon::VGONIO_MENU_LIGHT)
        };
        let desired_icon_height = (ui.max_rect().height() - 4.0).at_most(24.0);
        let image_size = icon_image.size_vec2() * (desired_icon_height / icon_image.size_vec2().y);
        let texture_id = icon_image.texture_id(ui.ctx());

        ui.menu_image_button(texture_id, image_size, |ui| {
            if ui.button("About").clicked() {
                self.event_loop
                    .send_event(VgonioEvent::Notify {
                        kind: ToastKind::Info,
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
                            kind: ToastKind::Info,
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
                                kind: ToastKind::Info,
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
                            kind: ToastKind::Info,
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
                                kind: ToastKind::Info,
                                text: "TODO: undo".into(),
                                time: 3.0,
                            })
                            .unwrap();
                    }
                    if ui.button("     Redo").clicked() {
                        self.event_loop
                            .send_event(VgonioEvent::Notify {
                                kind: ToastKind::Info,
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

                {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("     Visual grid");
                        ui.add_space(5.0);
                        ui.add(ToggleSwitch::new(&mut self.visual_grid_enabled));
                    });
                }

                ui.separator();

                if ui.button("\u{2699} Preferences").clicked() {
                    self.event_loop
                        .send_event(VgonioEvent::Notify {
                            kind: ToastKind::Info,
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
                    self.theme.set(Theme::Light);
                }
                if ui.button("🌙 Dark").clicked() {
                    self.theme.set(Theme::Dark);
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

        ui.add_space(ui.available_width() - 48.0);

        let mut left_panel_expanded = self.left_panel_expanded;
        if self
            .icon_toggle_button(ui, &Icon::LEFT_PANEL_TOGGLE, &mut left_panel_expanded)
            .clicked()
        {
            self.left_panel_expanded = left_panel_expanded;
        }

        let mut right_panel_expanded = self.right_panel_expanded;
        if self
            .icon_toggle_button(ui, &Icon::RIGHT_PANEL_TOGGLE, &mut right_panel_expanded)
            .clicked()
        {
            self.right_panel_expanded = right_panel_expanded;
        }
    }
}

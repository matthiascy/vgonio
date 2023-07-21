use crate::app::{
    cache::{Cache, Handle},
    gfx::GpuContext,
    gui::{
        bsdf_viewer::BsdfViewer,
        event::{EventLoopProxy, VgonioEvent},
        file_drop::FileDragDrop,
        gizmo::NavigationGizmo,
        icons,
        outliner::Outliner,
        simulations::Simulations,
        state::GuiRenderer,
        theme::ThemeKind,
        tools::{SamplingInspector, Scratch, Tools},
        widgets::ToggleSwitch,
        DebuggingInspector,
    },
    Config,
};
use egui::{NumExt, Ui, WidgetText};
use egui_gizmo::GizmoOrientation;
use egui_toast::ToastKind;
use std::sync::{Arc, RwLock};
use vgcore::math::Mat4;
use vgsurf::MicroSurface;

use super::{docking::DockSpace, event::EventResponse, tools::PlottingWidget};

/// Implementation of the GUI for vgonio application.
pub struct VgonioGui {
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    cache: Arc<RwLock<Cache>>,

    // pub simulation_workspace: SimulationWorkspace, // TODO: make private, simplify access
    /// The drag and drop state.
    drag_drop: FileDragDrop,

    /// Gizmo inside the viewport for navigating the scene.
    navigator: NavigationGizmo,

    /// Outliner of the scene.
    outliner: Outliner,

    plotting_inspectors: Vec<Box<dyn PlottingWidget>>,

    pub right_panel_expanded: bool,
    pub left_panel_expanded: bool,
    pub simulations: Simulations,

    /// Docking system for the UI.
    dockspace: DockSpace,
}

struct TabViewer;

impl egui_dock::TabViewer for TabViewer {
    type Tab = String;

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) { ui.label(format!("Content of {tab}")); }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText { (&*tab).into() }
}

impl VgonioGui {
    pub fn new(
        event_loop: EventLoopProxy,
        config: Arc<Config>,
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        bsdf_viewer: Arc<RwLock<BsdfViewer>>,
        cache: Arc<RwLock<Cache>>,
    ) -> Self {
        log::info!("Initializing UI");
        Self {
            config,
            event_loop: event_loop.clone(),
            tools: Tools::new(
                event_loop.clone(),
                gpu.clone(),
                &mut gui.write().unwrap(),
                cache.clone(),
            ),
            // simulation_workspace: SimulationWorkspace::new(event_loop.clone(), cache.clone()),
            cache,
            drag_drop: FileDragDrop::new(event_loop.clone()),
            navigator: NavigationGizmo::new(GizmoOrientation::Global),
            outliner: Outliner::new(gpu, bsdf_viewer, event_loop.clone()),
            right_panel_expanded: true,
            left_panel_expanded: false,
            simulations: Simulations::new(event_loop),
            plotting_inspectors: vec![],
            dockspace: DockSpace::default(),
        }
    }

    /// Handles a user event.
    ///
    /// Returns [`EventResponse::Ignored`] if the event was not handled,
    /// otherwise returns [`EventResponse::Handled`].
    pub fn on_user_event(&mut self, event: VgonioEvent) -> EventResponse {
        match event {
            _ => EventResponse::Ignored(event),
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.navigator.update_matrices(model, view, proj);
    }

    pub fn show(&mut self, ctx: &egui::Context, kind: ThemeKind, visual_grid_visble: &mut bool) {
        egui::TopBottomPanel::top("vgonio_top_panel")
            .exact_height(28.0)
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    self.main_menu(ui, kind, visual_grid_visble);
                });
            });
        if self.right_panel_expanded {
            egui::SidePanel::right("vgonio_right_panel")
                .min_width(300.0)
                .default_width(460.0)
                .resizable(true)
                .show(ctx, |ui| self.outliner.ui(ui, self.cache.clone()));
        }

        self.dockspace.show(ctx);

        self.tools.show(ctx);
        self.drag_drop.show(ctx);
        self.navigator.show(ctx);
        self.simulations.show_all(ctx);
    }

    pub fn outliner(&self) -> &Outliner { &self.outliner }

    pub fn outliner_mut(&mut self) -> &mut Outliner { &mut self.outliner }

    pub fn update_loaded_surfaces(&mut self, surfaces: &[Handle<MicroSurface>], cache: &Cache) {
        self.outliner_mut().update_surfaces(surfaces, cache);
        self.tools
            .get_tool_mut::<DebuggingInspector>()
            .unwrap()
            .update_surfaces(surfaces);
        self.simulations.update_loaded_surfaces(surfaces, cache);
    }

    fn main_menu(&mut self, ui: &mut egui::Ui, kind: ThemeKind, visual_grid_visible: &mut bool) {
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
                        ui.add(ToggleSwitch::new(visual_grid_visible));
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

        ui.add_space(ui.available_width() - 48.0);

        let mut left_panel_expanded = self.left_panel_expanded;
        if icon_toggle_button(ui, "left_panel_toggle", &mut left_panel_expanded).clicked() {
            self.left_panel_expanded = left_panel_expanded;
        }

        let mut right_panel_expanded = self.right_panel_expanded;
        if icon_toggle_button(ui, "right_panel_toggle", &mut right_panel_expanded).clicked() {
            self.right_panel_expanded = right_panel_expanded;
        }
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

use super::{state::GuiRenderer, VgonioEvent};
use crate::{
    app::{
        gfx::GpuContext,
        gui::{
            gizmo::NavigationGizmo,
            icons::Icon,
            outliner::Outliner,
            tools::{PlottingInspector, SamplingInspector, Scratch, Tools},
            DebuggingInspector,
        },
        Config,
    },
    error::Error,
};
use egui::NumExt;
use egui_extras::RetainedImage;
use egui_gizmo::GizmoOrientation;
use glam::Mat4;
use std::{
    collections::HashMap,
    fmt::Write,
    ops::Deref,
    sync::{Arc, Mutex},
};
use winit::event_loop::EventLoopProxy;

/// Implementation of the drag and drop functionality.
pub struct FileDragDrop {
    event_loop: EventLoopProxy<VgonioEvent>,
    files: Vec<egui::DroppedFile>,
}

impl FileDragDrop {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>) -> Self {
        Self {
            event_loop,
            files: vec![],
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        use egui::*;

        let hovered_files = ctx.input(|i| i.raw.hovered_files.clone());

        // Preview hovering files:
        if !hovered_files.is_empty() {
            let mut text = "Dropping files:\n".to_owned();
            for file in &hovered_files {
                if let Some(path) = &file.path {
                    write!(text, "\n{}", path.display()).unwrap();
                } else if !file.mime.is_empty() {
                    write!(text, " \n{}", file.mime).unwrap();
                } else {
                    text += "\n???";
                }
            }

            let painter =
                ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("file_drop_target")));
            let screen_rect = ctx.input(|i| i.screen_rect());
            painter.rect_filled(screen_rect, 0.0, Color32::from_black_alpha(210));
            painter.text(
                screen_rect.center(),
                Align2::CENTER_CENTER,
                text,
                TextStyle::Heading.resolve(&ctx.style()),
                Color32::WHITE,
            );
        }

        // Collect dropped files:
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        if !dropped_files.is_empty() {
            self.files = dropped_files;
        }

        if !self.files.is_empty() {
            let files = std::mem::take(&mut self.files)
                .into_iter()
                .filter_map(|f| {
                    f.path
                        .filter(|p| p.is_file() && p.exists())
                        .map(rfd::FileHandle::from)
                })
                .collect::<Vec<_>>();

            if self
                .event_loop
                .clone()
                .send_event(VgonioEvent::OpenFiles(files))
                .is_err()
            {
                log::warn!("[EVENT] Failed to send OpenFiles event");
            }
        }
    }
}

pub struct ThemeState {
    pub theme: Theme,
    pub theme_visuals: [ThemeVisuals; 2],
    pub need_update: bool,
}

impl Default for ThemeState {
    fn default() -> Self {
        Self {
            theme_visuals: [
                ThemeVisuals {
                    egui_visuals: egui::Visuals {
                        dark_mode: true,
                        ..egui::Visuals::dark()
                    },
                    clear_color: wgpu::Color {
                        r: 0.046, // no gamma correction
                        g: 0.046,
                        b: 0.046,
                        a: 1.0,
                    },
                    grid_line_color: wgpu::Color {
                        r: 0.4,
                        g: 0.4,
                        b: 0.4,
                        a: 1.0,
                    },
                },
                ThemeVisuals {
                    egui_visuals: egui::Visuals {
                        dark_mode: false,
                        panel_fill: egui::Color32::from_gray(190),
                        ..egui::Visuals::light()
                    },
                    clear_color: wgpu::Color {
                        r: 0.208, // no gamma correction
                        g: 0.208,
                        b: 0.208,
                        a: 1.0,
                    },
                    grid_line_color: wgpu::Color {
                        r: 0.68,
                        g: 0.68,
                        b: 0.68,
                        a: 1.0,
                    },
                },
            ],
            theme: Theme::Light,
            need_update: true,
        }
    }
}

impl ThemeState {
    pub fn update(&mut self, ctx: &egui::Context) {
        if self.need_update {
            self.need_update = false;
            ctx.set_visuals(self.theme_visuals[self.theme as usize].egui_visuals.clone());
        }
    }

    pub fn set(&mut self, theme: Theme) {
        if self.theme != theme {
            self.theme = theme;
            self.need_update = true;
        }
    }

    pub fn current_theme(&self) -> Theme { self.theme }

    pub fn current_theme_visuals(&self) -> &ThemeVisuals {
        &self.theme_visuals[self.theme as usize]
    }
}

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

fn load_image_from_bytes(bytes: &[u8]) -> Result<egui::ColorImage, Error> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| Error::Any(e.to_string()))?
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
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy<VgonioEvent>,

    theme: ThemeState,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    // pub simulation_workspace: SimulationWorkspace, // TODO: make private, simplify access
    /// The drag and drop state.
    drag_drop: FileDragDrop,

    /// Gizmo inside the viewport for navigating the scene.
    navigator: NavigationGizmo,

    /// Outliner of the scene.
    outliner: Outliner,

    image_cache: Arc<Mutex<ImageCache>>,

    /// Visibility of the visual grid.
    pub enable_visual_grid: bool,

    pub right_panel_expanded: bool,
    pub left_panel_expanded: bool,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Theme {
    Dark = 0,
    Light = 1,
}

pub struct ThemeVisuals {
    pub egui_visuals: egui::Visuals,
    pub clear_color: wgpu::Color,
    pub grid_line_color: wgpu::Color,
}

impl Deref for ThemeVisuals {
    type Target = egui::Visuals;

    fn deref(&self) -> &Self::Target { &self.egui_visuals }
}

impl VgonioUi {
    pub fn new(
        event_loop: EventLoopProxy<VgonioEvent>,
        config: Arc<Config>,
        gpu: &GpuContext,
        gui: &mut GuiRenderer,
    ) -> Self {
        Self {
            config,
            event_loop: event_loop.clone(),
            tools: Tools::new(event_loop.clone(), gpu, gui),
            // simulation_workspace: SimulationWorkspace::new(event_loop.clone(), cache.clone()),
            drag_drop: FileDragDrop::new(event_loop),
            theme: ThemeState::default(),
            navigator: NavigationGizmo::new(GizmoOrientation::Global),
            outliner: Outliner::new(),
            image_cache: Arc::new(Mutex::new(Default::default())),
            enable_visual_grid: true,
            right_panel_expanded: true,
            left_panel_expanded: false,
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.navigator.update_matrices(model, view, proj);
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        self.theme.update(ctx);
        egui::TopBottomPanel::top("vgonio_top_panel")
            .exact_height(28.0)
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    self.menu_bar(ui);
                });
            });

        self.tools.show(ctx);
        //self.simulation_workspace.show(ctx);
        self.drag_drop.show(ctx);

        self.navigator.show(ctx);

        if self.right_panel_expanded {
            egui::SidePanel::right("vgonio_right_panel")
                .min_width(300.0)
                .default_width(460.0)
                .resizable(true)
                .show(ctx, |ui| self.outliner.ui(ui));
        }
    }

    pub fn set_theme(&mut self, theme: Theme) { self.theme.set(theme); }

    pub fn current_theme_visuals(&self) -> &ThemeVisuals { self.theme.current_theme_visuals() }

    pub fn outliner(&self) -> &Outliner { &self.outliner }

    pub fn outliner_mut(&mut self) -> &mut Outliner { &mut self.outliner }
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

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        ui.set_height(28.0);
        let icon_image = if self.theme.current_theme() == Theme::Dark {
            self.icon_image(&Icon::VGONIO_MENU_DARK)
        } else {
            self.icon_image(&Icon::VGONIO_MENU_LIGHT)
        };
        let desired_icon_height = (ui.max_rect().height() - 4.0).at_most(28.0);
        let image_size = icon_image.size_vec2() * (desired_icon_height / icon_image.size_vec2().y);
        let texture_id = icon_image.texture_id(ui.ctx());

        ui.menu_image_button(texture_id, image_size, |ui| {
            if ui.button("About").clicked() {
                println!("TODO: about, print build info");
            }

            ui.menu_button("New", |ui| {
                ui.menu_button("Measurement", |ui| {
                    if ui.button("BRDF").clicked() {
                        println!("TODO: new angle measurement");
                    }
                    if ui.button("Area Distribution").clicked() {
                        println!("TODO: new distance measurement");
                    }
                    if ui.button("Masking/Shadowing").clicked() {
                        println!("TODO: new angle measurement");
                    }
                });

                if ui.button("General").clicked() {
                    println!("TODO: new general file")
                }
                if ui.button("Height field").clicked() {
                    println!("TODO: new height field");
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
                        println!("TODO: open item {i}");
                    }
                }
            });

            ui.add_space(6.0);

            {
                if ui.button("Save...").clicked() {
                    println!("TODO: save");
                }
            }

            ui.menu_button("Edit", |ui| {
                {
                    if ui.button("     Undo").clicked() {
                        println!("TODO: undo");
                    }
                    if ui.button("     Redo").clicked() {
                        println!("TODO: file");
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
                        ui.add(super::misc::toggle(&mut self.enable_visual_grid));
                    });
                }

                ui.separator();

                if ui.button("\u{2699} Preferences").clicked() {
                    println!("TODO: open preferences window");
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui.button("\u{1F4D8} Console").clicked() {
                    println!("TODO: open console window");
                }
                if ui.button("Scratch").clicked() {
                    self.tools.toggle::<Scratch>();
                }
                if ui.button("\u{1F5E0} Plotting").clicked() {
                    self.tools.toggle::<PlottingInspector>();
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

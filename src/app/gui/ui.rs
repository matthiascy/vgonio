use super::{simulation::SimulationWorkspace, state::GuiRenderer, VgonioEvent};
use crate::{
    app::{
        cache::{Cache, Handle},
        gfx::GpuContext,
        gui::{tools::Tools, VgonioApp},
        Config,
    },
    msurf::MicroSurface,
};
use glam::Mat4;
use std::{cell::RefCell, fmt::Write, ops::Deref, sync::Arc};
use winit::event_loop::EventLoopProxy;

/// Implementation of the GUI for vgonio application.
pub struct VgonioGuiState {
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy<VgonioEvent>,

    theme: Theme,

    theme_visuals: [ThemeVisuals; 2],

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    pub simulation_workspace: SimulationWorkspace, // TODO: make private

    /// Files dropped in the window area.
    dropped_files: Vec<egui::DroppedFile>,

    /// Handles to the micro-surfaces that are currently opened.
    opened_msurfs: Vec<Handle<MicroSurface>>,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Theme {
    Dark,
    Light,
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

impl VgonioGuiState {
    pub fn new(
        event_loop: EventLoopProxy<VgonioEvent>,
        config: Arc<Config>,
        cache: Arc<RefCell<Cache>>,
        gpu: &GpuContext,
        gui: &mut GuiRenderer,
    ) -> Self {
        Self {
            config,
            event_loop: event_loop.clone(),
            tools: Tools::new(event_loop.clone(), gpu, gui),
            simulation_workspace: SimulationWorkspace::new(event_loop, cache),
            dropped_files: vec![],
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
            theme: Theme::Dark,
            opened_msurfs: vec![],
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        self.simulation_workspace
            .update_gizmo_matrices(model, view, proj);
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("vgonio-menu-bar").show(ctx, |ui| {
            self.menu_bar(ui);
        });
        self.tools.show(ctx);
        self.simulation_workspace.show(ctx);
        self.file_drag_and_drop(ctx);
    }

    pub fn set_theme(&mut self, theme: Theme) { self.theme = theme; }

    pub fn theme_visuals(&self) -> &ThemeVisuals { &self.theme_visuals[self.theme as usize] }
}

impl VgonioGuiState {
    fn file_drag_and_drop(&mut self, ctx: &egui::Context) {
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
            painter.rect_filled(screen_rect, 0.0, Color32::from_black_alpha(192));
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
            self.dropped_files = dropped_files;
        }

        // Show dropped files (if any):
        if !self.dropped_files.is_empty() {
            let mut open = true;

            Window::new("Dropped files")
                .open(&mut open)
                .show(ctx, |ui| {
                    for file in &self.dropped_files {
                        let mut info = if let Some(path) = &file.path {
                            path.display().to_string()
                        } else if !file.name.is_empty() {
                            file.name.clone()
                        } else {
                            "???".to_owned()
                        };
                        if let Some(bytes) = &file.bytes {
                            write!(info, " ({} bytes)", bytes.len()).unwrap();
                        }
                        ui.label(info);
                    }
                });
            if !open {
                self.dropped_files.clear();
            }
        }
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        // FIXME: temporarily disabled when using the web backend.
        #[cfg(not(target_arch = "wasm32"))]
        {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    {
                        ui.menu_button("\u{1F4C4} New", |ui| {
                            if ui.button("General").clicked() {
                                println!("TODO: new general file")
                            }
                            if ui.button("Height field").clicked() {
                                println!("TODO: new height field");
                            }
                        });
                        if ui.button("\u{1F4C2} Open").clicked() {
                            use rfd::AsyncFileDialog;
                            let dir = self
                                .config
                                .user_data_dir()
                                .unwrap_or_else(|| self.config.sys_data_dir());
                            let task = AsyncFileDialog::new().set_directory(dir).pick_file();
                            let event_loop_proxy = self.event_loop.clone();
                            std::thread::spawn(move || {
                                pollster::block_on(async {
                                    let file_handle = task.await;
                                    if let Some(hd) = file_handle {
                                        if event_loop_proxy
                                            .send_event(VgonioEvent::OpenFile(hd))
                                            .is_err()
                                        {
                                            log::warn!("[EVENT] Failed to send OpenFile event");
                                        }
                                    }
                                })
                            });
                        }
                        ui.menu_button("\u{1F4DC} Open Recent", |ui| {
                            for i in 0..10 {
                                if ui.button(format!("item {i}")).clicked() {
                                    println!("TODO: open item {i}");
                                }
                            }
                        });
                    }

                    ui.separator();

                    {
                        if ui.button("\u{1F4E9} Save").clicked() {
                            println!("TODO: save");
                        }
                        if ui.button("     Save As").clicked() {
                            println!("TODO: save as");
                        }
                        if ui.button("     Save Copy").clicked() {
                            println!("TODO: save copy");
                        }
                    }

                    ui.separator();

                    {
                        ui.menu_button("     Clean up", |ui| {
                            ui.spacing();
                            if ui.button("Cache").clicked() {
                                println!("TODO: clear cache");
                            }
                        });
                    }

                    ui.separator();

                    if ui.button("     Quit").clicked()
                        && self.event_loop.send_event(VgonioEvent::Quit).is_err()
                    {
                        log::warn!("[EVENT] Failed to send Quit event.");
                    }
                });
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

                    if ui.button("\u{2699} Preferences").clicked() {
                        println!("TODO: open preferences window");
                    }
                });
                ui.menu_button("Tools", |ui| {
                    // TODO: iterater
                    if ui.button("\u{1F4D8} Console").clicked() {
                        println!("TODO: open console window");
                    }
                    if ui.button("Plot").clicked() {
                        self.tools.toggle("Plot");
                    }
                    if ui.button("Visual Debugger").clicked() {
                        self.tools.toggle("Visual Debugger");
                    }
                    if ui.button("Scratch").clicked() {
                        self.tools.toggle("Scratch");
                    }
                    if ui.button("Sampling Debugger").clicked() {
                        self.tools.toggle("Sampling Debugger");
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("\u{1F4DA} Docs").clicked() {
                        println!("TODO: docs");
                    }
                    if ui.button("     About").clicked() {
                        println!("TODO: about");
                    }
                });
                ui.separator();
                self.theme_toggle_button(ui);
            });
        }
    }

    fn theme_toggle_button(&mut self, ui: &mut egui::Ui) {
        match self.theme {
            Theme::Dark => {
                if ui
                    .add(egui::Button::new("☀").frame(false))
                    .on_hover_text("Switch to light mode")
                    .clicked()
                {
                    self.set_theme(Theme::Light);
                    ui.ctx()
                        .set_visuals(self.theme_visuals[self.theme as usize].egui_visuals.clone());
                }
            }
            Theme::Light => {
                if ui
                    .add(egui::Button::new("🌙").frame(false))
                    .on_hover_text("Switch to dark mode")
                    .clicked()
                {
                    self.set_theme(Theme::Dark);
                    ui.ctx()
                        .set_visuals(self.theme_visuals[self.theme as usize].egui_visuals.clone());
                }
            }
        }
    }
}

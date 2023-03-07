use super::{simulation::SimulationWorkspace, state::GuiRenderer, VgonioEvent};
use crate::app::{cache::Cache, gfx::GpuContext, gui::tools::Tools, Config};
use glam::Mat4;
use std::{cell::RefCell, fmt::Write, sync::Arc};
use winit::event_loop::EventLoopProxy;

/// Implementation of the GUI for vgonio application.
pub struct VgonioUi {
    /// The configuration of the application. See [`Config`].
    config: Arc<Config>,

    /// Files dropped in the window area.
    dropped_files: Vec<egui::DroppedFile>,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy<VgonioEvent>,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    pub simulation_workspace: SimulationWorkspace, // TODO: make private
}

impl VgonioUi {
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
            dropped_files: vec![],
            tools: Tools::new(event_loop.clone(), gpu, gui),
            simulation_workspace: SimulationWorkspace::new(event_loop, cache),
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
}

impl VgonioUi {
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
            egui::Window::new("Dropped files")
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
                egui::widgets::global_dark_light_mode_switch(ui);
            });
        }
    }
}

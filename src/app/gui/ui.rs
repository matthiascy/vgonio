use super::{analysis::AnalysisWorkspace, simulation::SimulationWorkspace, VgonioEvent};
use crate::app::{
    cache::Cache,
    gui::{tools::Tools, GuiContext},
    Config,
};
use glam::Mat4;
use std::{cell::RefCell, fmt::Write, sync::Arc};
use winit::event_loop::EventLoopProxy;

pub trait Workspace {
    fn name(&self) -> &str;

    fn show(&mut self, ctx: &egui::Context);
}

pub struct Workspaces {
    pub(crate) simulation: SimulationWorkspace,
    pub(crate) analysis: AnalysisWorkspace,
}

impl Workspaces {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>, cache: Arc<RefCell<Cache>>) -> Self {
        Self {
            simulation: SimulationWorkspace::new(event_loop, cache),
            analysis: AnalysisWorkspace {},
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut dyn Workspace)> {
        vec![
            ("Simulation", &mut self.simulation as &mut dyn Workspace),
            ("Analysis", &mut self.analysis as &mut dyn Workspace),
        ]
        .into_iter()
    }
}

/// Implementation of the GUI for vgonio application.
pub struct VgonioGui {
    /// The configuration of the application. See [`VgonioConfig`].
    config: Arc<Config>,

    /// Workspaces are essentially predefined window layouts for certain usage.
    pub(crate) workspaces: Workspaces,

    /// Files dropped in the window area.
    dropped_files: Vec<egui::DroppedFile>,

    /// Current activated workspace.
    selected_workspace: String,

    /// Tools are small windows that can be opened and closed.
    pub(crate) tools: Tools,

    /// Event loop proxy for sending user defined events.
    event_loop: EventLoopProxy<VgonioEvent>,
}

impl VgonioGui {
    pub fn new(
        event_loop: EventLoopProxy<VgonioEvent>,
        config: Arc<Config>,
        cache: Arc<RefCell<Cache>>,
    ) -> Self {
        let workspaces = Workspaces::new(event_loop.clone(), cache);

        Self {
            config,
            event_loop: event_loop.clone(),
            workspaces,
            dropped_files: vec![],
            selected_workspace: "".to_string(),
            tools: Tools::new(event_loop),
        }
    }

    pub fn update_gizmo_matrices(&mut self, model: Mat4, view: Mat4, proj: Mat4) {
        if self.selected_workspace == "Simulation" {
            self.workspaces
                .simulation
                .update_gizmo_matrices(model, view, proj);
        }
    }

    pub fn current_workspace_name(&self) -> &str { &self.selected_workspace }

    pub fn show(&mut self, ctx: &GuiContext) {
        let ctx = &ctx.egui_context();
        if self.selected_workspace.is_empty() {
            self.selected_workspace = self.workspaces.iter_mut().next().unwrap().0.to_owned();
        }

        egui::TopBottomPanel::top("vgonio_menu_bar").show(ctx, |ui| {
            self.menu_bar(ui);
        });

        self.tools.show(ctx);

        for (ws_name, ws) in self.workspaces.iter_mut() {
            if ws_name == self.selected_workspace || ctx.memory().everything_is_visible() {
                ws.show(ctx);
            }
        }

        self.file_drag_and_drop(ctx);
    }
}

impl VgonioGui {
    fn file_drag_and_drop(&mut self, ctx: &egui::Context) {
        use egui::*;

        // Preview hovering files:
        if !ctx.input().raw.hovered_files.is_empty() {
            let mut text = "Dropping files:\n".to_owned();
            for file in &ctx.input().raw.hovered_files {
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

            let screen_rect = ctx.input().screen_rect();
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
        if !ctx.input().raw.dropped_files.is_empty() {
            self.dropped_files = ctx.input().raw.dropped_files.clone();
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
                        let task = AsyncFileDialog::new()
                            .set_directory(&self.config.user_config.data_files_dir)
                            .pick_file();
                        let event_loop_proxy = self.event_loop.clone();
                        std::thread::spawn(move || {
                            pollster::block_on(async {
                                let file = task.await;
                                if let Some(file) = file {
                                    if event_loop_proxy
                                        .send_event(VgonioEvent::OpenFile(file.path().into()))
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
                            if ui.button(format!("item {}", i)).clicked() {
                                println!("TODO: open item {}", i);
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

            for (ws_name, ws) in self.workspaces.iter_mut() {
                if ui
                    .selectable_label(self.selected_workspace == ws_name, ws.name())
                    .clicked()
                {
                    self.selected_workspace = ws_name.to_owned();
                    ui.output().open_url(format!("#{}", ws_name));
                }
            }

            ui.separator();
            egui::widgets::global_dark_light_mode_switch(ui);
        });
    }
}

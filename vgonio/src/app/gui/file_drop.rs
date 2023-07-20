use crate::app::gui::{VgonioEvent, VgonioEventLoop};

/// Implementation of the drag and drop functionality.
pub struct FileDragDrop {
    event_loop: VgonioEventLoop,
    files: Vec<egui::DroppedFile>,
}

impl FileDragDrop {
    pub fn new(event_loop: VgonioEventLoop) -> Self {
        log::info!("Initialized file drag and drop");
        Self {
            event_loop,
            files: vec![],
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        use egui::*;
        use std::fmt::Write;

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

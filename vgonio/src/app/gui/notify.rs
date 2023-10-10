use egui::WidgetText;
use egui_toast::{Toast, ToastKind, ToastOptions, Toasts};

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum NotifyKind {
    Info,
    Warning,
    Error,
    Success,
}

impl From<NotifyKind> for ToastKind {
    fn from(value: NotifyKind) -> Self {
        match value {
            NotifyKind::Info => ToastKind::Info,
            NotifyKind::Warning => ToastKind::Warning,
            NotifyKind::Error => ToastKind::Error,
            NotifyKind::Success => ToastKind::Success,
        }
    }
}

pub struct NotifySystem {
    inner: Toasts,
}

impl NotifySystem {
    pub fn new() -> Self {
        Self {
            inner: Toasts::new()
                .anchor(egui::Align2::LEFT_BOTTOM, (10.0, -10.0))
                .direction(egui::Direction::BottomUp),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) { self.inner.show(ctx); }

    pub fn notify<M: Into<WidgetText>>(&mut self, kind: NotifyKind, msg: M, secs: f64) {
        self.inner.add(Toast {
            kind: kind.into(),
            text: msg.into(),
            options: ToastOptions::default()
                .duration_in_seconds(secs)
                .show_icon(true)
                .show_icon(true),
        });
    }
}

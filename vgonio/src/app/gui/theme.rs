/// ThemeKind is used to select the theme to use.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThemeKind {
    /// Dark theme.
    Dark = 0,
    /// Light theme.
    Light = 1,
}

/// Colour and style information for a theme.
pub struct ThemeVisuals {
    pub egui_visuals: egui::Visuals,
    pub clear_color: wgpu::Color,
    pub grid_line_color: wgpu::Color,
}

/// Structure to hold the current theme state and visuals.
pub struct ThemeState {
    kind: ThemeKind,
    visuals: [ThemeVisuals; 2],
    is_dirty: bool,
}

impl Default for ThemeState {
    fn default() -> Self {
        Self {
            visuals: [
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
            kind: ThemeKind::Light,
            is_dirty: true,
        }
    }
}

impl ThemeState {
    /// Updates the egui context with the current theme visuals.
    pub fn update_context(&mut self, ctx: &egui::Context) {
        if self.is_dirty {
            self.is_dirty = false;
            ctx.set_visuals(self.visuals[self.kind as usize].egui_visuals.clone());
        }
    }

    /// Sets the current theme kind.
    ///
    /// This will not update immediately, but will be updated on the next call
    /// to `update`.
    pub fn set_theme_kind(&mut self, kind: ThemeKind) {
        if self.kind != kind {
            self.kind = kind;
            self.is_dirty = true;
        }
    }

    /// Returns the current theme kind.
    pub fn kind(&self) -> ThemeKind { self.kind }

    /// Returns the current theme visuals.
    pub fn visuals(&self) -> &ThemeVisuals { &self.visuals[self.kind as usize] }

    /// Returns true if the current theme is dark mode.
    pub fn is_dark(&self) -> bool { self.kind == ThemeKind::Dark }

    /// Returns true if the current theme is light mode.
    pub fn is_light(&self) -> bool { self.kind == ThemeKind::Light }
}

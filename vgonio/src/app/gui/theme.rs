use std::ops::Deref;

pub struct ThemeState {
    pub theme: Theme,
    pub theme_visuals: [ThemeVisuals; 2],
    pub need_update: bool,
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

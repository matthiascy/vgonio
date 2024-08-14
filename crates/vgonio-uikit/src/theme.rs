use egui::{
    style::{Selection, Spacing, WidgetVisuals, Widgets},
    Color32, Rounding, Shadow, Stroke, Style, Visuals,
};

/// ThemeKind is used to select the theme to use.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThemeKind {
    /// Dark theme.
    Dark = 0,
    /// Light theme.
    Light = 1,
}

/// Trait for a theme.
pub trait Theme {
    fn rosewater(&self) -> Color32;
    fn maroon(&self) -> Color32;
    fn peach(&self) -> Color32;
    fn blue(&self) -> Color32;
    fn text(&self) -> Color32;
    fn overlay2(&self) -> Color32;
    fn overlay1(&self) -> Color32;
    fn overlay0(&self) -> Color32;
    fn bg_hovered_fill(&self) -> Color32;
    fn bg_active_fill(&self) -> Color32;
    fn bg_inactive_fill(&self) -> Color32;
    fn base(&self) -> Color32;
    fn mantle(&self) -> Color32;
    fn crust(&self) -> Color32;

    fn kind(&self) -> ThemeKind;

    fn is_light(&self) -> bool { self.kind() == ThemeKind::Light }

    /// Color with no gamma correction and no premultiplied alpha.
    fn clear_color(&self) -> (f32, f32, f32, f32);

    /// Color with no gamma correction and no premultiplied alpha.
    fn grid_line_color(&self) -> (f32, f32, f32, f32);

    fn visuals(&self) -> Visuals {
        Visuals {
            override_text_color: Some(self.text()),
            hyperlink_color: self.rosewater(),
            faint_bg_color: self.bg_inactive_fill(),
            extreme_bg_color: self.crust(),
            code_bg_color: self.mantle(),
            warn_fg_color: self.peach(),
            error_fg_color: self.maroon(),
            window_fill: self.base(),
            panel_fill: self.base(),
            window_stroke: Stroke::new(1.0, self.overlay1()),
            widgets: Widgets {
                noninteractive: self.widget_visuals(self.base()),
                inactive: self.widget_visuals(self.bg_inactive_fill()),
                hovered: self.widget_visuals(self.bg_hovered_fill()),
                active: self.widget_visuals(self.bg_active_fill()),
                open: self.widget_visuals(self.bg_inactive_fill()),
            },
            selection: Selection {
                bg_fill: self
                    .blue()
                    .linear_multiply(if self.is_light() { 0.4 } else { 0.2 }),
                stroke: Stroke::new(1.0, self.overlay1()),
            },
            window_shadow: Shadow::NONE,
            popup_shadow: Shadow {
                offset: egui::Vec2 { x: 0.0, y: 0.0 },
                blur: 10.0,
                spread: 1.0,
                color: self.base(),
            },
            dark_mode: self.is_light(),
            ..Visuals::default()
        }
    }

    fn style(&self) -> Style {
        Style {
            spacing: self.spacing(),
            text_styles: self.text_styles().into(),
            visuals: self.visuals(),
            ..Style::default()
        }
    }

    /// Returns the spacing for the theme style.
    fn spacing(&self) -> Spacing {
        Spacing {
            item_spacing: egui::Vec2 { x: 5.0, y: 5.0 },
            window_margin: egui::Margin {
                left: 6.0,
                right: 6.0,
                top: 6.0,
                bottom: 6.0,
            },
            button_padding: egui::Vec2 { x: 6.0, y: 6.0 },
            menu_margin: egui::Margin {
                left: 6.0,
                right: 6.0,
                top: 6.0,
                bottom: 6.0,
            },
            indent: 18.0,
            interact_size: egui::Vec2 { x: 40.0, y: 20.0 },
            slider_width: 100.0,
            slider_rail_height: 8.0,
            combo_width: 100.0,
            text_edit_width: 280.0,
            icon_width: 14.0,
            icon_width_inner: 8.0,
            icon_spacing: 4.0,
            default_area_size: egui::Vec2 { x: 400.0, y: 400.0 },
            tooltip_width: 500.0,
            menu_width: 400.0,
            menu_spacing: 2.0,
            indent_ends_with_horizontal_line: false,
            combo_height: 200.0,
            scroll: egui::style::ScrollStyle {
                floating: true,
                bar_width: 10.0,
                foreground_color: true,
                floating_allocated_width: 0.0,
                dormant_background_opacity: 0.0,
                dormant_handle_opacity: 0.0,
                handle_min_length: 12.0,
                bar_inner_margin: 4.0,
                bar_outer_margin: 0.0,
                floating_width: 2.0,
                active_background_opacity: 0.4,
                interact_background_opacity: 0.7,
                active_handle_opacity: 0.6,
                interact_handle_opacity: 1.0,
            },
        }
    }

    fn text_styles(&self) -> [(egui::TextStyle, egui::FontId); 5] {
        [
            (
                egui::TextStyle::Small,
                egui::FontId::new(11.0, egui::FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Body,
                egui::FontId::new(14.5, egui::FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Button,
                egui::FontId::new(14.5, egui::FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Heading,
                egui::FontId::new(20.0, egui::FontFamily::Proportional),
            ),
            (
                egui::TextStyle::Monospace,
                egui::FontId::new(14.5, egui::FontFamily::Monospace),
            ),
        ]
    }

    fn widget_visuals(&self, bg_fill: Color32) -> WidgetVisuals {
        WidgetVisuals {
            bg_fill,
            weak_bg_fill: bg_fill,
            bg_stroke: Stroke::new(0.0, self.overlay1()),
            fg_stroke: Stroke::new(2.0, self.text()),
            rounding: Rounding::same(6.0),
            expansion: 0.0,
        }
    }
}

pub struct LightTheme;

impl Theme for LightTheme {
    fn rosewater(&self) -> Color32 { Color32::from_rgb(220, 138, 120) }
    fn maroon(&self) -> Color32 { Color32::from_rgb(230, 69, 83) }
    fn peach(&self) -> Color32 { Color32::from_rgb(254, 100, 11) }
    fn blue(&self) -> Color32 { Color32::from_rgb(30, 102, 245) }
    fn text(&self) -> Color32 { Color32::from_rgb(76, 79, 105) }
    fn overlay2(&self) -> Color32 { Color32::from_rgb(124, 127, 147) }
    fn overlay1(&self) -> Color32 { Color32::from_rgb(140, 143, 161) }
    fn overlay0(&self) -> Color32 { Color32::from_rgb(156, 160, 176) }
    fn bg_hovered_fill(&self) -> Color32 { Color32::from_rgb(172, 176, 190) }
    fn bg_active_fill(&self) -> Color32 { Color32::from_rgb(188, 192, 204) }
    fn bg_inactive_fill(&self) -> Color32 { Color32::from_rgb(204, 208, 218) }
    fn base(&self) -> Color32 { Color32::from_rgb(239, 241, 245) }
    fn mantle(&self) -> Color32 { Color32::from_rgb(230, 233, 239) }
    fn crust(&self) -> Color32 { Color32::from_rgb(220, 224, 232) }

    fn kind(&self) -> ThemeKind { ThemeKind::Light }

    fn clear_color(&self) -> (f32, f32, f32, f32) { (0.208, 0.208, 0.208, 1.0) }

    fn grid_line_color(&self) -> (f32, f32, f32, f32) { (0.68, 0.68, 0.68, 1.0) }
}

pub struct DarkTheme;

impl Theme for DarkTheme {
    fn rosewater(&self) -> Color32 { Color32::from_rgb(245, 224, 220) }

    fn maroon(&self) -> Color32 { Color32::from_rgb(235, 160, 172) }

    fn peach(&self) -> Color32 { Color32::from_rgb(250, 179, 135) }

    fn blue(&self) -> Color32 { Color32::from_rgb(137, 180, 250) }

    fn text(&self) -> Color32 { Color32::from_rgb(205, 214, 244) }

    fn overlay2(&self) -> Color32 { Color32::from_rgb(147, 153, 178) }

    fn overlay1(&self) -> Color32 { Color32::from_rgb(127, 132, 156) }

    fn overlay0(&self) -> Color32 { Color32::from_rgb(108, 112, 134) }

    fn bg_hovered_fill(&self) -> Color32 { Color32::from_rgb(88, 91, 112) }

    fn bg_active_fill(&self) -> Color32 { Color32::from_rgb(69, 71, 90) }

    fn bg_inactive_fill(&self) -> Color32 { Color32::from_rgb(49, 50, 68) }

    fn base(&self) -> Color32 { Color32::from_rgb(30, 30, 46) }

    fn mantle(&self) -> Color32 { Color32::from_rgb(24, 24, 37) }

    fn crust(&self) -> Color32 { Color32::from_rgb(17, 17, 27) }

    fn kind(&self) -> ThemeKind { ThemeKind::Dark }

    fn clear_color(&self) -> (f32, f32, f32, f32) { (0.046, 0.046, 0.046, 1.0) }

    fn grid_line_color(&self) -> (f32, f32, f32, f32) { (0.4, 0.4, 0.4, 1.0) }
}

use egui::{Image, ImageSource};

#[derive(Clone, Copy, Debug)]
pub struct Icon {
    /// Textual unique id
    pub id: &'static str,
    /// PNG bytes
    pub bytes: &'static [u8],
}

impl Icon {
    #[inline]
    pub const fn new(id: &'static str, bytes: &'static [u8]) -> Self { Self { id, bytes } }

    #[inline]
    pub fn as_image_source(&self) -> ImageSource<'static> {
        ImageSource::Bytes {
            uri: self.id.into(),
            bytes: self.bytes.into(),
        }
    }

    #[inline]
    pub fn as_image(&self) -> Image { Image::new(self.as_image_source()) }
}

pub const VGONIO_MENU_LIGHT: Icon = Icon::new(
    "vgonio_menu_light",
    include_bytes!("assets/icons/vgonio_menu_light.png"),
);

pub const VGONIO_MENU_DARK: Icon = Icon::new(
    "vgonio_menu_dark",
    include_bytes!("assets/icons/vgonio_menu_dark.png"),
);

pub const BOTTOM_PANEL_TOGGLE: Icon = Icon::new(
    "bottom_panel_toggle",
    include_bytes!("assets/icons/bottom_panel_toggle.png"),
);

pub const LEFT_PANEL_TOGGLE: Icon = Icon::new(
    "left_panel_toggle",
    include_bytes!("assets/icons/left_panel_toggle.png"),
);

pub const RIGHT_PANEL_TOGGLE: Icon = Icon::new(
    "right_panel_toggle",
    include_bytes!("assets/icons/right_panel_toggle.png"),
);

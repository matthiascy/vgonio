use egui::epaint::ahash::HashMap;
use egui_extras::RetainedImage;
use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc, Mutex,
};
use vgcore::error::VgonioError;

#[derive(Clone, Copy, Debug)]
pub struct Icon {
    /// Textual unique id
    pub id: &'static str,
    /// PNG bytes
    pub bytes: &'static [u8],
}

impl Icon {
    pub const fn new(id: &'static str, bytes: &'static [u8]) -> Self { Self { id, bytes } }
}

pub const VGONIO_ICONS: [Icon; 5] = [
    Icon::new(
        "vgonio_menu_light",
        include_bytes!("assets/icons/vgonio_menu_light.png"),
    ),
    Icon::new(
        "vgonio_menu_dark",
        include_bytes!("assets/icons/vgonio_menu_dark.png"),
    ),
    Icon::new(
        "bottom_panel_toggle",
        include_bytes!("assets/icons/bottom_panel_toggle.png"),
    ),
    Icon::new(
        "left_panel_toggle",
        include_bytes!("assets/icons/left_panel_toggle.png"),
    ),
    Icon::new(
        "right_panel_toggle",
        include_bytes!("assets/icons/right_panel_toggle.png"),
    ),
];

/// A collection of icons.
pub struct Icons(Arc<Mutex<HashMap<&'static str, Arc<RetainedImage>>>>);

/// Returns the global icons cache.
fn get_icons_cache() -> &'static mut Icons {
    static ICONS: AtomicPtr<Icons> = AtomicPtr::new(std::ptr::null_mut());
    let mut ptr = ICONS.load(Ordering::Acquire);
    if ptr.is_null() {
        let icons = Box::new(Icons(Arc::new(Mutex::new(HashMap::default()))));
        ptr = Box::into_raw(icons);
        if ICONS
            .compare_exchange(
                std::ptr::null_mut(),
                ptr,
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_err()
        {
            drop(unsafe { Box::from_raw(ptr) })
        }
    }
    unsafe { &mut *ptr }
}

/// Returns the icon image with the given name.
pub fn get_icon_image(name: &'static str) -> Option<Arc<RetainedImage>> {
    let icons = get_icons_cache();
    let mut icons = icons.0.lock().unwrap();
    if let Some(icon) = VGONIO_ICONS.iter().find(|icon| icon.id == name) {
        Some(icons.entry(name).or_insert_with(|| {
            let image = load_image_from_bytes(icon.bytes).unwrap();
            Arc::new(RetainedImage::from_color_image(name, image))
        }))
        .cloned()
    } else {
        None
    }
}

fn load_image_from_bytes(bytes: &[u8]) -> Result<egui::ColorImage, VgonioError> {
    let image = image::load_from_memory(bytes)
        .map_err(|err| {
            VgonioError::new(format!("Failed to load image \"{}\"from bytes", err), None)
        })?
        .into_rgba8();
    let size = [image.width() as _, image.height() as _];
    let pixels = image.as_flat_samples();
    Ok(egui::ColorImage::from_rgba_unmultiplied(
        size,
        pixels.as_slice(),
    ))
}

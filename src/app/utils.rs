use winit::event::{ModifiersState, VirtualKeyCode};
use winit::window::CursorIcon;

#[inline]
pub(crate) fn winit_to_egui_key_code(key: VirtualKeyCode) -> Option<egui::Key> {
    Some(match key {
        VirtualKeyCode::Down => egui::Key::ArrowDown,
        VirtualKeyCode::Left => egui::Key::ArrowLeft,
        VirtualKeyCode::Right => egui::Key::ArrowRight,
        VirtualKeyCode::Up => egui::Key::ArrowUp,
        VirtualKeyCode::Escape => egui::Key::Escape,
        VirtualKeyCode::Tab => egui::Key::Tab,
        VirtualKeyCode::Back => egui::Key::Backspace,
        VirtualKeyCode::Return => egui::Key::Enter,
        VirtualKeyCode::Space => egui::Key::Space,
        VirtualKeyCode::Insert => egui::Key::Insert,
        VirtualKeyCode::Delete => egui::Key::Delete,
        VirtualKeyCode::Home => egui::Key::Home,
        VirtualKeyCode::End => egui::Key::End,
        VirtualKeyCode::PageUp => egui::Key::PageUp,
        VirtualKeyCode::PageDown => egui::Key::PageDown,
        VirtualKeyCode::Key0 => egui::Key::Num0,
        VirtualKeyCode::Key1 => egui::Key::Num1,
        VirtualKeyCode::Key2 => egui::Key::Num2,
        VirtualKeyCode::Key3 => egui::Key::Num3,
        VirtualKeyCode::Key4 => egui::Key::Num4,
        VirtualKeyCode::Key5 => egui::Key::Num5,
        VirtualKeyCode::Key6 => egui::Key::Num6,
        VirtualKeyCode::Key7 => egui::Key::Num7,
        VirtualKeyCode::Key8 => egui::Key::Num8,
        VirtualKeyCode::Key9 => egui::Key::Num9,
        VirtualKeyCode::A => egui::Key::A,
        VirtualKeyCode::B => egui::Key::B,
        VirtualKeyCode::C => egui::Key::C,
        VirtualKeyCode::D => egui::Key::D,
        VirtualKeyCode::E => egui::Key::E,
        VirtualKeyCode::F => egui::Key::F,
        VirtualKeyCode::G => egui::Key::G,
        VirtualKeyCode::H => egui::Key::H,
        VirtualKeyCode::I => egui::Key::I,
        VirtualKeyCode::J => egui::Key::J,
        VirtualKeyCode::K => egui::Key::K,
        VirtualKeyCode::L => egui::Key::L,
        VirtualKeyCode::M => egui::Key::M,
        VirtualKeyCode::N => egui::Key::N,
        VirtualKeyCode::O => egui::Key::O,
        VirtualKeyCode::P => egui::Key::P,
        VirtualKeyCode::Q => egui::Key::Q,
        VirtualKeyCode::R => egui::Key::R,
        VirtualKeyCode::S => egui::Key::S,
        VirtualKeyCode::T => egui::Key::T,
        VirtualKeyCode::U => egui::Key::U,
        VirtualKeyCode::V => egui::Key::V,
        VirtualKeyCode::W => egui::Key::W,
        VirtualKeyCode::X => egui::Key::X,
        VirtualKeyCode::Y => egui::Key::Y,
        VirtualKeyCode::Z => egui::Key::Z,
        _ => return None,
    })
}

#[inline]
pub(crate) fn winit_to_egui_modifiers(modifiers: ModifiersState) -> egui::Modifiers {
    egui::Modifiers {
        alt: modifiers.alt(),
        ctrl: modifiers.ctrl(),
        shift: modifiers.shift(),
        #[cfg(target_os = "macos")]
        mac_cmd: modifiers.logo(),
        #[cfg(target_os = "macos")]
        command: modifiers.logo(),
        #[cfg(not(target_os = "macos"))]
        mac_cmd: false,
        #[cfg(not(target_os = "macos"))]
        command: modifiers.ctrl(),
    }
}

#[inline]
pub(crate) fn egui_to_winit_cursor_icon(
    icon: egui::CursorIcon,
) -> Option<winit::window::CursorIcon> {
    use egui::CursorIcon::*;

    match icon {
        Default => Some(CursorIcon::Default),
        ContextMenu => Some(CursorIcon::ContextMenu),
        Help => Some(CursorIcon::Help),
        PointingHand => Some(CursorIcon::Hand),
        Progress => Some(CursorIcon::Progress),
        Wait => Some(CursorIcon::Wait),
        Cell => Some(CursorIcon::Cell),
        Crosshair => Some(CursorIcon::Crosshair),
        Text => Some(CursorIcon::Text),
        VerticalText => Some(CursorIcon::VerticalText),
        Alias => Some(CursorIcon::Alias),
        Copy => Some(CursorIcon::Copy),
        Move => Some(CursorIcon::Move),
        NoDrop => Some(CursorIcon::NoDrop),
        NotAllowed => Some(CursorIcon::NotAllowed),
        Grab => Some(CursorIcon::Grab),
        Grabbing => Some(CursorIcon::Grabbing),
        AllScroll => Some(CursorIcon::AllScroll),
        ResizeHorizontal => Some(CursorIcon::EwResize),
        ResizeNeSw => Some(CursorIcon::NeswResize),
        ResizeNwSe => Some(CursorIcon::NwseResize),
        ResizeVertical => Some(CursorIcon::NsResize),
        ZoomIn => Some(CursorIcon::ZoomIn),
        ZoomOut => Some(CursorIcon::ZoomOut),
        None => Option::None,
    }
}

#[inline]
pub(crate) fn is_character_printable(chr: char) -> bool {
    let is_in_private_use_area = '\u{e000}' <= chr && chr <= '\u{f8ff}'
        || '\u{f0000}' <= chr && chr <= '\u{ffffd}'
        || '\u{100000}' <= chr && chr <= '\u{10fffd}';
    !is_in_private_use_area && !chr.is_ascii_control()
}

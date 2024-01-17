use std::collections::HashMap;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

pub struct InputState {
    pub(crate) key_map: HashMap<KeyCode, bool>,
    pub(crate) mouse_map: HashMap<MouseButton, bool>,
    pub(crate) scroll_delta: f32,
    pub(crate) cursor_delta: [f32; 2],
    pub(crate) cursor_pos: [f32; 2],
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        }
    }
}

impl InputState {
    pub fn new() -> Self { Self::default() }

    pub fn update_key_map(&mut self, key_code: KeyCode, state: ElementState) {
        *self.key_map.entry(key_code).or_insert(false) = state == ElementState::Pressed;
    }

    pub fn update_mouse_map(&mut self, button: MouseButton, state: ElementState) {
        *self.mouse_map.entry(button).or_insert(false) = state == ElementState::Pressed;
    }

    pub fn update_cursor_delta(&mut self, new_pos: PhysicalPosition<f32>) {
        self.cursor_delta = [
            new_pos.x - self.cursor_pos[0],
            new_pos.y - self.cursor_pos[1],
        ];
        self.cursor_pos = new_pos.into();
    }

    pub fn update_scroll_delta(&mut self, delta: MouseScrollDelta) {
        self.scroll_delta = match delta {
            MouseScrollDelta::LineDelta(_, y) => {
                -y * 100.0 // assuming a line is about 100 pixels
            }
            MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
        };
    }

    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        *self.key_map.get(&key_code).unwrap_or(&false)
    }

    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        *self.mouse_map.get(&button).unwrap_or(&false)
    }
}

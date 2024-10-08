//! Input handling module.

use std::collections::HashMap;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

/// Input state including key and mouse button states, cursor position and
/// scroll delta.
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
    /// Creates a new input state.
    pub fn new() -> Self { Self::default() }

    /// Returns the cursor delta.
    pub fn cursor_delta(&self) -> [f32; 2] { self.cursor_delta }

    /// Returns the scroll delta.
    pub fn scroll_delta(&self) -> f32 { self.scroll_delta }

    /// Returns the cursor position.
    pub fn cursor_pos(&self) -> [f32; 2] { self.cursor_pos }

    /// Resets the input state.
    pub fn reset(&mut self) {
        self.scroll_delta = 0.0;
        self.cursor_delta = [0.0, 0.0];
    }

    /// Updates the key map with the given key code and state.
    pub fn update_key_map(&mut self, key_code: KeyCode, state: ElementState) {
        *self.key_map.entry(key_code).or_insert(false) = state == ElementState::Pressed;
    }

    /// Updates the mouse map with the given button and state.  
    pub fn update_mouse_map(&mut self, button: MouseButton, state: ElementState) {
        *self.mouse_map.entry(button).or_insert(false) = state == ElementState::Pressed;
    }

    /// Updates the cursor position.
    pub fn update_cursor_delta(&mut self, new_pos: PhysicalPosition<f32>) {
        self.cursor_delta = [
            new_pos.x - self.cursor_pos[0],
            new_pos.y - self.cursor_pos[1],
        ];
        self.cursor_pos = new_pos.into();
    }

    /// Updates the scroll delta.
    pub fn update_scroll_delta(&mut self, delta: MouseScrollDelta) {
        self.scroll_delta = match delta {
            MouseScrollDelta::LineDelta(_, y) => {
                -y * 100.0 // assuming a line is about 100 pixels
            },
            MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
        };
    }

    /// Returns whether the given key is pressed.
    pub fn is_key_pressed(&self, key_code: KeyCode) -> bool {
        *self.key_map.get(&key_code).unwrap_or(&false)
    }

    /// Returns whether the given mouse button is pressed.
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        *self.mouse_map.get(&button).unwrap_or(&false)
    }
}

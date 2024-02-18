use egui_winit::EventResponse;
use std::ops::{Deref, DerefMut};
use winit::{event::WindowEvent, window::Window};

pub struct RawGuiContext {
    /// Context for using egui.
    pub(crate) inner: egui::Context, // tmp, remove pub
    /// States translated from winit events.
    state: egui_winit::State,
    /// A copy of the [`egui::RawInput`].
    input: egui::RawInput,
}

impl RawGuiContext {
    /// Creates a new GUI context.
    pub(crate) fn new(window: &Window) -> Self {
        let context = egui::Context::default();
        let state =
            egui_winit::State::new(context.clone(), egui::ViewportId::ROOT, window, None, None);
        Self {
            inner: context,
            state,
            input: egui::RawInput::default(),
        }
    }

    /// Prepares the context for the next frame.
    ///
    /// This function should be called before calling `begin_frame`.
    pub fn prepare(&mut self, window: &Window) { self.input = self.state.take_egui_input(window); }

    /// Updates the GUI context when a window event is received.
    ///
    /// This function should always be called before rendering the GUI.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.state.on_window_event(window, event)
    }

    /// Processes the non-rendering outputs of each frame.
    pub fn handle_platform_output(&mut self, window: &Window, output: egui::PlatformOutput) {
        self.state.handle_platform_output(window, output);
    }

    /// Runs the Ui code for one frame then returns the output.
    pub fn run(&mut self, ui: impl FnOnce(&egui::Context)) -> egui::FullOutput {
        self.inner.run(self.input.take(), ui)
    }
}

impl Deref for RawGuiContext {
    type Target = egui::Context;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl DerefMut for RawGuiContext {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

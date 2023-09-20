mod debugging;
mod sampling;
mod scratch;

use crate::app::{cache::Cache, gfx::GpuContext, gui::event::EventLoopProxy};
pub(crate) use debugging::DebuggingInspector;
pub(crate) use sampling::SamplingInspector;
pub(crate) use scratch::Scratch;
use std::sync::{Arc, RwLock};

use super::state::GuiRenderer;

pub trait Tool {
    fn name(&self) -> &str;

    fn show(&mut self, ctx: &egui::Context, open: &mut bool);

    fn ui(&mut self, ui: &mut egui::Ui);

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn as_any(&self) -> &dyn std::any::Any;
}

pub struct Tools {
    windows: Vec<Box<dyn Tool>>,
    states: Vec<bool>,
}

impl Tools {
    pub fn new(
        event_loop: EventLoopProxy,
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        cache: Cache,
    ) -> Self {
        log::info!("Initializing tools...");
        Self {
            windows: vec![
                Box::<Scratch>::default(),
                Box::new(DebuggingInspector::new(event_loop.clone(), cache)),
                Box::new(SamplingInspector::new(
                    gpu,
                    gui,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    event_loop,
                )),
            ],
            states: vec![false, false, false, false],
        }
    }

    pub fn toggle<T: 'static>(&mut self) {
        if let Some(i) = self
            .windows
            .iter()
            .position(|w| w.as_any().type_id() == std::any::TypeId::of::<T>())
        {
            self.states[i] = !self.states[i];
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        for (i, window) in self.windows.iter_mut().enumerate() {
            window.show(ctx, &mut self.states[i]);
        }
    }

    pub fn get_tool<T: 'static>(&self) -> Option<&T> {
        self.windows
            .iter()
            .find(|w| w.as_any().type_id() == std::any::TypeId::of::<T>())
            .and_then(|w| w.as_any().downcast_ref::<T>())
    }

    pub fn get_tool_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.windows
            .iter_mut()
            .find(|w| w.as_any().type_id() == std::any::TypeId::of::<T>())
            .and_then(|w| w.as_any_mut().downcast_mut::<T>())
    }
}

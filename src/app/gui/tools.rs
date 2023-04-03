mod plotting;
mod sampling;
mod scratch;
mod visual_debug;

use crate::{
    app::{gfx::GpuContext, gui::VgonioEvent},
    measure::{rtc::Ray, TrajectoryNode},
};
#[cfg(feature = "embree")]
use embree::Config;

use winit::event_loop::EventLoopProxy;

use crate::{
    measure::rtc::grid::{Grid, MultilevelGrid},
    msurf::MicroSurfaceMesh,
};
pub(crate) use plotting::Plotting;
pub(crate) use sampling::SamplingDebugger;
pub(crate) use scratch::Scratch;
pub(crate) use visual_debug::VisualDebugger;

use super::state::GuiRenderer;

pub(crate) fn trace_ray_grid_dbg(ray: Ray, max_bounces: u32, grid_rt: &MultilevelGrid) -> Vec<Ray> {
    log::debug!("trace_ray_grid_dbg: {:?}", ray);
    let mut rays = vec![];
    // grid_rt.trace_one_ray_dbg(ray, max_bounces, 0, None, &mut rays);
    // TODO: fix this
    log::debug!("traced rays: {:?}", rays);
    rays
}

#[cfg(feature = "embree")]
pub(crate) fn trace_ray_standard_dbg(
    ray: Ray,
    max_bounces: u32,
    surface: &MicroSurfaceMesh,
) -> Vec<Ray> {
    // let mut embree_rt = EmbreeRT::new(Config::default());
    // let scn_id = embree_rt.create_scene();
    // let mesh = embree_rt.create_triangle_mesh(surface);
    // embree_rt.attach_geometry(scn_id, mesh);
    // let mut nodes: Vec<TrajectoryNode> = vec![];
    // embree_rt.trace_one_ray_dbg(scn_id, ray, max_bounces, 0, None, &mut nodes);
    // nodes.into_iter().map(|n| n.ray).collect()
    todo!("trace_ray_standard_dbg")
}

pub trait Tool {
    fn name(&self) -> &'static str;

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
        event_loop: EventLoopProxy<VgonioEvent>,
        gpu: &GpuContext,
        gui: &mut GuiRenderer,
    ) -> Self {
        Self {
            windows: vec![
                Box::<Scratch>::default(),
                Box::new(VisualDebugger::new(event_loop.clone())),
                Box::<Plotting>::default(),
                Box::new(SamplingDebugger::new(
                    gpu,
                    gui,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    event_loop,
                )),
            ],
            states: vec![false, false, false, false],
        }
    }

    pub fn toggle(&mut self, name: &str) {
        if let Some(i) = self.windows.iter().position(|w| w.name() == name) {
            self.states[i] = !self.states[i];
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        for (i, window) in self.windows.iter_mut().enumerate() {
            window.show(ctx, &mut self.states[i]);
        }
    }

    pub fn get_tool<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.windows
            .iter_mut()
            .find(|w| w.name() == name)
            .and_then(|w| w.as_any_mut().downcast_mut::<T>())
    }
}

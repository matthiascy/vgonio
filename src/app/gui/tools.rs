mod debugging;
mod plotting;
mod sampling;
mod scratch;

use crate::{
    app::{gfx::GpuContext, gui::VgonioEvent},
    measure::rtc::Ray,
};
#[cfg(feature = "embree")]
use embree::Config;
use std::rc::Rc;

use winit::event_loop::EventLoopProxy;

use crate::measure::rtc::grid::MultilevelGrid;
pub(crate) use debugging::DebuggingInspector;
pub(crate) use plotting::PlottingInspector;
pub(crate) use sampling::SamplingInspector;
pub(crate) use scratch::Scratch;

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
        event_loop: EventLoopProxy<VgonioEvent>,
        gpu: &GpuContext,
        gui: &mut GuiRenderer,
    ) -> Self {
        Self {
            windows: vec![
                Box::<Scratch>::default(),
                Box::new(DebuggingInspector::new(event_loop.clone())),
                Box::new(PlottingInspector::new("Graph".to_string(), Rc::new(()))),
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

    pub fn get_tool<T: 'static>(&mut self) -> Option<&mut T> {
        self.windows
            .iter_mut()
            .find(|w| w.as_any().type_id() == std::any::TypeId::of::<T>())
            .and_then(|w| w.as_any_mut().downcast_mut::<T>())
    }
}

mod visual_debug;

use crate::{
    acq::{EmbreeRayTracing, GridRayTracing, Ray, TrajectoryNode},
    mesh::TriangleMesh,
};
use embree::Config;
pub(crate) use visual_debug::VisualDebugTool;

pub(crate) fn trace_ray_grid_dbg(ray: Ray, max_bounces: u32, grid_rt: &GridRayTracing) -> Vec<Ray> {
    log::debug!("trace_ray_grid_dbg: {:?}", ray);
    let mut rays = vec![];
    grid_rt.trace_one_ray_dbg(ray, max_bounces, 0, None, &mut rays);
    log::debug!("traced rays: {:?}", rays);
    rays
}

pub(crate) fn trace_ray_standard_dbg(
    ray: Ray,
    max_bounces: u32,
    surface: &TriangleMesh,
) -> Vec<Ray> {
    let mut embree_rt = EmbreeRayTracing::new(Config::default());
    let scn_id = embree_rt.create_scene();
    let mesh = embree_rt.create_triangle_mesh(surface);
    embree_rt.attach_geometry(scn_id, mesh);
    let mut nodes: Vec<TrajectoryNode> = vec![];
    embree_rt.trace_one_ray_dbg(scn_id, ray.into(), max_bounces, 0, None, &mut nodes);
    nodes.into_iter().map(|n| n.ray).collect()
}

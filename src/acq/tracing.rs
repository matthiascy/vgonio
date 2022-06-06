use crate::acq::embree_rt::EmbreeRayTracing;
use crate::acq::ray::Ray;
use crate::htfld::Heightfield;
use crate::mesh::TriangulationMethod;
use embree::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayTracingMethod {
    Standard,
    Grid,
    Hybrid,
}

pub fn trace_ray_grid(ray: Ray, surface: &Heightfield) {
    println!("grid");
}

pub fn trace_ray_standard(ray: Ray, surface: &Heightfield) {
    let mut embree_rt = EmbreeRayTracing::new(Config::default());
    let scene_id = embree_rt.create_scene();
    let triangulated_surface = surface.triangulate(TriangulationMethod::Regular);
    let surface_mesh = embree_rt.create_triangle_mesh(&triangulated_surface);
    let surface_id = embree_rt.attach_geometry(scene_id, surface_mesh);

    let mut coherent_ctx = embree::IntersectContext::coherent();
    let ray_hit = embree_rt.intersect(scene_id, ray.into_embree_ray(), &mut coherent_ctx);

    if ray_hit.hit.hit() {
        println!("hit");
    } else {
        println!("miss");
    }
}

pub fn trace_ray_hybrid(ray: Ray, surface: &Heightfield) {
    println!("hybrid");
}

use crate::acq::embree_rt::EmbreeRayTracing;
use crate::acq::ior::RefractiveIndex;
use crate::acq::ray::{scattering_air_conductor, Ray, RayTraceRecord, Scattering};
use crate::htfld::Heightfield;
use crate::mesh::{TriangleMesh, TriangulationMethod};
use embree::{Config, RayHit};
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayTracingMethod {
    Standard,
    Grid,
    Hybrid,
}

pub fn trace_ray_grid(ray: Ray, surface: &TriangleMesh) {
    println!("grid");
}

pub fn trace_ray_standard(ray: Ray, max_bounces: u32, surface: &TriangleMesh) -> Vec<embree::Ray> {
    let mut embree_rt = EmbreeRayTracing::new(Config::default());
    let scn_id = embree_rt.create_scene();
    let mesh = embree_rt.create_triangle_mesh(surface);
    let geom_id = embree_rt.attach_geometry(scn_id, mesh);
    let mut rays: Vec<embree::Ray> = vec![];

    embree_rt.trace_one_ray_dbg_auto_adjust(
        scn_id,
        ray.into_embree_ray(),
        max_bounces,
        0,
        None,
        &mut rays,
    );

    rays
}

pub fn trace_ray_hybrid(ray: Ray, surface: &TriangleMesh) {
    println!("hybrid");
}

/// Intermediate structure used to store the results of a ray intersection.
/// Used to rewind self intersections.
#[derive(Debug)]
pub struct IntersectRecord {
    /// The ray that hit the surface.
    pub ray: Ray,

    /// The surface that was hit.
    pub geom_id: u32,

    /// The primitive of the surface that was hit.
    pub prim_id: u32,

    /// The hit point.
    pub hit_point: Vec3,

    /// The normal at the hit point.
    pub normal: Vec3,

    /// How many times the ray has been rewound.
    pub nudged_times: u32,
}

const NUDGE_AMOUNT: f32 = f32::EPSILON * 10.0;

// rtcIntersect/rtcOccluded calls can be invoked from multiple threads.
// embree api calls to the same object are not thread safe in general.
pub fn trace_one_ray_embree(
    ray: Ray,
    scene: &mut embree::Scene,
    context: &mut embree::IntersectContext,
    ior_t: RefractiveIndex,
    max_bounces: u32,
    prev_isect: Option<IntersectRecord>,
    prev_record: Option<RayTraceRecord>,
) -> Option<RayTraceRecord> {
    // Either the ray hasn't reach the bounce limit or it hasn't hit the surface.
    if prev_record.is_some_and(|record| record.bounces < max_bounces) || prev_record.is_none() {
        log::debug!("- prev_record {:?}", prev_record);
        let mut ray_hit = RayHit::new(ray.into_embree_ray());
        scene.intersect(context, &mut ray_hit);

        // Does the hit something?
        if ray_hit.hit.hit() && (f32::EPSILON..f32::INFINITY).contains(&ray_hit.ray.tfar) {
            log::debug!("  - hit something");
            // It's not the first time that the ray has hit the surface.
            // Is it the first time that the ray hit some geometry?
            if let Some(prev_isect) = prev_isect {
                log::debug!("    - has previous intersection");
                // Is the ray repeat hitting the same primitive of the same geometry?
                if prev_isect.geom_id == ray_hit.hit.geomID
                    && prev_isect.prim_id == ray_hit.hit.primID
                {
                    log::debug!("      - is repeat intersection");
                    // Self intersection happens, recalculate the hit point.
                    let nudged_times = prev_isect.nudged_times + 1;
                    log::debug!("        - new nudged_times {}", nudged_times);
                    let amount = NUDGE_AMOUNT * (nudged_times * nudged_times) as f32 * 0.5;
                    let hit_point =
                        nudge_hit_point(prev_isect.hit_point, prev_isect.normal, amount);
                    log::debug!("        - new hit_point {:?}", hit_point);
                    // Update the intersection record.
                    let curr_isect = IntersectRecord {
                        hit_point,
                        nudged_times,
                        ..prev_isect
                    };
                    log::debug!("        - new curr_isect {:?}", curr_isect);
                    // Recalculate the reflected ray.
                    if let Some(Scattering { reflected, .. }) = scattering_air_conductor(
                        prev_isect.ray,
                        hit_point,
                        prev_isect.normal,
                        ior_t.eta,
                        ior_t.k,
                    ) {
                        log::debug!("        - new reflected ray {:?}", reflected);
                        if reflected.e >= 0.0 {
                            // Update ray tracing information.
                            let curr_record = RayTraceRecord {
                                initial: prev_record.as_ref().unwrap().initial,
                                current: reflected,
                                bounces: prev_record.as_ref().unwrap().bounces,
                            };
                            trace_one_ray_embree(
                                reflected,
                                scene,
                                context,
                                ior_t,
                                max_bounces,
                                Some(curr_isect),
                                Some(curr_record),
                            )
                        } else {
                            // The ray is absorbed by the surface.
                            prev_record
                        }
                    } else {
                        // The ray is absorbed by the surface.
                        prev_record
                    }
                } else {
                    // Not hitting the same primitive of the same geometry, trace the reflected ray.
                    let normal = Vec3::new(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z);
                    let hit_point = nudge_hit_point(
                        compute_hit_point(scene, &ray_hit.hit),
                        normal,
                        NUDGE_AMOUNT,
                    );
                    // Record the current intersection information.
                    let curr_isect = IntersectRecord {
                        ray,
                        geom_id: ray_hit.hit.geomID,
                        prim_id: ray_hit.hit.primID,
                        hit_point,
                        normal,
                        nudged_times: 1, // Initially the ray has been nudged once.
                    };
                    // Compute the new reflected ray.
                    if let Some(Scattering { reflected, .. }) =
                        scattering_air_conductor(ray, hit_point, normal, ior_t.eta, ior_t.k)
                    {
                        // The ray is not absorbed by the surface.
                        if reflected.e >= 0.0 {
                            // Record the current ray tracing information and trace the reflected
                            // ray.
                            let curr_record = RayTraceRecord {
                                initial: prev_record.as_ref().unwrap().initial,
                                current: ray,
                                bounces: prev_record.as_ref().unwrap().bounces + 1,
                            };
                            trace_one_ray_embree(
                                reflected,
                                scene,
                                context,
                                ior_t,
                                max_bounces,
                                Some(curr_isect),
                                Some(curr_record),
                            )
                        } else {
                            // The ray is absorbed by the surface.
                            prev_record
                        }
                    } else {
                        // The ray is absorbed by the surface.
                        prev_record
                    }
                }
            } else {
                log::debug!("    - has no previous intersection");
                // Not previously hit, so we can continue tracing reflected ray if the ray is
                // not absorbed.
                let normal = Vec3::new(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z);
                let hit_point =
                    nudge_hit_point(compute_hit_point(scene, &ray_hit.hit), normal, NUDGE_AMOUNT);
                // Record the current intersection information.
                let curr_isect = IntersectRecord {
                    ray,
                    geom_id: ray_hit.hit.geomID,
                    prim_id: ray_hit.hit.primID,
                    hit_point,
                    normal,
                    nudged_times: 1, // Initially the ray has been nudged once.
                };
                log::debug!("      - new curr_isect {:?}", curr_isect);
                // Compute the new reflected ray.
                if let Some(Scattering { reflected, .. }) =
                    scattering_air_conductor(ray, hit_point, normal, ior_t.eta, ior_t.k)
                {
                    log::debug!("      - new reflected ray {:?}", reflected);
                    // The ray is not absorbed by the surface.
                    if reflected.e >= 0.0 {
                        // Record the current ray tracing information and trace the reflected ray.
                        let curr_record = RayTraceRecord {
                            initial: ray,
                            current: ray,
                            bounces: 1,
                        };
                        trace_one_ray_embree(
                            reflected,
                            scene,
                            context,
                            ior_t,
                            max_bounces,
                            Some(curr_isect),
                            Some(curr_record),
                        )
                    } else {
                        // The ray is absorbed by the surface.
                        prev_record
                    }
                } else {
                    // The ray is absorbed by the surface.
                    prev_record
                }
            }
        } else {
            // The ray didn't hit the surface.
            prev_record
        }
    } else {
        // The ray has reached the bounce limit.
        prev_record
    }
}

/// Nudge the hit point along the normal to avoid self-intersection.
fn nudge_hit_point(hit_point: Vec3, normal: Vec3, amount: f32) -> Vec3 {
    hit_point + normal * amount
}

/// Compute intersection point.
fn compute_hit_point(scene: &embree::Scene, record: &embree::Hit) -> Vec3 {
    let geom = scene.geometry(record.geomID).unwrap().handle();
    let prim_id = record.primID as isize;
    let points = unsafe {
        let vertices: *const f32 =
            embree::sys::rtcGetGeometryBufferData(geom, embree::BufferType::VERTEX, 0) as _;
        let indices: *const u32 =
            embree::sys::rtcGetGeometryBufferData(geom, embree::BufferType::INDEX, 0) as _;

        let mut points = [Vec3::ZERO; 3];
        for (i, p) in points.iter_mut().enumerate() {
            let idx = *indices.offset(prim_id * 3 + i as isize) as isize;
            p.x = *vertices.offset(idx * 4);
            p.y = *vertices.offset(idx * 4 + 1);
            p.z = *vertices.offset(idx * 4 + 2);
        }

        points
    };
    log::debug!(
        "geom_id: {}, prim_id: {}, u: {}, v: {}, p0: {}, p1: {}, p2: {}",
        record.geomID,
        record.primID,
        record.u,
        record.v,
        points[0],
        points[1],
        points[2]
    );
    (1.0 - record.u - record.v) * points[0] + record.u * points[1] + record.v * points[2]
}

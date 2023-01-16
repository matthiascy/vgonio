// rtcIntersect/rtcOccluded calls can be invoked from multiple threads.
// embree api calls to the same object are not thread safe in general.

use crate::{
    acq::{fresnel, ior::Ior, scattering::reflect, Ray, RtcRecord, TrajectoryNode},
    mesh,
};
use embree::{
    Config, Device, Geometry, Hit, IntersectContext, RayHit, RayHitN, RayN, Scene, SceneFlags,
    TriangleMesh,
};
use glam::Vec3;
use std::sync::Arc;

/// Struct managing the ray/geometry intersection with the help of embree.
pub struct EmbreeRayTracing {
    device: Arc<Device>,
    scenes: Vec<Arc<Scene>>,
}

impl Default for EmbreeRayTracing {
    fn default() -> Self {
        Self {
            device: Device::with_config(Config::default()),
            scenes: Vec::new(),
        }
    }
}

impl EmbreeRayTracing {
    /// Amount to add to the ray origin to avoid self-intersection.
    pub const NUDGE_AMOUNT: f32 = f32::EPSILON * 10.0;

    /// Initialise embree framework.
    pub fn new(config: Config) -> Self {
        Self {
            device: Device::with_config(config),
            scenes: vec![],
        }
    }

    /// Creates a scene for ray intersection tests.
    pub fn create_scene(&mut self) -> usize {
        self.scenes.push(Scene::new(self.device.clone()));
        self.scenes.len() - 1
    }

    /// Returns the mutable reference of a scene according to it's id.
    pub fn scene_mut(&mut self, id: usize) -> &mut Scene {
        Arc::get_mut(&mut self.scenes[id]).unwrap()
    }

    /// Returns the reference of a scene according to it's id.
    pub fn scene(&self, id: usize) -> &Scene { &self.scenes[id] }

    /// Uploads a triangle mesh to embree.
    pub fn create_triangle_mesh(&self, mesh: &mesh::MicroSurfaceTriMesh) -> Arc<TriangleMesh> {
        let mut embree_mesh =
            TriangleMesh::unanimated(self.device.clone(), mesh.num_facets, mesh.num_verts);
        {
            let mesh_ref_mut = Arc::get_mut(&mut embree_mesh).unwrap();
            {
                let mut verts_buffer = mesh_ref_mut.vertex_buffer.map();
                let mut indxs_buffer = mesh_ref_mut.index_buffer.map();
                for (i, vert) in mesh.verts.iter().enumerate() {
                    verts_buffer[i] = [vert.x, vert.y, vert.z, 1.0];
                }
                // TODO: replace with slice::as_chunks when it's stable.
                (0..mesh.num_facets).for_each(|tri| {
                    indxs_buffer[tri] = [
                        mesh.facets[tri * 3],
                        mesh.facets[tri * 3 + 1],
                        mesh.facets[tri * 3 + 2],
                    ];
                })
            }
            mesh_ref_mut.commit()
        }
        embree_mesh
    }

    /// Attach a geometry onto a embree scene.
    pub fn attach_geometry(&mut self, scene_id: usize, geometry: Arc<dyn Geometry>) -> u32 {
        let scene_mut = self.scene_mut(scene_id);
        let id = scene_mut.attach_geometry(geometry);
        scene_mut.commit();
        id
    }

    /// Tests if a stream of rays intersect with a scene.
    pub fn intersect_stream_soa(
        &self,
        scene_id: usize,
        stream: RayN,
        context: &mut IntersectContext,
    ) -> RayHitN {
        let mut ray_hit = RayHitN::new(stream);
        let scene = self.scene(scene_id);
        scene.intersect_stream_soa(context, &mut ray_hit);
        ray_hit
    }

    /// Tests if a ray intersects with a scene.
    pub fn intersect(
        &mut self,
        scene_id: usize,
        ray: embree::Ray,
        context: &mut IntersectContext,
    ) -> RayHit {
        let mut ray_hit = RayHit::new(ray);
        let scene = self.scene_mut(scene_id);
        scene.set_flags(scene.flags() | SceneFlags::ROBUST);
        scene.intersect(context, &mut ray_hit);
        ray_hit
    }

    /// Traces one ray, only records its path.
    pub fn trace_one_ray_dbg(
        &mut self,
        scene_id: usize,
        ray: Ray,
        max_bounces: u32,
        curr_bounces: u32,
        prev: Option<EmbreeIsectRecord>,
        trajectory: &mut Vec<TrajectoryNode>,
    ) {
        let mut context = IntersectContext::incoherent();
        let scene = self.scene_mut(scene_id);

        trace_one_ray_inner(
            scene,
            &mut context,
            ray,
            max_bounces,
            curr_bounces,
            prev,
            trajectory,
        );
    }

    /// Traces one ray, records its path and compute energy.
    pub fn trace_one_ray(
        &mut self,
        scene_id: usize,
        ray: Ray,
        max_bounces: u32,
        ior_t: &[Ior],
    ) -> Option<RtcRecord> {
        let mut context = IntersectContext::incoherent();
        let scene = self.scene_mut(scene_id);
        let mut trajectory = vec![]; // including the ray itself

        // First, trace the ray to get its trajectory.
        trace_one_ray_inner(
            scene,
            &mut context,
            ray,
            max_bounces,
            0,
            None,
            &mut trajectory,
        );

        // Then, according to the trajectory, compute the energy.
        if trajectory.len() < 2 {
            None
        } else {
            log::debug!("final trajectory  > 1: {:?}", trajectory);
            let mut energy_each_bounce = vec![vec![1.0; ior_t.len()]; trajectory.len()];
            log::debug!("energy_each_bounce: {:?}", energy_each_bounce);
            for i in 0..trajectory.len() - 1 {
                let node = &trajectory[i];
                let reflectance = fresnel::reflectance_air_conductor_spectrum(node.cos, ior_t);
                log::debug!("calculated reflectance: {:?}", reflectance);
                for (j, energy) in reflectance.iter().enumerate().take(ior_t.len()) {
                    energy_each_bounce[i + 1][j] = energy_each_bounce[i][j] * energy;
                }
            }
            Some(RtcRecord {
                trajectory,
                energy_each_bounce,
            })
        }
    }
}

fn trace_one_ray_inner(
    scn: &mut Scene,
    ctx: &mut IntersectContext,
    ray: Ray,
    max_bounces: u32,
    curr_bounces: u32,
    prev: Option<EmbreeIsectRecord>,
    trajectory: &mut Vec<TrajectoryNode>,
) {
    log::debug!("---- current bounce {} ----", curr_bounces);
    trajectory.push(TrajectoryNode { ray, cos: 0.0 });
    log::debug!("push ray: {:?} | len: {:?}", ray, trajectory.len());

    if curr_bounces >= max_bounces {
        log::debug!("  > reached max bounces");
        return;
    }

    let mut ray_hit = RayHit::new(ray.into());
    scn.intersect(ctx, &mut ray_hit);

    // Checks if the ray hit something.
    if ray_hit.hit.hit() && (f32::EPSILON..f32::INFINITY).contains(&ray_hit.ray.tfar) {
        log::debug!("  > YES[HIT]");
        log::debug!(
            "    geom {} - prim {}",
            ray_hit.hit.geomID,
            ray_hit.hit.primID
        );

        // Check with the previous hit record if the same primitive has been hit.
        log::debug!("    > has prev? {}", prev.is_some());
        if let Some(prev) = prev {
            log::debug!(
                "      -- geom_id - prev: {}, now: {}, prim_id - prev: {}, now: {}",
                prev.geom_id,
                ray_hit.hit.geomID,
                prev.prim_id,
                ray_hit.hit.primID
            );
            if prev.geom_id == ray_hit.hit.geomID && prev.prim_id == ray_hit.hit.primID {
                log::debug!("       > THE SAME PRIMITIVE");
                trajectory.pop();
                log::debug!(
                    "        -- nudging | pop out last | after len: {:?}",
                    trajectory.len()
                );
                // nudge more
                let nudged_times = prev.nudged_times + 1;
                let amount = nudged_times as f32 * EmbreeRayTracing::NUDGE_AMOUNT;
                let new_hit_point = prev.hit_point + prev.normal * amount; // update intersection record
                let new_isect = EmbreeIsectRecord {
                    hit_point: new_hit_point,
                    nudged_times,
                    ..prev
                };
                let new_ray = Ray::new(new_hit_point, prev.dir_r);
                return trace_one_ray_inner(
                    scn,
                    ctx,
                    new_ray,
                    max_bounces,
                    curr_bounces,
                    Some(new_isect),
                    trajectory,
                );
            }
        }

        // Not hitting the same primitive.
        let normal = Vec3::new(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z).normalize();
        // let dir = Vec3::new(ray.dir_x, ray.dir_y, ray.dir_z).normalize();
        let hit_point =
            compute_hit_point(scn, &ray_hit.hit) + normal * EmbreeRayTracing::NUDGE_AMOUNT;
        let reflected_dir = reflect(ray.d, normal).normalize();
        let new_ray = Ray::new(hit_point, reflected_dir);
        trajectory.last_mut().unwrap().cos = ray.d.dot(normal).abs();
        let isect = EmbreeIsectRecord {
            ray,
            dir_r: reflected_dir,
            geom_id: ray_hit.hit.geomID,
            prim_id: ray_hit.hit.primID,
            normal,
            hit_point,
            nudged_times: 1,
        };
        trace_one_ray_inner(
            scn,
            ctx,
            new_ray,
            max_bounces,
            curr_bounces + 1,
            Some(isect),
            trajectory,
        );
    } else {
        // Nothing hit.
        log::debug!("  > NO[HIT] Quitting [{}]...", curr_bounces);
    }
}

#[derive(Debug)]
pub struct EmbreeIsectRecord {
    // Incident ray
    pub ray: Ray,
    // Reflected dir
    pub dir_r: Vec3,
    pub geom_id: u32,
    pub prim_id: u32,
    pub normal: Vec3,
    // Cached hit point without any nudging.
    pub hit_point: Vec3,
    pub nudged_times: u32,
}

/// Compute intersection point.
pub fn compute_hit_point(scene: &Scene, record: &Hit) -> Vec3 {
    let geom = scene.geometry(record.geomID).unwrap().handle();
    let prim_id = record.primID as isize;
    let points = unsafe {
        let vertices: *const f32 =
            embree::sys::rtcGetGeometryBufferData(geom, embree::BufferType::VERTEX, 0) as _;
        let indices: *const u32 =
            embree::sys::rtcGetGeometryBufferData(geom, embree::BufferType::INDEX, 0) as _;

        let mut points = [Vec3::ZERO; 3];
        log::debug!(
            "    prim: ({}) {}-{}-{}",
            prim_id,
            *indices.offset(prim_id * 3),
            *indices.offset(prim_id * 3 + 1),
            *indices.offset(prim_id * 3 + 2)
        );
        for (i, p) in points.iter_mut().enumerate() {
            let idx = *indices.offset(prim_id * 3 + i as isize) as isize;
            p.x = *vertices.offset(idx * 4);
            p.y = *vertices.offset(idx * 4 + 1);
            p.z = *vertices.offset(idx * 4 + 2);
        }
        points
    };
    log::debug!(
        "    calc hit point -- geom_id: {}, prim_id: {}, u: {}, v: {}\n          p0: {}, p1: {}, \
         p2: {}",
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

use crate::acq::ray::reflect;
use crate::acq::tracing::IntersectRecord;
use crate::mesh;
use embree::{
    Config, Device, Geometry, Hit, IntersectContext, RayHit, RayHitN, RayN, Scene, SceneFlags,
    TriangleMesh,
};
use glam::Vec3;
use std::sync::Arc;

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
    pub const NUDGE_AMOUNT: f32 = f32::EPSILON * 10.0;

    pub fn new(config: Config) -> Self {
        Self {
            device: Device::with_config(config),
            scenes: vec![],
        }
    }

    pub fn create_scene(&mut self) -> usize {
        self.scenes.push(Scene::new(self.device.clone()));
        self.scenes.len() - 1
    }

    pub fn scene_mut(&mut self, id: usize) -> &mut Scene {
        Arc::get_mut(&mut self.scenes[id]).unwrap()
    }

    pub fn scene(&self, id: usize) -> &Scene {
        &self.scenes[id]
    }

    pub fn create_triangle_mesh(&self, mesh: &mesh::TriangleMesh) -> Arc<TriangleMesh> {
        let mut embree_mesh =
            TriangleMesh::unanimated(self.device.clone(), mesh.num_tris, mesh.num_verts);
        {
            let mesh_ref_mut = Arc::get_mut(&mut embree_mesh).unwrap();
            {
                let mut verts_buffer = mesh_ref_mut.vertex_buffer.map();
                let mut indxs_buffer = mesh_ref_mut.index_buffer.map();
                for (i, vert) in mesh.verts.iter().enumerate() {
                    verts_buffer[i] = [vert.x, vert.y, vert.z, 1.0];
                }
                // TODO: replace with slice::as_chunks when it's stable.
                (0..mesh.num_tris).for_each(|tri| {
                    indxs_buffer[tri] = [
                        mesh.faces[tri * 3],
                        mesh.faces[tri * 3 + 1],
                        mesh.faces[tri * 3 + 2],
                    ];
                })
            }
            mesh_ref_mut.commit()
        }
        embree_mesh
    }

    pub fn attach_geometry(&mut self, scene_id: usize, geometry: Arc<dyn Geometry>) -> u32 {
        let scene_mut = self.scene_mut(scene_id);
        let id = scene_mut.attach_geometry(geometry);
        scene_mut.commit();
        id
    }

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

    pub fn trace_one_ray_dbg(
        &mut self,
        scene_id: usize,
        ray: embree::Ray,
        max_bounces: u32,
        curr_bounces: u32,
        prev: Option<EmbreeIsectRecord>,
        output: &mut Vec<embree::Ray>,
    ) {
        let mut context = IntersectContext::incoherent();
        let scene = self.scene_mut(scene_id);

        trace_one_ray_dbg_inner(
            scene,
            &mut context,
            ray,
            max_bounces,
            curr_bounces,
            prev,
            output,
        );

        fn trace_one_ray_dbg_inner(
            scn: &mut Scene,
            ctx: &mut IntersectContext,
            ray: embree::Ray,
            max_bounces: u32,
            curr_bounces: u32,
            prev: Option<EmbreeIsectRecord>,
            output: &mut Vec<embree::Ray>,
        ) {
            log::debug!("[{}]", curr_bounces);
            println!("output: {:?}", output);
            output.push(ray);
            println!("push ray: {:?} | out: {:?}", ray, output);

            if curr_bounces >= max_bounces {
                log::debug!("  > reached max bounces");
                return;
            }

            let mut ray_hit = RayHit::new(ray);
            scn.intersect(ctx, &mut ray_hit);

            if ray_hit.hit.hit() && (f32::EPSILON..f32::INFINITY).contains(&ray_hit.ray.tfar) {
                log::debug!("  > YES[HIT]");
                log::debug!(
                    "    geom {} - prim {}",
                    ray_hit.hit.geomID,
                    ray_hit.hit.primID
                );
                if let Some(prev) = prev {
                    // Check with the previous hit record if the same primitive has been hit.
                    if prev.geom_id == ray_hit.hit.geomID && prev.prim_id == ray_hit.hit.primID {
                        log::debug!("    > THE SAME PRIMITIVE");
                        output.pop();
                        log::debug!("      -- nudging | pop out last ray | out: {:?}", output);
                        // nudge more
                        let nudged_times = prev.nudged_times + 1;
                        let amount = nudged_times as f32 * EmbreeRayTracing::NUDGE_AMOUNT;
                        let new_hit_point = prev.hit_point + prev.normal * amount; // update intersection record
                        let new_isect = EmbreeIsectRecord {
                            hit_point: new_hit_point,
                            nudged_times,
                            ..prev
                        };
                        let new_ray = embree::Ray::new(new_hit_point.into(), prev.dir_r.into());
                        trace_one_ray_dbg_inner(
                            scn,
                            ctx,
                            new_ray,
                            max_bounces,
                            curr_bounces + 1,
                            Some(new_isect),
                            output,
                        );
                    }
                }

                // Previous hit record is not available.
                let normal =
                    Vec3::new(ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z).normalize();
                let dir = Vec3::new(ray.dir_x, ray.dir_y, ray.dir_z).normalize();
                let hit_point =
                    compute_hit_point(scn, &ray_hit.hit) + normal * EmbreeRayTracing::NUDGE_AMOUNT;
                let reflected_dir = reflect(dir, normal).normalize();
                let new_ray = embree::Ray::new(hit_point.into(), reflected_dir.into());
                let isect = EmbreeIsectRecord {
                    ray_i: ray,
                    dir_r: reflected_dir,
                    geom_id: ray_hit.hit.geomID,
                    prim_id: ray_hit.hit.primID,
                    normal,
                    hit_point,
                    nudged_times: 1,
                };
                trace_one_ray_dbg_inner(
                    scn,
                    ctx,
                    new_ray,
                    max_bounces,
                    curr_bounces + 1,
                    Some(isect),
                    output,
                );
            } else {
                log::debug!("  > NO[HIT] Quitting [{}]...", curr_bounces);
            }
        }
    }
}

#[derive(Debug)]
pub struct EmbreeIsectRecord {
    // Incident ray
    ray_i: embree::Ray,
    // Reflected dir
    dir_r: Vec3,
    geom_id: u32,
    prim_id: u32,
    normal: Vec3,
    // Cached hit point without any nudging.
    hit_point: Vec3,
    nudged_times: u32,
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
        "    calc hit point -- geom_id: {}, prim_id: {}, u: {}, v: {}\
        \n          p0: {}, p1: {}, p2: {}",
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

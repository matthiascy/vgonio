use crate::mesh;
use embree::{
    Config, Device, Geometry, IntersectContext, RayHit, RayHitN, RayN, Scene, TriangleMesh,
};
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
        &self,
        scene_id: usize,
        ray: embree::Ray,
        context: &mut IntersectContext,
    ) -> RayHit {
        let mut ray_hit = RayHit::new(ray);
        let scene = self.scene(scene_id);
        scene.intersect(context, &mut ray_hit);
        ray_hit
    }
}

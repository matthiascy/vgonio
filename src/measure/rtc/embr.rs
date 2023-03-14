// rtcIntersect/rtcOccluded calls can be invoked from multiple threads.
// embree api calls to the same object are not thread safe in general.

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    measure::{
        bsdf::SpectrumSampler,
        measurement::{BsdfMeasurement, Radius},
        scattering::reflect,
        Collector, Emitter, Ray, RtcRecord,
    },
    msurf::{MicroSurface, MicroSurfaceMesh},
    optics,
    optics::RefractiveIndex,
    units::{metres, mm, um},
};
use embree::{
    BufferUsage, Config, Device, Format, Geometry, GeometryKind, Hit, HitN, IntersectContext,
    IntersectContextExt, IntersectContextFlags, Ray as EmbreeRay, RayHit, RayHitN, RayHitNp, RayN,
    RayNp, Scene, SceneFlags, SoAHit, SoARay, TriangleMesh, ValidMask, ValidityN, INVALID_ID,
};
use glam::{Vec3, Vec3A};
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};

impl MicroSurfaceMesh {
    /// Constructs an embree geometry from the `MicroSurfaceMesh`.
    pub fn as_embree_geometry<'a, 'b, 'g>(&'a self, device: &'b Device) -> Geometry<'g> {
        let mut geom = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
        geom.set_new_buffer(BufferUsage::VERTEX, 0, Format::FLOAT3, 16, self.num_verts)
            .unwrap()
            .view_mut::<[f32; 4]>()
            .unwrap()
            .iter_mut()
            .zip(self.verts.iter())
            .for_each(|(vert, pos)| {
                vert[0] = pos.x;
                vert[1] = pos.y;
                vert[2] = pos.z;
                vert[3] = 1.0;
            });
        geom.set_new_buffer(BufferUsage::INDEX, 0, Format::UINT3, 12, self.num_facets)
            .unwrap()
            .view_mut::<u32>()
            .unwrap()
            .copy_from_slice(&self.facets);
        geom.commit();
        geom
    }
}

#[derive(Debug, Clone, Copy)]
struct HitData {
    pub geom_id: u32,
    pub prim_id: u32,
    pub normal: Vec3A,
}

#[derive(Debug, Clone, Copy)]
struct TrajectoryNode {
    pub org: Vec3A,
    pub dir: Vec3A,
}

#[derive(Debug, Clone)]
pub struct RayStreamData<'a> {
    /// The micro-surface geometry.
    msurf: Arc<Geometry<'a>>,
    /// Final energy of each ray.
    energy: Vec<f32>,
    /// Number of bounces for each ray.
    bounce: Vec<u32>,
    /// The last hit for each ray.
    last_hit: Vec<HitData>,
    /// The trajectory of each ray.
    trajectory: Vec<Vec<TrajectoryNode>>,
}

unsafe impl Send for RayStreamData<'_> {}
unsafe impl Sync for RayStreamData<'_> {}

type QueryContext<'a, 'b> = IntersectContextExt<&'b mut RayStreamData<'a>>;

fn intersect_filter_stream<'a, 'b>(
    rays: RayN<'a>,
    hits: HitN<'a>,
    mut valid: ValidityN,
    ctx: &'b mut QueryContext,
    _user_data: Option<&mut ()>,
) {
    let n = rays.len();
    for i in 0..n {
        // Ignore invalid rays.
        if valid[i] != ValidMask::Valid {
            continue;
        }

        // if the ray hit something
        if hits.is_valid(i) {
            let prim_id = hits.prim_id(i);
            let ray_id = rays.id(i);
            log::trace!("ray {} -- hit: {}", ray_id, prim_id);
            let last_hit = ctx.ext.last_hit[ray_id as usize];

            if prim_id != INVALID_ID && prim_id == last_hit.prim_id {
                log::trace!("nudging ray origin");
                // if the ray hit the same primitive as the previous bounce,
                // nudging the ray origin slightly along normal direction of
                // the last hit point and re-tracing the ray.
                valid[i] = ValidMask::Valid as i32;
                let traj_node = ctx.ext.trajectory[ray_id as usize].last_mut().unwrap();
                traj_node.org = traj_node.org + last_hit.normal * 1e-4;
                continue;
            } else {
                // calculate the intersection point using the u,v coordinates
                let [u, v] = hits.uv(i);
                let [i0, i1, i2] = ctx
                    .ext
                    .msurf
                    .get_buffer(BufferUsage::INDEX, 0)
                    .unwrap()
                    .view::<[u32; 3]>()
                    .unwrap()[prim_id as usize];
                let vertices = ctx
                    .ext
                    .msurf
                    .get_buffer(BufferUsage::VERTEX, 0)
                    .unwrap()
                    .view::<[f32; 4]>()
                    .unwrap();
                let v0 = {
                    let v = vertices[i0 as usize];
                    Vec3A::new(v[0], v[1], v[2])
                };
                let v1 = {
                    let v = vertices[i1 as usize];
                    Vec3A::new(v[0], v[1], v[2])
                };
                let v2 = {
                    let v = vertices[i2 as usize];
                    Vec3A::new(v[0], v[1], v[2])
                };
                let point = v0 * (1.0 - u - v) + v1 * u + v2 * v;
                // spawn a new ray from the intersection point
                let new_dir = reflect(rays.unit_dir(i).into(), hits.unit_normal(i).into());
                ctx.ext.bounce[ray_id as usize] += 1;
                ctx.ext.trajectory[ray_id as usize].push(TrajectoryNode {
                    org: point,
                    dir: new_dir,
                });
            }
        } else {
            // if the ray didn't hit anything, mark it as invalid
            log::trace!("ray {} missed", rays.id(i));
            valid[i] = ValidMask::Invalid as i32;
        }
    }
}

const MAX_RAY_STREAM_SIZE: usize = 1024;

pub fn measure_bsdf(
    desc: &BsdfMeasurement,
    surf: &MicroSurface,
    mesh: &MicroSurfaceMesh,
    cache: &Cache,
) {
    let mut collector: Collector = desc.collector.clone();
    let mut emitter: Emitter = desc.emitter.clone();
    collector.init();
    emitter.init();

    let mut device = Device::with_config(Config::default()).unwrap();
    let mut scene = device.create_scene().unwrap();
    scene.set_flags(SceneFlags::ROBUST);

    // Update emitter's radius to match the surface's dimensions.
    if emitter.radius().is_auto() {
        emitter.set_radius(Radius::Auto(
            um!(mesh.extent.max_edge() * 2.5).in_millimetres(),
        ));
    }

    log::debug!("mesh extent: {:?}", mesh.extent);
    log::debug!("emitter radius: {}", emitter.radius());

    // Upload the surface's mesh to the Embree scene.
    let mut geometry = mesh.as_embree_geometry(&mut device);
    geometry.set_intersect_filter_function(intersect_filter_stream);
    geometry.commit();

    scene.attach_geometry(&geometry);
    scene.commit();

    let spectrum = SpectrumSampler::from(emitter.spectrum).samples();
    log::debug!("spectrum samples: {:?}", spectrum);

    // Load the surface's reflectance data.
    let ior_t = cache
        .iors
        .ior_of_spectrum(desc.transmitted_medium, &spectrum)
        .expect("transmitted medium IOR not found");

    let total_num_rays = emitter.num_rays;

    // Iterate over every incident direction.
    for pos in emitter.meas_points() {
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Emit rays from {}° {}°",
            pos.zenith.in_degrees().value(),
            pos.azimuth.in_degrees().value()
        );

        let t = Instant::now();
        let emitted_rays = emitter.emit_rays(pos);
        let num_emitted_rays = emitted_rays.len();
        let elapsed = t.elapsed();

        let max_bounces = emitter.max_bounces;

        log::debug!(
            "emitted {} rays with direction {} from position {}° {}° in {:?} secs.",
            num_emitted_rays,
            emitted_rays[0].d,
            pos.zenith.in_degrees().value(),
            pos.azimuth.in_degrees().value(),
            elapsed.as_secs_f64(),
        );

        let arc_geom = Arc::new(geometry.clone());
        let n_streams = (num_emitted_rays + MAX_RAY_STREAM_SIZE - 1) / MAX_RAY_STREAM_SIZE;

        // In case the number of rays is less than one stream, we need to
        // adjust the stream size to match the number of rays.
        let stream_size = if n_streams == 1 {
            num_emitted_rays
        } else {
            MAX_RAY_STREAM_SIZE
        };

        let mut stream_data = vec![
            RayStreamData {
                msurf: arc_geom.clone(),
                energy: vec![0.0; stream_size],
                bounce: vec![0; stream_size],
                last_hit: vec![
                    HitData {
                        geom_id: INVALID_ID,
                        prim_id: INVALID_ID,
                        normal: Vec3A::ZERO,
                    };
                    stream_size
                ],
                trajectory: vec![Vec::with_capacity(max_bounces as usize); stream_size],
            };
            n_streams
        ];

        println!(
            "        {BRIGHT_YELLOW}>{RESET} Trace {} rays ({} streams)",
            emitted_rays.len(),
            n_streams
        );

        emitted_rays
            .chunks(MAX_RAY_STREAM_SIZE)
            .zip(stream_data.iter_mut())
            // rays.par_chunks_exact(MAX_RAY_STREAM_SIZE)
            //     .zip(stream_data.par_iter_mut())
            .enumerate()
            .for_each(|(i, (rays, data))| {
                log::trace!("stream {} of {}", i, n_streams);
                // Populate embree ray stream with generated rays.
                let chunk_size = rays.len();
                let mut ray_hit_n = RayHitNp::new(RayNp::new(chunk_size));
                for (i, mut ray) in ray_hit_n.ray.iter_mut().enumerate() {
                    ray.set_org(rays[i].o.into());
                    ray.set_dir(rays[i].d.into());
                    ray.set_id(i as u32);
                    ray.set_tnear(0.0);
                    ray.set_tfar(f32::INFINITY);
                }

                for i in 0..chunk_size {
                    data.trajectory[i].push(TrajectoryNode {
                        org: rays[i].o.into(),
                        dir: rays[i].d.into(),
                    });
                }

                // Trace primary rays with coherent context
                let mut ctx = QueryContext {
                    ctx: IntersectContext::coherent(),
                    ext: data,
                };

                let mut validities = vec![ValidMask::Valid; chunk_size];
                let mut bounces = 0;
                let mut active_rays = validities.len();
                while bounces < max_bounces && active_rays > 0 {
                    if bounces != 0 {
                        ctx.ctx.flags = IntersectContextFlags::INCOHERENT;
                    }

                    log::trace!(
                        "------------ bounce {}, active rays {}\n {:?} | {:?}",
                        bounces,
                        active_rays,
                        ctx.ext.trajectory,
                        ctx.ext.bounce
                    );
                    log::trace!("validities: {:?}", validities);

                    scene.intersect_stream_soa(&mut ctx, &mut ray_hit_n);

                    for i in 0..chunk_size {
                        if validities[i] == ValidMask::Invalid {
                            continue;
                        }

                        // Terminate ray if it didn't hit anything.
                        if !ray_hit_n.hit.is_valid(i) {
                            validities[i] = ValidMask::Invalid;
                            active_rays -= 1;
                            continue;
                        }

                        // Update the ray with the hit information.
                        let next_traj = ctx.ext.trajectory[i].last().unwrap();
                        ray_hit_n.ray.set_org(i, next_traj.org.into());
                        ray_hit_n.ray.set_dir(i, next_traj.dir.into());
                        ray_hit_n.ray.set_tnear(i, 0.00001);
                        ray_hit_n.ray.set_tfar(i, f32::INFINITY);
                        // Reset the hit information.
                        ray_hit_n.hit.set_geom_id(i, INVALID_ID);
                        ray_hit_n.hit.set_prim_id(i, INVALID_ID);
                    }

                    bounces += 1;
                }

                log::trace!(
                    "------------ result {}, active rays {}\n {:?} | {:?}\n{:?}",
                    bounces,
                    active_rays,
                    data.trajectory,
                    data.bounce,
                    validities
                );
            });
    }
}

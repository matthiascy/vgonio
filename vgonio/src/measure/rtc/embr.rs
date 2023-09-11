//! Embree ray tracing.

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    measure::{
        bsdf::{BsdfMeasurementDataPoint, MeasuredBsdfData},
        collector::{BounceAndEnergy, CollectorPatches},
        emitter::EmitterSamples,
        measurement::BsdfMeasurementParams,
        rtc::{LastHit, RayTrajectory, RayTrajectoryNode, MAX_RAY_STREAM_SIZE},
        Emitter,
    },
    optics::fresnel,
};
use embree::{
    BufferUsage, Config, Device, Geometry, HitN, IntersectContext, IntersectContextExt,
    IntersectContextFlags, RayHitNp, RayN, RayNp, Scene, SceneFlags, SoAHit, SoARay, ValidMask,
    ValidityN, INVALID_ID,
};
use rayon::prelude::*;
use std::sync::Arc;
#[cfg(all(debug_assertions, feature = "verbose_debug"))]
use std::time::Instant;
use vgcore::math::{Sph3, Vec3A};
use vgsurf::MicroSurfaceMesh;

/// Extra data associated with a ray stream.
///
/// This is used to record the trajectory, energy, last hit info and bounce
/// count of each ray. It will be attached to the intersection context to be
/// used by the intersection filter.
#[derive(Debug, Clone)]
pub struct RayStreamData<'a> {
    /// The micro-surface geometry.
    msurf: Arc<Geometry<'a>>,
    /// The last hit for each ray.
    last_hit: Vec<LastHit>,
    /// The trajectory of each ray. Each ray's trajectory is started with the
    /// initial ray and then followed by the rays after each bounce. The cosine
    /// of the incident angle at each bounce is also recorded.
    trajectory: Vec<RayTrajectory>,
}

unsafe impl Send for RayStreamData<'_> {}
unsafe impl Sync for RayStreamData<'_> {}

type QueryContext<'a, 'b> = IntersectContextExt<&'b mut RayStreamData<'a>>;

fn intersect_filter_stream<'a>(
    rays: RayN<'a>,
    hits: HitN<'a>,
    mut valid: ValidityN,
    ctx: &mut QueryContext,
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
            #[cfg(all(debug_assertions, feature = "verbose_debug"))]
            log::trace!("ray {} -- hit: {}", ray_id, prim_id);
            let last_hit = ctx.ext.last_hit[ray_id as usize];

            if prim_id != INVALID_ID && prim_id == last_hit.prim_id {
                #[cfg(all(debug_assertions, feature = "verbose_debug"))]
                log::trace!("nudging ray origin");
                // if the ray hit the same primitive as the previous bounce,
                // nudging the ray origin slightly along normal direction of
                // the last hit point and re-tracing the ray.
                valid[i] = ValidMask::Valid as i32;
                let traj_node = ctx.ext.trajectory[ray_id as usize].last_mut().unwrap();
                traj_node.org += last_hit.normal * 1e-4;
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
                let normal: Vec3A = hits.unit_normal(i).into();
                let ray_dir: Vec3A = rays.unit_dir(i).into();

                if ray_dir.dot(normal) > 0.0 {
                    #[cfg(all(debug_assertions, feature = "verbose_debug"))]
                    log::trace!("ray {} hit backface", ray_id);
                    valid[i] = ValidMask::Invalid as i32;
                    continue;
                }

                let new_dir = fresnel::reflect(ray_dir, normal);
                let last_id = ctx.ext.trajectory[ray_id as usize].len() - 1;
                ctx.ext.trajectory[ray_id as usize][last_id].cos = Some(ray_dir.dot(normal));
                ctx.ext.trajectory[ray_id as usize].push(RayTrajectoryNode {
                    org: point,
                    dir: new_dir,
                    cos: None,
                });
                let last_hit = &mut ctx.ext.last_hit[ray_id as usize];
                last_hit.geom_id = hits.geom_id(i);
                last_hit.prim_id = prim_id;
                last_hit.normal = normal;
            }
        } else {
            // if the ray didn't hit anything, mark it as invalid
            #[cfg(all(debug_assertions, feature = "verbose_debug"))]
            log::trace!("ray {} missed", rays.id(i));
            valid[i] = ValidMask::Invalid as i32;
        }
    }
}

/// Measures the BSDF of a micro-surface mesh.
///
/// Full BSDF means that the BSDF is measured for all the emitter positions
/// and all the collector patches.
///
/// # Arguments
///
/// * `desc` - The BSDF measurement description.
/// * `mesh` - The micro-surface's mesh.
/// * `cache` - The cache to use.
/// * `emitter_samples` - The emitter samples.
/// * `collector_patches` - The collector patches.
pub fn measure_full_bsdf(
    params: &BsdfMeasurementParams,
    mesh: &MicroSurfaceMesh,
    samples: &EmitterSamples,
    patches: &CollectorPatches,
    cache: &Cache,
) -> MeasuredBsdfData {
    let device = Device::with_config(Config::default()).unwrap();
    let mut scene = device.create_scene().unwrap();
    scene.set_flags(SceneFlags::ROBUST);

    // Calculate emitter's radius to match the surface's dimensions.
    let emitter_orbit_radius = params.emitter.estimate_orbit_radius(mesh);
    let emitter_shape_radius = params.emitter.estimate_disk_radius(mesh);
    log::debug!("mesh extent: {:?}", mesh.bounds);
    log::debug!("emitter orbit radius: {}", emitter_orbit_radius);
    log::debug!("emitter disk radius: {:?}", emitter_shape_radius);
    // Upload the surface's mesh to the Embree scene.
    let mut geometry = mesh.as_embree_geometry(&device);
    geometry.set_intersect_filter_function(intersect_filter_stream);
    geometry.commit();

    scene.attach_geometry(&geometry);
    scene.commit();

    let mut data = Vec::with_capacity(params.emitter.samples_count());
    // Iterate over every incident direction.
    for pos in params.emitter.measurement_points() {
        data.push(measure_bsdf_at_point_inner(
            pos,
            params,
            mesh,
            samples,
            patches,
            emitter_orbit_radius,
            emitter_shape_radius,
            Arc::new(geometry.clone()),
            &scene,
            cache,
        ))
    }

    MeasuredBsdfData {
        params: *params,
        samples: data,
    }
}

/// Measures the BSDF of micro-surface mesh at the given position.
/// The BSDF is measured by emitting rays from the given position.
pub(crate) fn measure_bsdf_at_point(
    params: &BsdfMeasurementParams,
    mesh: &MicroSurfaceMesh,
    samples: &EmitterSamples,
    patches: &CollectorPatches,
    cache: &Cache,
    position: Sph3,
) -> BsdfMeasurementDataPoint<BounceAndEnergy> {
    let device = Device::with_config(Config::default()).unwrap();
    let mut scene = device.create_scene().unwrap();
    scene.set_flags(SceneFlags::ROBUST);

    // Calculate emitter's radius to match the surface's dimensions.
    let emitter_orbit_radius = params.emitter.estimate_orbit_radius(mesh);
    let emitter_shape_radius = params.emitter.estimate_disk_radius(mesh);
    log::debug!("mesh extent: {:?}", mesh.bounds);
    log::debug!("emitter orbit radius: {}", emitter_orbit_radius);
    log::debug!("emitter disk radius: {:?}", emitter_shape_radius);
    // Upload the surface's mesh to the Embree scene.
    let mut geometry = mesh.as_embree_geometry(&device);
    geometry.set_intersect_filter_function(intersect_filter_stream);
    geometry.commit();

    scene.attach_geometry(&geometry);
    scene.commit();

    measure_bsdf_at_point_inner(
        position,
        params,
        mesh,
        samples,
        patches,
        emitter_orbit_radius,
        emitter_shape_radius,
        Arc::new(geometry.clone()),
        &scene,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_bsdf_at_point_inner(
    pos: Sph3,
    params: &BsdfMeasurementParams,
    mesh: &MicroSurfaceMesh,
    samples: &EmitterSamples,
    patches: &CollectorPatches,
    emitter_orbit_radius: f32,
    emitter_shape_radius: Option<f32>,
    geometry: Arc<Geometry>,
    scene: &Scene,
    cache: &Cache,
) -> BsdfMeasurementDataPoint<BounceAndEnergy> {
    println!("      {BRIGHT_YELLOW}>{RESET} Emit rays from {}", pos);
    #[cfg(all(debug_assertions, feature = "verbose_debug"))]
    let t = Instant::now();
    let emitted_rays = Emitter::emit_rays(samples, pos, emitter_orbit_radius, emitter_shape_radius);
    let num_emitted_rays = emitted_rays.len();
    #[cfg(all(debug_assertions, feature = "verbose_debug"))]
    let elapsed = t.elapsed();
    let max_bounces = params.emitter.max_bounces;

    #[cfg(all(debug_assertions, feature = "verbose_debug"))]
    log::debug!(
        "emitted {} rays with dir: {} from: {} in {} secs.",
        num_emitted_rays,
        emitted_rays[0].dir,
        pos,
        elapsed.as_secs_f64(),
    );
    let num_streams = (num_emitted_rays + MAX_RAY_STREAM_SIZE - 1) / MAX_RAY_STREAM_SIZE;
    // In case the number of rays is less than one stream, we need to
    // adjust the stream size to match the number of rays.
    let stream_size = if num_streams == 1 {
        num_emitted_rays
    } else {
        MAX_RAY_STREAM_SIZE
    };

    let mut stream_data = vec![
        RayStreamData {
            msurf: geometry.clone(),
            last_hit: vec![
                LastHit {
                    geom_id: INVALID_ID,
                    prim_id: INVALID_ID,
                    normal: Vec3A::ZERO,
                };
                stream_size
            ],
            trajectory: vec![RayTrajectory(Vec::with_capacity(max_bounces as usize)); stream_size],
        };
        num_streams
    ];

    println!(
        "        {BRIGHT_YELLOW}>{RESET} Trace {} rays ({} streams)",
        emitted_rays.len(),
        num_streams
    );

    emitted_rays
        .par_chunks(MAX_RAY_STREAM_SIZE)
        .zip(stream_data.par_iter_mut())
        // .chunks(MAX_RAY_STREAM_SIZE).zip(stream_data.iter_mut())
        .enumerate()
        .for_each(|(_i, (rays, data))| {
            #[cfg(all(debug_assertions, feature = "verbose_debug"))]
            log::trace!("stream {} of {}", _i, num_streams);
            // Populate embree ray stream with generated rays.
            let chunk_size = rays.len();
            let mut ray_hit_n = RayHitNp::new(RayNp::new(chunk_size));
            for (i, mut ray) in ray_hit_n.ray.iter_mut().enumerate() {
                ray.set_org(rays[i].org.into());
                ray.set_dir(rays[i].dir.into());
                ray.set_id(i as u32);
                ray.set_tnear(0.0);
                ray.set_tfar(f32::INFINITY);
            }

            for (i, ray) in rays.iter().enumerate() {
                data.trajectory[i].push(RayTrajectoryNode {
                    org: ray.org.into(),
                    dir: ray.dir.into(),
                    cos: None,
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

                #[cfg(all(debug_assertions, feature = "verbose_debug"))]
                {
                    log::trace!(
                        "------------ bounce {}, active rays {}\n {:?} | {}",
                        bounces,
                        active_rays,
                        ctx.ext.trajectory,
                        ctx.ext.trajectory.len(),
                    );
                    log::trace!("validities: {:?}", validities);
                }

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

            #[cfg(all(debug_assertions, feature = "verbose_debug"))]
            log::trace!(
                "------------ result {}, active rays: {}, valid rays: {:?}\ntrajectory: {:?} | {}",
                bounces,
                active_rays,
                validities,
                data.trajectory,
                data.trajectory.len(),
            );
        });
    // Extract the trajectory of each ray.
    let trajectories = stream_data
        .into_iter()
        .flat_map(|d| d.trajectory)
        .collect::<Vec<_>>();

    #[cfg(all(debug_assertions, feature = "verbose_debug"))]
    {
        let collected = params
            .collector
            .collect(params, mesh, pos, &trajectories, patches, cache);
        log::debug!("collected stats: {:#?}", collected.stats);
        log::trace!("collected: {:?}", collected.data);
        collected
    }
    #[cfg(not(all(debug_assertions, feature = "verbose_debug")))]
    params
        .collector
        .collect(params, mesh, pos, &trajectories, patches, cache)
}

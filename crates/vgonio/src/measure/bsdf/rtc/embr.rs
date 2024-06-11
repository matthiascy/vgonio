//! Embree ray tracing.

use crate::{
    app::cli::ansi,
    measure::bsdf::{
        emitter::Emitter,
        rtc::{compute_num_of_streams, HitInfo, MAX_RAY_STREAM_SIZE},
        SingleSimulationResult,
    },
};

#[cfg(feature = "visu-dbg")]
use crate::measure::bsdf::rtc::{RayTrajectory, RayTrajectoryNode};
#[cfg(not(feature = "visu-dbg"))]
use crate::measure::params::BsdfMeasurementParams;
#[cfg(not(feature = "visu-dbg"))]
use base::optics::ior::Ior;
use base::{
    math::{Sph2, Vec3, Vec3A},
    optics::fresnel,
};
use embree::{
    BufferUsage, Config, Device, Geometry, HitN, IntersectContext, IntersectContextExt,
    IntersectContextFlags, RayHitNp, RayN, RayNp, Scene, SceneFlags, SoAHit, SoARay, ValidMask,
    ValidityN, INVALID_ID,
};
use rayon::prelude::*;
use std::sync::Arc;
#[cfg(all(debug_assertions, feature = "verbose-dbg"))]
use std::time::Instant;
use surf::MicroSurfaceMesh;

/// SoA ray stream data for the whole ray stream.
#[derive(Debug, Clone)]
struct SoARayStreams<'g> {
    /// The total number of rays in the stream.
    pub n_ray: usize,
    /// The number of sub-streams, each with a maximum size of
    /// [`Self::RAY_STREAM_SIZE`].
    pub n_stream: usize,
    /// The number of rays in the last stream in case the total number of rays
    /// is not a multiple of [`Self::RAY_STREAM_SIZE`].
    pub last_stream_size: usize,
    /// The total size of the ray stream in the whole stream; should be equal to
    /// or greater than `n_ray`.
    pub total_stream_size: usize,
    /// The micro-surface geometry.
    pub msurf: Arc<Geometry<'g>>,
    /// The last hit for each ray.
    pub last_hit: Box<[HitInfo]>,
    #[cfg(feature = "visu-dbg")]
    /// The trajectory of each ray. Each ray's trajectory is started with
    /// the initial ray and then followed by the rays after each
    /// bounce. The cosine of the incident angle at each bounce is
    /// also recorded.
    pub trajectory: Box<[RayTrajectory]>,
    #[cfg(not(feature = "visu-dbg"))]
    /// The refractive indices of the incident medium.
    pub iors_i: &'g [Ior],
    #[cfg(not(feature = "visu-dbg"))]
    /// The refractive indices of the transmitted medium.
    pub iors_t: &'g [Ior],
    #[cfg(not(feature = "visu-dbg"))]
    /// Bounce count for each ray.
    pub bounce: Box<[u32]>,
    #[cfg(not(feature = "visu-dbg"))]
    /// Energy of each ray per wavelength. The first dimension is the index
    /// of the ray, and the second dimension is the wavelength index
    /// [id, wl].
    pub energy: Box<[f32]>,
}

unsafe impl Send for SoARayStreams<'_> {}
unsafe impl Sync for SoARayStreams<'_> {}

impl<'g> SoARayStreams<'g> {
    /// Maximum number of rays that can be traced in a single stream.
    pub const RAY_STREAM_SIZE: usize = 1024;

    /// Create a new instance of the ray stream data.
    pub fn new(
        msurf: Arc<Geometry<'g>>,
        n_ray: usize,
        #[cfg(feature = "visu-dbg")] max_bounces: u32,
        #[cfg(not(feature = "visu-dbg"))] iors_i: &'g [Ior],
        #[cfg(not(feature = "visu-dbg"))] iors_t: &'g [Ior],
    ) -> Self {
        let n_stream = compute_num_of_streams(n_ray);
        let last_stream_size = n_ray % Self::RAY_STREAM_SIZE;
        let total_stream_size = n_stream * Self::RAY_STREAM_SIZE;
        let last_hit = vec![HitInfo::new(); n_ray].into_boxed_slice();
        Self {
            n_ray,
            n_stream,
            last_stream_size,
            total_stream_size,
            msurf,
            last_hit,
            #[cfg(feature = "visu-dbg")]
            trajectory: vec![RayTrajectory(Vec::with_capacity(max_bounces as usize / 2)); n_ray]
                .into_boxed_slice(),
            #[cfg(not(feature = "visu-dbg"))]
            iors_i,
            #[cfg(not(feature = "visu-dbg"))]
            iors_t,
            #[cfg(not(feature = "visu-dbg"))]
            bounce: vec![0; n_ray].into_boxed_slice(),
            #[cfg(not(feature = "visu-dbg"))]
            energy: vec![1.0; n_ray * iors_t.len()].into_boxed_slice(),
        }
    }

    /// Returns a mutable iterator over the ray streams.
    pub fn streams_mut(&mut self) -> SoARayStreamsIterMut<'g> {
        SoARayStreamsIterMut {
            data: self as *mut Self,
            stream_idx: 0,
        }
    }
}

struct SoARayStreamsIterMut<'g> {
    data: *mut SoARayStreams<'g>,
    stream_idx: usize,
}

unsafe impl Send for SoARayStreamsIterMut<'_> {}

struct SoARayStreamMut<'a> {
    idx: usize,
    size: usize,
    msurf: Arc<Geometry<'a>>,
    last_hit: &'a mut [HitInfo],
    #[cfg(feature = "visu-dbg")]
    trajectory: &'a mut [RayTrajectory],
    #[cfg(not(feature = "visu-dbg"))]
    iors_i: &'a [Ior],
    #[cfg(not(feature = "visu-dbg"))]
    iors_t: &'a [Ior],
    #[cfg(not(feature = "visu-dbg"))]
    bounce: &'a mut [u32],
    #[cfg(not(feature = "visu-dbg"))]
    energy: &'a mut [f32],
}

unsafe impl Send for SoARayStreamMut<'_> {}

impl<'a> Iterator for SoARayStreamsIterMut<'a> {
    type Item = SoARayStreamMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let data = unsafe { &mut *self.data };
        if self.stream_idx < data.n_stream {
            let stream_size = if self.stream_idx == data.n_stream - 1 {
                data.last_stream_size
            } else {
                SoARayStreams::RAY_STREAM_SIZE
            };
            let data_range = self.stream_idx * SoARayStreams::RAY_STREAM_SIZE
                ..self.stream_idx * SoARayStreams::RAY_STREAM_SIZE + stream_size;

            let stream = {
                #[cfg(not(feature = "visu-dbg"))]
                {
                    let n_spectrum = data.iors_t.len();
                    let energy_range = self.stream_idx * SoARayStreams::RAY_STREAM_SIZE * n_spectrum
                        ..(self.stream_idx * SoARayStreams::RAY_STREAM_SIZE + stream_size)
                            * n_spectrum;
                    SoARayStreamMut {
                        idx: self.stream_idx,
                        size: stream_size,
                        msurf: data.msurf.clone(),
                        last_hit: &mut data.last_hit[data_range.clone()],
                        iors_i: data.iors_i,
                        iors_t: data.iors_t,
                        bounce: &mut data.bounce[data_range.clone()],
                        energy: &mut data.energy[energy_range],
                    }
                }
                #[cfg(feature = "visu-dbg")]
                {
                    SoARayStreamMut {
                        idx: self.stream_idx,
                        size: stream_size,
                        msurf: data.msurf.clone(),
                        last_hit: &mut data.last_hit[data_range.clone()],
                        trajectory: &mut data.trajectory[data_range],
                    }
                }
            };
            self.stream_idx += 1;
            Some(stream)
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for SoARayStreamsIterMut<'a> {
    fn len(&self) -> usize { unsafe { &*self.data }.n_stream }
}

type QueryContext<'a> = IntersectContextExt<SoARayStreamMut<'a>>;

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
            let ray_id = rays.id(i) as usize;
            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
            log::trace!("ray {} -- hit: {}", ray_id, prim_id);

            let hit_info = &mut ctx.ext.last_hit[ray_id];
            if prim_id != INVALID_ID && prim_id == hit_info.last_prim_id {
                #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                log::trace!("nudging ray origin");
                // if the ray hit the same primitive as the previous bounce,
                // nudging the ray origin slightly along the normal direction of
                // the last hit point and re-tracing the ray.
                valid[i] = ValidMask::Valid as i32;

                #[cfg(not(feature = "visu-dbg"))]
                {
                    hit_info.last_pos += hit_info.last_normal * 1e-4;
                    hit_info.factor = -1.0;
                }

                #[cfg(feature = "visu-dbg")]
                {
                    let traj_node = ctx.ext.trajectory[ray_id].last_mut().unwrap();
                    traj_node.org += hit_info.last_normal * 1e-4;
                }
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
                    #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                    log::trace!("ray {} hit backface", ray_id);
                    valid[i] = ValidMask::Invalid as i32;
                    continue;
                }

                let new_dir = fresnel::reflect(ray_dir, normal);
                let cos_i = ray_dir.dot(normal);
                #[cfg(feature = "visu-dbg")]
                {
                    let last_id = ctx.ext.trajectory[ray_id].len() - 1;
                    ctx.ext.trajectory[ray_id][last_id].cos = Some(cos_i);
                    ctx.ext.trajectory[ray_id].push(RayTrajectoryNode {
                        org: point,
                        dir: new_dir,
                        cos: None,
                    });
                }
                #[cfg(not(feature = "visu-dbg"))]
                {
                    hit_info.last_pos = point;
                    hit_info.next_dir = new_dir;
                    hit_info.factor = cos_i;
                }
                hit_info.last_geom_id = hits.geom_id(i);
                hit_info.last_prim_id = prim_id;
                hit_info.last_normal = normal;
            }
        } else {
            // if the ray didn't hit anything, mark it as invalid
            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
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
/// `emitter` - The emitter.
/// `mesh` - The micro-surface mesh to measure.
/// `w_i` - The incident direction. If `None`, the BSDF is measured for all
///        the emitter positions.
pub fn simulate_bsdf_measurement<'a, 'b: 'a>(
    #[cfg(not(feature = "visu-dbg"))] params: &'a BsdfMeasurementParams,
    emitter: &'a Emitter,
    mesh: &'a MicroSurfaceMesh,
    #[cfg(not(feature = "visu-dbg"))] iors_i: &'b [Ior],
    #[cfg(not(feature = "visu-dbg"))] iors_t: &'b [Ior],
) -> Box<[SingleSimulationResult]> {
    #[cfg(feature = "bench")]
    let t = std::time::Instant::now();

    let device = Device::with_config(Config::default()).unwrap();
    let mut scene = device.create_scene().unwrap();
    scene.set_flags(SceneFlags::ROBUST);

    // Upload the surface's mesh to the Embree scene.
    let mut geometry = mesh.as_embree_geometry(&device);
    geometry.set_intersect_filter_function(intersect_filter_stream);
    geometry.commit();

    scene.attach_geometry(&geometry);
    scene.commit();

    #[cfg(feature = "bench")]
    {
        let elapsed = t.elapsed();
        log::debug!("embree scene creation took {} secs.", elapsed.as_secs_f64());
    }

    let mut results = Box::new_uninit_slice(emitter.measpts.len());

    for (i, w_i) in emitter.measpts.iter().enumerate() {
        #[cfg(feature = "visu-dbg")]
        results[i].write(simulate_bsdf_measurement_single_point(
            *w_i,
            emitter,
            mesh,
            Arc::new(geometry.clone()),
            &scene,
        ));

        #[cfg(not(feature = "visu-dbg"))]
        results[i].write(simulate_bsdf_measurement_single_point(
            *w_i,
            emitter,
            mesh,
            Arc::new(geometry.clone()),
            &scene,
            params.fresnel,
            iors_i,
            iors_t,
        ));
    }

    #[cfg(feature = "bench")]
    {
        let elapsed = t.elapsed();
        log::debug!(
            "bsdf measurement simulation (one snapshot) took {} secs.",
            elapsed.as_secs_f64()
        );
    }

    unsafe { results.assume_init() }
}

/// Simulates the BSDF measurement for a single incident direction (point).
fn simulate_bsdf_measurement_single_point<'a, 'b: 'a>(
    w_i: Sph2,
    emitter: &Emitter,
    mesh: &MicroSurfaceMesh,
    geometry: Arc<Geometry<'a>>,
    scene: &Scene,
    #[cfg(not(feature = "visu-dbg"))] fresnel: bool,
    #[cfg(not(feature = "visu-dbg"))] iors_i: &'b [Ior],
    #[cfg(not(feature = "visu-dbg"))] iors_t: &'b [Ior],
) -> SingleSimulationResult {
    println!(
        "      {}>{} Emit rays from {}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        w_i
    );
    #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
    let t = Instant::now();
    let emitted_rays = emitter.emit_rays(w_i, mesh);
    let num_emitted_rays = emitted_rays.len();
    #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
    let elapsed = t.elapsed();
    let max_bounces = emitter.params.max_bounces;

    #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
    log::debug!(
        "emitted {} rays with dir: {} from: {} in {} secs.",
        num_emitted_rays,
        emitted_rays[0].dir,
        w_i,
        elapsed.as_secs_f64(),
    );

    #[cfg(not(feature = "visu-dbg"))]
    let n_spectrum = iors_t.len();

    // Allocate the stream data for all the rays.
    let mut stream_data = SoARayStreams::new(
        geometry.clone(),
        num_emitted_rays,
        #[cfg(feature = "visu-dbg")]
        max_bounces,
        #[cfg(not(feature = "visu-dbg"))]
        iors_i,
        #[cfg(not(feature = "visu-dbg"))]
        iors_t,
    );
    #[cfg(all(debug_assertions))]
    println!(
        "        {} Trace {} rays ({} streams rays)",
        ansi::YELLOW_GT,
        emitted_rays.len(),
        stream_data.total_stream_size
    );

    // // TODO: use SOA for the stream data.
    // let mut stream_data = {
    //     let mut data = Box::new_uninit_slice(num_streams);
    //     for i in 0..num_streams {
    //         let stream_size = if i == num_streams - 1 {
    //             last_stream_size
    //         } else {
    //             stream_size
    //         };
    //         data[i].write(RayStreamData {
    //             msurf: geometry.clone(),
    //             last_hit: vec![HitInfo::new(); stream_size],
    //             #[cfg(not(feature = "visu-dbg"))]
    //             iors_i,
    //             #[cfg(not(feature = "visu-dbg"))]
    //             iors_t,
    //             #[cfg(not(feature = "visu-dbg"))]
    //             bounce: vec![0; stream_size],
    //             #[cfg(not(feature = "visu-dbg"))]
    //             energy: DyArr::ones([stream_size, n_spectrum]),
    //             #[cfg(feature = "visu-dbg")]
    //             trajectory: vec![
    //                 RayTrajectory(Vec::with_capacity(max_bounces as usize));
    //                 stream_size
    //             ],
    //         });
    //     }
    //     unsafe { data.assume_init().into_vec() }
    // };

    emitted_rays
        .chunks(MAX_RAY_STREAM_SIZE)
        .zip(stream_data.streams_mut())
        .par_bridge()
        .for_each(|(rays, data)| {
            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
            log::trace!("stream {} of {}", data.idx, stream_data.n_stream);

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

            #[cfg(feature = "visu-dbg")]
            {
                for (i, ray) in rays.iter().enumerate() {
                    data.trajectory[i].push(RayTrajectoryNode {
                        org: ray.org.into(),
                        dir: ray.dir.into(),
                        cos: None,
                    });
                    data.trajectory[i].shrink_to_fit();
                }
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

                #[cfg(all(debug_assertions, feature = "verbose-dbg", feature = "visu-dbg"))]
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
                    #[cfg(not(feature = "visu-dbg"))]
                    {
                        // Only update new ray origin and direction if the ray
                        // is reflected.
                        let hit_info = &mut ctx.ext.last_hit[i];
                        ray_hit_n.ray.set_org(i, hit_info.last_pos.into());
                        ray_hit_n.ray.set_dir(i, hit_info.next_dir.into());
                        ray_hit_n.ray.set_tnear(i, 0.00001);
                        ray_hit_n.ray.set_tfar(i, f32::INFINITY);
                        // Reset the hit information.
                        ray_hit_n.hit.set_geom_id(i, INVALID_ID);
                        ray_hit_n.hit.set_prim_id(i, INVALID_ID);
                        ctx.ext.bounce[i] += 1;

                        // Only update the energy if we are not dealing with the
                        // self-intersection case.
                        if hit_info.factor >= 0.0 {
                            let cos_i_abs = hit_info.factor.abs();
                            if fresnel {
                                for k in 0..n_spectrum {
                                    ctx.ext.energy[i * n_spectrum + k] *=
                                        fresnel::reflectance(cos_i_abs, &iors_i[k], &iors_t[k]);
                                }
                            } else {
                                for k in 0..n_spectrum {
                                    ctx.ext.energy[i * n_spectrum + k] *= cos_i_abs;
                                }
                            }
                        }
                    }

                    #[cfg(feature = "visu-dbg")]
                    {
                        let next_traj = ctx.ext.trajectory[i].last().unwrap();
                        ray_hit_n.ray.set_org(i, next_traj.org.into());
                        ray_hit_n.ray.set_dir(i, next_traj.dir.into());
                        ray_hit_n.ray.set_tnear(i, 0.00001);
                        ray_hit_n.ray.set_tfar(i, f32::INFINITY);
                        // Reset the hit information.
                        ray_hit_n.hit.set_geom_id(i, INVALID_ID);
                        ray_hit_n.hit.set_prim_id(i, INVALID_ID);
                    }
                }

                bounces += 1;
            }

            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
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
    #[cfg(feature = "visu-dbg")]
    {
        SingleSimulationResult {
            wi: w_i,
            trajectories: stream_data.trajectory,
        }
    }

    #[cfg(not(feature = "visu-dbg"))]
    {
        use jabr::array::DyArr;
        // Unpack the stream data into a single result.
        let dirs = stream_data
            .last_hit
            .iter()
            .map(|h| Vec3::from(h.next_dir))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        // TODO: number of emitted rays should be checked.
        SingleSimulationResult {
            wi: w_i,
            bounces: stream_data.bounce,
            dirs,
            energy: DyArr::from_boxed_slice([num_emitted_rays, n_spectrum], stream_data.energy),
        }
    }
}

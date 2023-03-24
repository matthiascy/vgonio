//! Embree ray tracing.

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    measure::{
        bsdf::SpectrumSampler,
        measurement::{BsdfMeasurement, Radius},
    },
    msurf::MicroSurfaceMesh,
    optics::fresnel,
    units::um,
};
use embree::{
    BufferUsage, Config, Device, Format, Geometry, GeometryKind, HitN, IntersectContext,
    IntersectContextExt, IntersectContextFlags, RayHitNp, RayN, RayNp, SceneFlags, SoAHit, SoARay,
    ValidMask, ValidityN, INVALID_ID,
};
use glam::{Vec3, Vec3A};
use std::{sync::Arc, time::Instant};
use std::ops::{Deref, DerefMut};
use crate::measure::collector::CollectorPatches;
use crate::optics::ior::RefractiveIndex;

impl MicroSurfaceMesh {
    /// Constructs an embree geometry from the `MicroSurfaceMesh`.
    pub fn as_embree_geometry<'g>(&self, device: &Device) -> Geometry<'g> {
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

/// Records the status of a traced ray.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryNode {
    /// The origin of the ray.
    pub org: Vec3A,
    /// The direction of the ray.
    pub dir: Vec3A,
    /// The cosine of the incident angle (always positive),
    /// only has value if the ray has hit the micro-surface.
    pub cos: Option<f32>,
}

/// Records the trajectory of a ray from the moment it is spawned.
///
/// The trajectory always starts with the ray that is spawned.
#[derive(Debug, Clone)]
pub struct Trajectory(pub (crate) Vec<TrajectoryNode>);

impl Deref for Trajectory {
    type Target = Vec<TrajectoryNode>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Trajectory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Trajectory {
    /// Returns `true` if the ray did not hit anything.
    pub fn is_missed(&self) -> bool {
        self.0.len() <= 1
    }

    /// Returns the last ray of the trajectory if the ray hit the micro-surface or was absorbed,
    /// `None` in case if the ray did not hit anything.
    pub fn last(&self) -> Option<&TrajectoryNode> {
        if self.is_missed() {
            None
        } else {
            Some(&self.0.last().unwrap())
        }
    }
}

// #[derive(Debug, Clone, Copy)]
// /// The status of a ray after it has been traced.
// pub enum TracingStatus {
//     /// The ray did not hit anything.
//     Missed,
//     /// The ray hit the micro-surface or was absorbed.
//     /// Stores the last ray exiting the micro-surface.
//     Reflected(TrajectoryNode),
//     /// The ray hit the micro-surface or was absorbed.
//     /// Stores the last ray entering the micro-surface.
//     Absorbed(TrajectoryNode),
// }

// pub type PerWavelengthVec<T> = Vec<T>;
// pub type PerWavelengthSlice<T> = [T];

// /// Records the status of a traced ray stream.
// #[derive(Debug, Clone)]
// pub struct RayStreamStats {
//     /// The last ray exiting/entering the micro-surface for each wavelength.
//     status: PerWavelengthVec<Vec<TracingStatus>>,
//     /// Final energy per wavelength for each ray in the stream.
//     energy: PerWavelengthVec<Vec<f32>>,
//     /// Number of bounces per wavelength for each ray in the stream.
//     bounce: PerWavelengthVec<Vec<u32>>,
// }
//
// impl<'a> IntoIterator for &'a RayStreamStats {
//     type Item = RayStatus<'a>;
//     type IntoIter = RayStreamStatsIter<'a>;
//
//     fn into_iter(self) -> Self::IntoIter {
//         RayStreamStatsIter {
//             stats: self,
//             index: 0,
//         }
//     }
// }
//
// /// Records the status per wavelength of a traced ray.
// #[derive(Debug, Clone)]
// pub struct RayStatus<'a> {
//     /// The last ray exiting the micro-surface.
//     pub tracing_status: &'a PerWavelengthSlice<TracingStatus>,
//     /// Final energy of the ray.
//     pub energy: &'a PerWavelengthSlice<f32>,
//     /// Number of bounces for each ray.
//     pub bounce: &'a PerWavelengthSlice<u32>,
// }
//
// /// Iterator for `RayStatsPerStream`.
// pub struct RayStreamStatsIter<'a> {
//     stats: &'a RayStreamStats,
//     index: usize,
// }
//
// impl<'a> Iterator for RayStreamStatsIter<'a> {
//     type Item = RayStatus<'a>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index < self.stats.status.len() {
//             let status = &self.stats.status[self.index];
//             let energy = &self.stats.energy[self.index];
//             let bounce = &self.stats.bounce[self.index];
//             self.index += 1;
//             Some(RayStatus {
//                 tracing_status: status,
//                 energy,
//                 bounce,
//             })
//         } else {
//             None
//         }
//     }
// }

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
    last_hit: Vec<HitData>,
    /// The trajectory of each ray. Each ray's trajectory is started with the
    /// initial ray and then followed by the rays after each bounce. The cosine
    /// of the incident angle at each bounce is also recorded.
    trajectory: Vec<Trajectory>,
}

unsafe impl Send for RayStreamData<'_> {}
unsafe impl Sync for RayStreamData<'_> {}

// impl<'a> RayStreamData<'a> {
//     /// Converts the ray stream data used during tracing into the final ray stats.
//     pub fn collect_into_stats(self, iors_i: &[RefractiveIndex], iors_t: &[RefractiveIndex]) -> RayStreamStats {
//         debug_assert_eq!(self.trajectory.len(), self.last_hit.len(), "number of trajectories and last hits must be the same");
//         debug_assert_eq!(iors_i.len(), iors_t.len(), "number of incident and transmitted iors must be the same");
//         let spectrum_len = iors_t.len();
//         let (status, (energy, bounce)) = self.trajectory
//             .into_iter()
//             .map(|traj| {
//                 debug_assert!(!traj.is_empty());
//                 // Calculate the final energy of the ray.
//                 match traj.len() {
//                     0 => unreachable!("trajectory cannot be empty since we have the original ray as the first element"),
//                     1 => {
//                         (vec![TracingStatus::Missed; spectrum_len], (vec![1.0; spectrum_len], vec![0; spectrum_len]))
//                     },
//                     n => {
//                         let mut energy = vec![1.0; spectrum_len];
//                         let mut bounce = vec![0u32; spectrum_len];
//                         let last_ray = traj[n - 1];
//                         let mut status = (0..spectrum_len)
//                             .map(|_| TracingStatus::Reflected(last_ray)).collect::<Vec<_>>();
//                         for node in traj.iter().take(n - 1) {
//                             for i in 0..spectrum_len {
//                                 if energy[i] <= 0.0 {
//                                     status[i] = TracingStatus::Absorbed(last_ray);
//                                     continue;
//                                 }
//                                 let cos = node.cos.unwrap_or(1.0);
//                                 energy[i] *= fresnel::reflectance(
//                                     cos,
//                                     iors_i[i],
//                                     iors_t[i]
//                                 );
//                                 bounce[i] += 1;
//                             }
//                         }
//                         (status, (energy, bounce))
//                     }
//                 }
//             })
//             .unzip();
//         RayStreamStats {
//             status,
//             energy,
//             bounce,
//         }
//     }
// }

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
            log::trace!("ray {} -- hit: {}", ray_id, prim_id);
            let last_hit = ctx.ext.last_hit[ray_id as usize];

            if prim_id != INVALID_ID && prim_id == last_hit.prim_id {
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
                let normal = hits.unit_normal(i).into();
                let ray_dir = rays.unit_dir(i).into();
                let new_dir = fresnel::reflect(ray_dir, normal);
                let last_id = ctx.ext.trajectory[ray_id as usize].len() - 1;
                ctx.ext.trajectory[ray_id as usize][last_id].cos = Some(ray_dir.dot(normal));
                ctx.ext.trajectory[ray_id as usize].push(TrajectoryNode {
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
            log::trace!("ray {} missed", rays.id(i));
            valid[i] = ValidMask::Invalid as i32;
        }
    }
}

const MAX_RAY_STREAM_SIZE: usize = 1024;

/// Measures the BSDF of a micro-surface.
///
/// # Arguments
///
/// * `desc` - The BSDF measurement description.
/// * `surf` - The micro-surface to measure.
/// * `mesh` - The micro-surface's mesh.
/// * `cache` - The cache to use.
pub fn measure_bsdf(desc: &BsdfMeasurement, mesh: &MicroSurfaceMesh, cache: &Cache, emitter_samples: &[Vec3], collector_patches: &CollectorPatches) {
    let device = Device::with_config(Config::default()).unwrap();
    let mut scene = device.create_scene().unwrap();
    scene.set_flags(SceneFlags::ROBUST);

    // Calculate emitter's radius to match the surface's dimensions.
    let radius = match desc.emitter.radius() {
        Radius::Auto(_) => um!(mesh.extent.max_edge() * 2.5),
        Radius::Fixed(r) => r.in_micrometres(),
    };

    log::debug!("mesh extent: {:?}", mesh.extent);
    log::debug!("emitter radius: {}", radius);

    // Upload the surface's mesh to the Embree scene.
    let mut geometry = mesh.as_embree_geometry(&device);
    geometry.set_intersect_filter_function(intersect_filter_stream);
    geometry.commit();

    scene.attach_geometry(&geometry);
    scene.commit();

    // Iterate over every incident direction.
    for pos in desc.emitter.meas_points() {
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Emit rays from {}° {}°",
            pos.zenith.in_degrees().value(),
            pos.azimuth.in_degrees().value()
        );

        let t = Instant::now();
        let emitted_rays = desc.emitter.emit_rays_with_radius(&emitter_samples, pos, radius);
        let num_emitted_rays = emitted_rays.len();
        let elapsed = t.elapsed();

        let max_bounces = desc.emitter.max_bounces;

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
                last_hit: vec![
                    HitData {
                        geom_id: INVALID_ID,
                        prim_id: INVALID_ID,
                        normal: Vec3A::ZERO,
                    };
                    stream_size
                ],
                trajectory: vec![Trajectory(Vec::with_capacity(max_bounces as usize)); stream_size],
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

                for ray in rays {
                    data.trajectory.push(Trajectory(vec![TrajectoryNode {
                        org: ray.o.into(),
                        dir: ray.d.into(),
                        cos: None,
                    }]));
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
                        ctx.ext.trajectory.len() - 1,
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
                    data.trajectory.len() - 1,
                    validities
                );
            });
        // Extract the trajectory of each ray.
        let trajectories = stream_data.into_iter().flat_map(|d| {
            d.trajectory
        }).collect::<Vec<_>>();
        desc.collector.collect_embree_rt(desc.kind, desc.incident_medium, desc.transmitted_medium, desc.emitter.spectrum, &trajectories, &collector_patches, &cache);
    }
}

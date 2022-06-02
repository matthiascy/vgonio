use crate::acq::desc::{MeasurementDesc, Range};
use crate::acq::embree_rt::EmbreeRayTracing;
use crate::acq::grid_rt::GridRayTracing;
use crate::acq::ior::{RefractiveIndex, RefractiveIndexDatabase};
use crate::acq::ray::{scattering_air_conductor, Ray, RayTraceRecord, Scattering};
use crate::acq::{Collector, Emitter, Patch};
use crate::htfld::Heightfield;
use crate::mesh::{TriangleMesh, TriangulationMethod};
use embree::{Config, RayHit, SoARay};
use glam::Vec3;

#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum BxdfKind {
    InPlane,
}

/// Measurement statistics for a single incident direction.
pub struct Stats<PatchData: Copy, const N_PATCH: usize, const N_BOUNCE: usize> {
    /// Wavelength of emitted rays.
    pub wavelength: f32,

    /// Incident polar angle in radians.
    pub zenith: f32,

    /// Incident azimuth angle in radians.
    pub azimuth: f32,

    /// Number of emitted rays.
    pub n_emitted: u32,

    /// Number of emitted rays that hit the surface.
    pub n_received: u32,

    /// Number of emitted rays that hit the surface and were reflected.
    pub n_reflected: u32,

    /// Number of emitted rays that hit the surface and were transmitted.
    pub n_transmitted: u32,

    /// Number of emitted rays captured by the collector.
    pub n_captured: u32,

    /// Energy of emitted rays.
    pub energy_emitted: f32,

    /// Energy of emitted rays that hit the surface.
    pub energy_received: f32,

    /// Energy captured by the collector.
    pub energy_captured: f32,

    /// Patch's zenith angle span in radians.
    pub zenith_span: f32,

    /// Patch's azimuth angle span in radians.
    pub azimuth_span: f32,

    /// Patches of the collector.
    pub patches: [Patch; N_PATCH],

    /// Per patch data.
    pub patches_data: [PatchData; N_PATCH],

    /// Histogram of reflected rays by number of bounces.
    pub hist_reflections: [u32; N_BOUNCE],

    /// Histogram of energy of reflected rays by number of bounces.
    pub hist_reflections_energy: [f32; N_BOUNCE],
}

/// Measurement of the in-plane BRDF (incident angle and outgoing angle are on
/// the same plane).
/// // <
// //     PatchData: Copy,
// //     const N_PATCH: usize,
// //     const N_BOUNCE: usize,
// // >
// // Vec<Stats<PatchData, N_PATCH, N_BOUNCE>>
pub fn measure_in_plane_brdf_embree(desc: &MeasurementDesc, ior_db: &RefractiveIndexDatabase,
                                    surfaces: &[Heightfield])
{
    let collector: Collector = desc.collector.into();
    let emitter: Emitter = desc.emitter.into();
    log::debug!("Emitter generated {} patches.", emitter.patches.len());

    let mut embree_rt = EmbreeRayTracing::new(Config::default());

    for surface in surfaces {
        let scene_id = embree_rt.create_scene();
        let triangulated = surface.triangulate(TriangulationMethod::Regular);
        let radius = triangulated.extent.max_edge() * 2.5;
        let surface_mesh = embree_rt.create_triangle_mesh(&triangulated);
        let surface_id = embree_rt.attach_geometry(scene_id, surface_mesh);
        let spectrum_samples = SpectrumSampler::from(desc.emitter.spectrum).samples();
        log::debug!("spectrum samples: {:?}", spectrum_samples);
        // let ior_i = ior_db
        //     .ior_of_spectrum(desc.incident_medium, &spectrum_samples)
        //     .unwrap();
        // let ior_t = ior_db
        //     .ior_of_spectrum(desc.transmitted_medium, &spectrum_samples)
        //     .unwrap();

        for wavelength in spectrum_samples {
            println!("Capturing with wavelength = {}", wavelength);
            let ior_t = ior_db.refractive_index_of(desc.transmitted_medium, wavelength).unwrap();

            // For all incident angles; generate samples on each patch
            for (i, patch) in emitter.patches.iter().enumerate() {
                // Emit rays from the patch of the emitter. Uniform sampling over the patch.
                let rays = patch.emit_rays(desc.emitter.num_rays, radius);
                log::debug!("Emitted {} rays from patch {} - {:?}: {:?}", rays.len(), i, patch, rays);

                // Populate Embree ray stream with generated rays.
                let mut ray_stream = embree::RayN::new(rays.len());
                for (i, mut ray) in ray_stream.iter_mut().enumerate() {
                    ray.set_origin(rays[i].o.into());
                    ray.set_dir(rays[i].d.into());
                }

                // Trace primary rays with coherent context.
                let mut coherent_ctx = embree::IntersectContext::coherent();
                let ray_hit = embree_rt.intersect_stream_soa(scene_id, ray_stream, &mut coherent_ctx);

                // Filter out primary rays that hit the surface.
                let filtered = ray_hit
                    .iter()
                    .enumerate()
                    .filter_map(|(i, (_, h))| h.hit().then(|| i));

                let mut incoherent_ctx = embree::IntersectContext::incoherent();
                let records = filtered
                    .into_iter()
                    .map(|i| {
                        let ray = Ray {
                            o: ray_hit.ray.org(i).into(),
                            d: ray_hit.ray.dir(i).into(),
                            e: 1.0,
                        };
                        trace_one_ray_embree(
                            ray,
                            embree_rt.scene_mut(scene_id),
                            &mut incoherent_ctx,
                            ior_t,
                            desc.emitter.max_bounces,
                            None,
                            None,
                        )
                    })
                    .collect::<Vec<_>>();
                println!("{:?}", records);
            }
        }
    }
}

pub fn measure_in_plane_brdf_grid(desc: &MeasurementDesc, ior_db: &RefractiveIndexDatabase,
                                  surfaces: &[Heightfield])
{
    let collector: Collector = desc.collector.into();
    let emitter: Emitter = desc.emitter.into();
    log::debug!("Emitter generated {} patches.", emitter.patches.len());

    let mut embree_rt = EmbreeRayTracing::new(Config::default());

    for surface in surfaces {
        let scene_id = embree_rt.create_scene();
        let triangulated = surface.triangulate(TriangulationMethod::Regular);
        let radius = triangulated.extent.max_edge() * 2.5;
        let surface_mesh = embree_rt.create_triangle_mesh(&triangulated);
        let surface_id = embree_rt.attach_geometry(scene_id, surface_mesh);
        let spectrum_samples = SpectrumSampler::from(desc.emitter.spectrum).samples();
        let grid_rt = GridRayTracing::new(surface, &triangulated);
        log::debug!("Grid - min: {}, max: {} | origin: {:?}", grid_rt.min, grid_rt.max, grid_rt.origin);
        // let ior_i = ior_db
        //     .ior_of_spectrum(desc.incident_medium, &spectrum_samples)
        //     .unwrap();
        // let ior_t = ior_db
        //     .ior_of_spectrum(desc.transmitted_medium, &spectrum_samples)
        //     .unwrap();

        for wavelength in spectrum_samples {
            println!("Capturing with wavelength = {}", wavelength);
            let ior_t = ior_db.refractive_index_of(desc.transmitted_medium, wavelength).unwrap();

            // For all incident angles; generate samples on each patch
            for (i, patch) in emitter.patches.iter().enumerate() {
                // Emit rays from the patch of the emitter. Uniform sampling over the patch.
                let rays = patch.emit_rays(desc.emitter.num_rays, radius);
                log::debug!("Emitted {} rays from patch {} - {:?}: {:?}", rays.len(), i, patch, rays);

                // Populate Embree ray stream with generated rays.
                let mut ray_stream = embree::RayN::new(rays.len());
                for (i, mut ray) in ray_stream.iter_mut().enumerate() {
                    ray.set_origin(rays[i].o.into());
                    ray.set_dir(rays[i].d.into());
                }

                // Trace primary rays with coherent context.
                let mut coherent_ctx = embree::IntersectContext::coherent();
                let ray_hit = embree_rt.intersect_stream_soa(scene_id, ray_stream, &mut coherent_ctx);

                // Filter out primary rays that hit the surface.
                let filtered = ray_hit
                    .iter()
                    .enumerate()
                    .filter_map(|(i, (_, h))| h.hit().then(|| i));

                let records = filtered
                    .into_iter()
                    .map(|i| {
                        let ray = Ray {
                            o: ray_hit.ray.org(i).into(),
                            d: ray_hit.ray.dir(i).into(),
                            e: 1.0,
                        };
                        trace_one_ray_grid_tracing(ray, &grid_rt, ior_t, None)
                    })
                    .collect::<Vec<_>>();
                println!("{:?}", records);
            }
        }
    }
}

// Approach 1: sort filtered rays to continue take advantage of
// coherent tracing
// Approach 2: trace each filtered ray with incoherent context
// Approach 3: using heightfield tracing method to trace rays

/// Structure to sample over a spectrum.
pub(crate) struct SpectrumSampler {
    range: Range<f32>,
    num_samples: usize,
}

impl From<Range<f32>> for SpectrumSampler {
    fn from(range: Range<f32>) -> Self {
        let num_samples = ((range.stop - range.start) / range.step) as usize + 1;
        Self { range, num_samples }
    }
}

impl SpectrumSampler {
    /// Returns the nth wavelength of the spectrum.
    pub fn nth_sample(&self, n: usize) -> f32 {
        self.range.start + self.range.step * n as f32
    }

    /// Returns the spectrum's whole wavelength range.
    pub fn samples(&self) -> Vec<f32> {
        (0..self.num_samples).map(|i| self.nth_sample(i)).collect()
    }
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
fn trace_one_ray_embree(
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
        let mut ray_hit = RayHit::new(ray.into_embree_ray());
        scene.intersect(context, &mut ray_hit);

        // Does the hit something?
        if ray_hit.hit.hit() && (f32::EPSILON..f32::INFINITY).contains(&ray_hit.ray.tfar) {
            // It's not the first time that the ray has hit the surface.
            // Is it the first time that the ray hit some geometry?
            if let Some(prev_isect) = prev_isect {
                // Is the ray repeat hitting the same primitive of the same geometry?
                if prev_isect.geom_id == ray_hit.hit.geomID
                    && prev_isect.prim_id == ray_hit.hit.primID
                {
                    // Self intersection happens, recalculate the hit point.
                    let nudged_times = prev_isect.nudged_times + 1;
                    let amount = NUDGE_AMOUNT * (nudged_times * nudged_times) as f32 * 0.5;
                    let hit_point =
                        nudge_hit_point(prev_isect.hit_point, prev_isect.normal, amount);
                    // Update the intersection record.
                    let curr_isect = IntersectRecord {
                        hit_point,
                        nudged_times,
                        ..prev_isect
                    };
                    // Recalculate the reflected ray.
                    if let Some(Scattering { reflected, .. }) = scattering_air_conductor(
                        prev_isect.ray,
                        hit_point,
                        prev_isect.normal,
                        ior_t.eta,
                        ior_t.k,
                    ) {
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
                // Compute the new reflected ray.
                if let Some(Scattering { reflected, .. }) =
                    scattering_air_conductor(ray, hit_point, normal, ior_t.eta, ior_t.k)
                {
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

fn trace_one_ray_grid_tracing(
    ray: Ray,
    rt_grid: &GridRayTracing,
    ior_t: RefractiveIndex,
    record: Option<RayTraceRecord>,
) -> Option<RayTraceRecord> {
    if let Some(isect) = rt_grid.trace_ray(ray) {
        if let Some(Scattering { reflected, .. }) =
            scattering_air_conductor(ray, isect.hit_point, isect.normal, ior_t.eta, ior_t.k)
        {
            if reflected.e >= 0.0 {
                let curr_record = RayTraceRecord {
                    initial: record.as_ref().unwrap().initial,
                    current: ray,
                    bounces: record.as_ref().unwrap().bounces + 1,
                };
                trace_one_ray_grid_tracing(reflected, rt_grid, ior_t, Some(curr_record))
            } else {
                record
            }
        } else {
            record
        }
    } else {
        record
    }
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

/// Nudge the hit point along the normal to avoid self-intersection.
fn nudge_hit_point(hit_point: Vec3, normal: Vec3, amount: f32) -> Vec3 {
    hit_point + normal * amount
}

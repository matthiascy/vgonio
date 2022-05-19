use crate::acq::desc::{MeasurementDesc, Range};
use crate::acq::ior::{RefractiveIndex, RefractiveIndexDatabase};
use crate::acq::ray::{
    fresnel_scattering_air_conductor, fresnel_scattering_air_conductor_spectrum, Ray,
    RayTraceRecord, Scattering,
};
use crate::acq::{Collector, Emitter, Patch};
use crate::htfld::{regular_triangulation, Heightfield};
use embree::{Geometry, RayHit, SoARay};
use glam::{Vec2, Vec3};
use std::sync::Arc;
use crate::acq::dda::dda;
use crate::isect::Aabb;

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
pub fn measure_in_plane_brdf<PatchData: Copy, const N_PATCH: usize, const N_BOUNCE: usize>(
    desc: &MeasurementDesc,
    ior_db: &RefractiveIndexDatabase,
    surfaces: &[Heightfield],
) -> Vec<Stats<PatchData, N_PATCH, N_BOUNCE>> {
    let collector: Collector = desc.collector.into();
    let emitter: Emitter = desc.emitter.into();

    let device = embree::Device::with_config(embree::Config::default());

    for surface in surfaces {
        let (vertices, aabb) = surface.generate_vertices();
        let indices = regular_triangulation(&vertices, surface.rows, surface.cols);
        let num_tris = indices.len() / 3;
        let mut surface_mesh =
            embree::TriangleMesh::unanimated(device.clone(), num_tris, vertices.len());
        {
            let surface_mesh_mut = Arc::get_mut(&mut surface_mesh).unwrap();
            {
                let mut v_buffer = surface_mesh_mut.vertex_buffer.map();
                let mut i_buffer = surface_mesh_mut.index_buffer.map();
                // TODO: customise embree to use compatible format to fill the buffer
                // Fill vertex buffer with height field vertices.
                for (i, vertex) in vertices.iter().enumerate() {
                    v_buffer[i] = [vertex.x, vertex.y, vertex.z, 1.0];
                }
                // Fill index buffer with height field triangles.
                (0..num_tris).for_each(|i| {
                    i_buffer[i] = [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
                });
            }
            surface_mesh_mut.commit()
        }

        let mut scene = embree::Scene::new(device.clone());
        let scene_mut = Arc::get_mut(&mut scene).unwrap();
        let surface_geom_id = {
            let id = scene_mut.attach_geometry(surface_mesh);
            scene_mut.commit();
            id
        };

        let spectrum_samples = SpectrumSampler::from(desc.emitter.spectrum).samples();
        // let ior_i = ior_db
        //     .ior_of_spectrum(desc.incident_medium, &spectrum_samples)
        //     .unwrap();
        // let ior_t = ior_db
        //     .ior_of_spectrum(desc.transmitted_medium, &spectrum_samples)
        //     .unwrap();

        for wavelength in spectrum_samples {
            let ior_t = ior_db.ior_of(desc.transmitted_medium, wavelength).unwrap();

            // For all incident angles; generate samples on each patch
            for patch in &emitter.patches {
                // Emit rays from the patch of the emitter. Uniform sampling over the patch.
                let rays = emitter.emit_from_patch(patch);

                // Populate Embree ray stream with generated rays.
                let mut ray_stream = embree::RayN::new(rays.len());
                for (i, mut ray) in ray_stream.iter_mut().enumerate() {
                    ray.set_origin(rays[i].o.into());
                    ray.set_dir(rays[i].d.into());
                }

                // Trace primary rays with coherent context.
                let mut context = embree::IntersectContext::coherent();
                let mut ray_hit = embree::RayHitN::new(ray_stream);
                scene.intersect_stream_soa(&mut context, &mut ray_hit);

                // Filter out primary rays that hit the surface.
                let filtered = ray_hit
                    .iter()
                    .enumerate()
                    .filter_map(|(i, (_, h))| h.hit().then(|| i));

                // Approach 1: sort filtered rays to continue take advantage of
                // coherent tracing

                // Approach 2: trace each filtered ray with incoherent context
                let mut incoherent_context = embree::IntersectContext::incoherent();
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
                            Arc::get_mut(&mut scene).unwrap(),
                            &mut incoherent_context,
                            ior_t,
                            desc.emitter.max_bounces,
                            None,
                            None,
                        )
                    })
                    .collect::<Vec<_>>();

                // Approach 3: using heightfield tracing method to trace rays
            }
        }
    }

    vec![]
}

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
struct IntersectRecord {
    /// The ray that hit the surface.
    ray: Ray,

    /// The surface that was hit.
    geom_id: u32,

    /// The primitive of the surface that was hit.
    prim_id: u32,

    /// The hit point.
    hit_point: Vec3,

    /// The normal at the hit point.
    normal: Vec3,

    /// How many times the ray has been rewound.
    nudged_times: u32,
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
                    if let Some(Scattering { reflected, .. }) = fresnel_scattering_air_conductor(
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
                        fresnel_scattering_air_conductor(ray, hit_point, normal, ior_t.eta, ior_t.k)
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
                    fresnel_scattering_air_conductor(ray, hit_point, normal, ior_t.eta, ior_t.k)
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

fn trace_one_ray_grid_tracing(ray: Ray, hf: Heightfield, aabb: Aabb, prev_record: Option<RayTraceRecord>) -> RayTraceRecord {
    // Calculate the x, y, z coordinates of the intersection of the ray with
    // the global bounding box of the height field.
    let entering_point = aabb.intersect(ray, f32::NEG_INFINITY, f32::INFINITY);

    // 2. Use standard DDA to traverse the grid in x, y coordinates
    // until the ray exits the bounding box. Identify all cells traversed by the ray.
    let (cells, isects) = dda(entering_point.)

    // 4. Modify the DDA to track the z
    // (altitude) values of the endpoints of the ray within each cell.
    // 5. Test the z values of the ray at each cell against the altitudes at the
    // four corners of the cell.    If this test indicates that the ray
    // passes between those altitudes at that cell, proceed with the steps
    // below;    otherwise, go to the next cell.
    // 6. Create two triangles (only the plane description is needed) from the
    // four altitudes at the corners of the cell. 7. Intersect the ray with
    // the two triangles which tessellate the surface within the cell.
    // 8. Include an inverse skewing and scaling transformation for the ray, so
    // the triangles may be equilateral rather than right triangles.
    todo!()
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
        for i in 0..3 {
            let idx = *indices.offset(prim_id * 3 + i as isize) as isize;
            points[i] = Vec3::new(
                *vertices.offset(idx * 4),
                *vertices.offset(idx * 4 + 1),
                *vertices.offset(idx * 4 + 2),
            )
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

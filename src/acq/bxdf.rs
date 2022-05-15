use crate::acq::desc::{MeasurementDesc, Range};
use crate::acq::ior::RefractiveIndexDatabase;
use crate::acq::{Collector, Emitter, Patch};
use crate::htfld::{regular_triangulation, Heightfield};
use crate::acq::ray::{Ray, RayTraceRecord};

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
    let mut scene = embree::Scene::new(device.clone());

    for surface in surfaces {
        let (vertices, _) = surface.generate_vertices();
        let indices = regular_triangulation(&vertices, surface.rows, surface.cols);
        let num_triangles = indices.len() / 3;
        let num_vertices = vertices.len();
        let mut tri_mesh =
            embree::TriangleMesh::unanimated(device.clone(), num_triangles, num_vertices);
        {
            let mut v_buffer = tri_mesh.vertex_buffer.map();
            let mut i_buffer = tri_mesh.index_buffer.map();
            // TODO: customise embree to use compatible format to fill the buffer
            // Fill vertex buffer with height field vertices.
            for (i, vertex) in vertices.iter().enumerate() {
                v_buffer[i] = [vertex.x, vertex.y, vertex.z, 1.0];
            }
            // Fill index buffer with height field triangles.
            (0..num_triangles).for_each(|i| {
                i_buffer[i] = [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]];
            });
        }

        scene.attach_geometry(tri_mesh);
        scene.commit();

        let spectrum_sampler = SpectrumSampler::from(desc.emitter.spectrum);

        // For all wavelengths
        for wavelength in spectrum_sampler.samples() {
            let ior_i = ior_db.ior_of(desc.incident_medium, wavelength).unwrap();
            let ior_t = ior_db.ior_of(desc.transmitted_medium, wavelength).unwrap();

            println!(
                "    > Tracing rays of {}, refractive index of transmitted medium is: \
            η = {:.3}, k = {:.3}",
                wavelength, ior_t.eta, ior_t.k
            );

            // For all incident angles; generate samples on each patch
            for patch in emitter.patches {
                // Emit rays from the patch of the emitter. Uniform sampling over the patch.
                let rays = emitter.emit_from_patch(&patch);

                // Populate Embree ray stream with generated rays.
                let mut ray_stream = embree::RayN::new(rays.len());
                for (i, mut ray) in ray_stream.iter_mut().enumerate() {
                    ray.set_origin(rays[i].o.into());
                    ray.set_dir(rays[i].d.into());
                }

                // Trace primary rays with coherent context.
                let mut isect_ctx = embree::IntersectContext::coherent();
                let mut ray_hit = embree::RayHitN::new(ray_stream);
                scene.intersect_stream_soa(&mut isect_ctx, &mut ray_hit);

                // Filter out primary rays that hit the surface.

                // Approach 1: sort filtered rays to continue take advantage of coherent tracing

                // Approach 2: trace each filtered ray with incoherent context

                // Approach 3: using heightfield tracing method to trace rays
            }
        }
    }

    vec![]
}

/// Structure to sample over a spectrum.
pub(crate) struct SpectrumSampler {
    pub range: Range<f32>,
    pub num_samples: usize,
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

// rtcIntersect/rtcOccluded calls can be invoked from multiple threads.
// embree api calls to the same object are not thread safe in general.
fn trace_one_ray_embree(
    scene: &mut embree::Scene,
    ray: &mut embree::Ray,
    isect_ctx: &mut embree::IntersectContext,
    ray_hit: &mut embree::RayHit,
) -> RayTraceRecord {
    let mut record = RayTraceRecord {
        ray: Ray::from_embree_ray(ray, 1.0),
        bounces: 0
    };
}

fn trace_one_ray_htfld_rt() -> RayTraceRecord {
    let mut record = RayTraceRecord {
        ray: Ray {},
        bounces: 0
    };
}
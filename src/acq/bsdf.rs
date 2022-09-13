use crate::{
    acq::{
        desc::{MeasurementDesc, Range},
        Collector, Emitter, Patch, Ray,
    },
    app::{
        cache::{SurfaceHandle, VgonioCache, VgonioDatafiles},
        BRIGHT_YELLOW, RESET,
    },
};
use embree::{Config, SoARay};
use std::fmt::{Display, Formatter};

/// Type of the BSDF to be measured.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")] // TODO: use case_insensitive in the future
pub enum BsdfKind {
    /// In-plane BRDF.
    InPlaneBrdf,

    /// Whole-hemisphere BRDF.
    Brdf,
}

impl Display for BsdfKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BsdfKind::InPlaneBrdf => {
                write!(f, "in-plane brdf")
            }
            BsdfKind::Brdf => {
                write!(f, "brdf")
            }
        }
    }
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
pub fn measure_bsdf_embree_rt(
    desc: &MeasurementDesc,
    cache: &VgonioCache,
    db: &VgonioDatafiles,
    surfaces: &[SurfaceHandle],
) {
    use crate::acq::EmbreeRayTracing;
    let mut collector: Collector = desc.collector.into();
    let mut emitter: Emitter = desc.emitter.into();
    let mut embree_rt = EmbreeRayTracing::new(Config::default());
    let (surfaces, meshes) = {
        (
            cache.get_surfaces(surfaces).unwrap(),
            cache.get_surface_meshes(surfaces),
        )
    };

    // Iterate over every surface.
    for (surface, mesh) in surfaces.iter().zip(meshes.iter()) {
        println!(
            "    {BRIGHT_YELLOW}>{RESET} Measure surface {}",
            surface.path.as_ref().unwrap().display()
        );
        let scene_id = embree_rt.create_scene();
        // Update emitter's radius
        if desc.emitter.radius.is_auto() {
            emitter.set_radius(mesh.extent.max_edge() * 2.5);
        }
        let embree_mesh = embree_rt.create_triangle_mesh(mesh);
        let _surface_id = embree_rt.attach_geometry(scene_id, embree_mesh);
        let spectrum = SpectrumSampler::from(desc.emitter.spectrum).samples();
        log::debug!("spectrum samples: {:?}", spectrum);

        let ior_t = db
            .ior_db
            .ior_of_spectrum(desc.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");

        // Iterate over every incident direction.
        for pos in emitter.positions() {
            let rays = emitter.emit_rays(pos);
            log::debug!(
                "emitted {} rays with direction {} from position {} {}",
                rays.len(),
                rays[0].d,
                pos.zenith,
                pos.azimuth
            );

            // Populate embree ray stream with generated rays.
            let mut ray_stream = embree::RayN::new(rays.len());
            for (i, mut ray) in ray_stream.iter_mut().enumerate() {
                ray.set_origin(rays[i].o.into());
                ray.set_dir(rays[i].d.into());
            }

            // Trace primary rays with coherent context
            let mut coherent_ctx = embree::IntersectContext::coherent();
            let ray_hit = embree_rt.intersect_stream_soa(scene_id, ray_stream, &mut coherent_ctx);

            // Filter out primary rays (index) that hit the surface.
            let filtered: Vec<_> = ray_hit
                .iter()
                .enumerate()
                .filter_map(|(i, (_, h))| h.hit().then_some(i))
                .collect();

            println!("first hit count: {}", filtered.len());

            let records = filtered.iter().filter_map(|i| {
                let ray = Ray::new(ray_hit.ray.org(*i).into(), ray_hit.ray.dir(*i).into());
                embree_rt.trace_one_ray(scene_id, ray, desc.emitter.max_bounces, &ior_t)
            });

            collector.collect(records, BsdfKind::InPlaneBrdf);
        }
    }
}

/// Brdf measurement.
pub fn measure_bsdf_grid_rt(
    desc: &MeasurementDesc,
    cache: &VgonioCache,
    db: &VgonioDatafiles,
    surfaces: &[SurfaceHandle],
) {
    use crate::acq::GridRayTracing;

    let collector: Collector = desc.collector.into();
    let mut emitter: Emitter = desc.emitter.into();
    let (surfaces, meshes) = {
        (
            cache.get_surfaces(surfaces).unwrap(),
            cache.get_surface_meshes(surfaces),
        )
    };

    for (surface, mesh) in surfaces.iter().zip(meshes.iter()) {
        let mut grid_rt = GridRayTracing::new(surface, mesh);
        println!(
            "    {BRIGHT_YELLOW}>{RESET} Measure surface {}",
            surface.path.as_ref().unwrap().display()
        );
        if desc.emitter.radius.is_auto() {
            emitter.set_radius(mesh.extent.max_edge() * 2.5);
        }
        let spectrum = SpectrumSampler::from(desc.emitter.spectrum).samples();
        let ior_t = db
            .ior_db
            .ior_of_spectrum(desc.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");

        // for pos in emitter.positions() {
        //     let rays = emitter.emit_rays(pos);
        //     let records = rays.iter().filter_map(|ray| {
        //         grid_rt.trace_one_ray_dbg()
        //         grid_rt.trace_one_ray(ray, desc.emitter.max_bounces, &ior_t)
        //     });
        //     collector.collect(records, BsdfKind::InPlaneBrdf);
        // }
    }
    // log::debug!("Emitter has {} patches.", emitter.patches.len());
    // let mut grid_rt = GridRayTracing::new(Config::default());
}

// pub fn measure_in_plane_brdf_grid(
//     desc: &MeasurementDesc,
//     ior_db: &RefractiveIndexDatabase,
//     surfaces: &[Heightfield],
// ) {
//     let collector: Collector = desc.collector.into();
//     let emitter: Emitter = desc.emitter.into();
//     log::debug!("Emitter generated {} patches.", emitter.patches.len());
//
//     let mut embree_rt = EmbreeRayTracing::new(Config::default());
//
//     for surface in surfaces {
//         let scene_id = embree_rt.create_scene();
//         let triangulated = surface.triangulate(TriangulationMethod::Regular);
//         let radius = triangulated.extent.max_edge() * 2.5;
//         let surface_mesh = embree_rt.create_triangle_mesh(&triangulated);
//         let surface_id = embree_rt.attach_geometry(scene_id, surface_mesh);
//         let spectrum_samples =
// SpectrumSampler::from(desc.emitter.spectrum).samples();         let grid_rt =
// GridRayTracing::new(surface, &triangulated);         log::debug!(
//             "Grid - min: {}, max: {} | origin: {:?}",
//             grid_rt.min,
//             grid_rt.max,
//             grid_rt.origin
//         );
//         // let ior_i = ior_db
//         //     .ior_of_spectrum(desc.incident_medium, &spectrum_samples)
//         //     .unwrap();
//         // let ior_t = ior_db
//         //     .ior_of_spectrum(desc.transmitted_medium, &spectrum_samples)
//         //     .unwrap();
//
//         for wavelength in spectrum_samples {
//             println!("Capturing with wavelength = {}", wavelength);
//             let ior_t = ior_db
//                 .refractive_index_of(desc.transmitted_medium, wavelength)
//                 .unwrap();
//
//             // For all incident angles; generate samples on each patch
//             for (i, patch) in emitter.patches.iter().enumerate() {
//                 // Emit rays from the patch of the emitter. Uniform sampling
// over the patch.                 let rays =
// patch.emit_rays(desc.emitter.num_rays, radius);                 log::debug!(
//                     "Emitted {} rays from patch {} - {:?}: {:?}",
//                     rays.len(),
//                     i,
//                     patch,
//                     rays
//                 );
//
//                 // Populate Embree ray stream with generated rays.
//                 let mut ray_stream = embree::RayN::new(rays.len());
//                 for (i, mut ray) in ray_stream.iter_mut().enumerate() {
//                     ray.set_origin(rays[i].o.into());
//                     ray.set_dir(rays[i].d.into());
//                 }
//
//                 // Trace primary rays with coherent context.
//                 let mut coherent_ctx = embree::IntersectContext::coherent();
//                 let ray_hit =
//                     embree_rt.intersect_stream_soa(scene_id, ray_stream, &mut
// coherent_ctx);
//
//                 // Filter out primary rays that hit the surface.
//                 let filtered = ray_hit
//                     .iter()
//                     .enumerate()
//                     .filter_map(|(i, (_, h))| h.hit().then(|| i));
//
//                 let records = filtered
//                     .into_iter()
//                     .map(|i| {
//                         let ray = Ray {
//                             o: ray_hit.ray.org(i).into(),
//                             d: ray_hit.ray.dir(i).into(),
//                             e: 1.0,
//                         };
//                         trace_one_ray_grid_tracing(ray, &grid_rt, ior_t,
// None)                     })
//                     .collect::<Vec<_>>();
//                 println!("{:?}", records);
//             }
//         }
//     }
// }

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
    pub fn nth_sample(&self, n: usize) -> f32 { self.range.start + self.range.step * n as f32 }

    /// Returns the spectrum's whole wavelength range.
    pub fn samples(&self) -> Vec<f32> {
        (0..self.num_samples).map(|i| self.nth_sample(i)).collect()
    }
}

// fn trace_one_ray_grid_tracing(
//     ray: Ray,
//     rt_grid: &GridRayTracing,
//     ior_t: RefractiveIndex,
//     record: Option<RayTraceRecord>,
// ) -> Option<RayTraceRecord> {
//     if let Some(isect) = rt_grid.trace_ray(ray) {
//         if let Some(Scattering { reflected, .. }) =
//             scattering_air_conductor(ray, isect.hit_point, isect.normal,
// ior_t.eta, ior_t.k)         {
//             if reflected.e >= 0.0 {
//                 let curr_record = RayTraceRecord {
//                     initial: record.as_ref().unwrap().initial,
//                     current: ray,
//                     bounces: record.as_ref().unwrap().bounces + 1,
//                 };
//                 trace_one_ray_grid_tracing(reflected, rt_grid, ior_t,
// Some(curr_record))             } else {
//                 record
//             }
//         } else {
//             record
//         }
//     } else {
//         record
//     }
// }

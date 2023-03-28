use crate::{
    app::{
        cache::{Cache, Handle},
        cli::{BRIGHT_CYAN, BRIGHT_YELLOW, RESET},
    },
    common::RangeByStepSize,
    measure::{measurement::Radius, rtc::grid::GridRT, Collector, CollectorScheme, Emitter, Patch},
    msurf::MicroSurface,
    units::{metres, mm, Nanometres},
};
use serde::{ser::SerializeTupleStruct, Deserialize, Serialize, Serializer};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Deref, DerefMut},
};

use super::measurement::BsdfMeasurement;

/// Type of the BSDF to be measured.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")] // TODO: use case_insensitive in the future
pub enum BsdfKind {
    /// Bidirectional reflectance distribution function.
    Brdf,

    /// Bidirectional transmittance distribution function.
    Btdf,

    /// Bidirectional scattering-surface distribution function.
    Bssdf,

    /// Bidirectional scattering-surface reflectance distribution function.
    Bssrdf,

    /// Bidirectional scattering-surface transmittance distribution function.
    Bsstdf,
}

impl Display for BsdfKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BsdfKind::Brdf => {
                write!(f, "brdf")
            }
            BsdfKind::Btdf => {
                write!(f, "btdf")
            }
            BsdfKind::Bssdf => {
                write!(f, "bssdf")
            }
            BsdfKind::Bssrdf => {
                write!(f, "bssrdf")
            }
            BsdfKind::Bsstdf => {
                write!(f, "bsstdf")
            }
        }
    }
}

#[derive(Debug)]
pub struct PerWavelength<T>(pub Vec<T>);

impl<T> Clone for PerWavelength<T>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T> PerWavelength<T> {
    pub fn empty() -> Self { Self(Vec::new()) }
}

impl<T> Deref for PerWavelength<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> DerefMut for PerWavelength<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

#[derive(Debug, Clone)]
pub struct BsdfStats {
    /// Number of emitted rays; invariant over wavelength.
    pub n_emitted: u32,

    /// Number of emitted rays that hit the surface; invariant over wavelength.
    pub n_received: u32,

    /// Wavelengths of emitted rays.
    pub wavelength: Vec<Nanometres>,

    /// Number of emitted rays that hit the surface and were absorbed;
    pub n_absorbed: PerWavelength<u32>,

    /// Number of emitted rays that hit the surface and were reflected.
    pub n_reflected: PerWavelength<u32>,

    /// Number of emitted rays captured by the collector.
    pub n_captured: PerWavelength<u32>,

    /// Initial energy of emitted rays per wavelength; invariant over
    /// wavelength.
    pub total_energy_emitted: f32,

    /// Energy captured by the collector; variant over wavelength.
    pub total_energy_captured: PerWavelength<f32>,

    /// Histogram of reflected rays by number of bounces, variant over
    /// wavelength.
    pub num_rays_per_bounce: PerWavelength<Vec<u32>>,

    /// Histogram of energy of reflected rays by number of bounces, variant
    /// over wavelength.
    pub energy_per_bounce: PerWavelength<Vec<f32>>,
}

/// Bsdf measurement point and its data.
pub struct BsdfMeasurementPoint<PerPatchData> {
    /// Patch containing the measurement point.
    pub patch: Patch,

    /// Per patch data. This is used to either store the measured data for each
    /// patch in case the collector is partitioned or for the collector's
    /// region at different places in the scene. You need to check the collector
    /// to know which one is the case and how to interpret the data.
    pub data: PerWavelength<PerPatchData>,
}

impl<PerPatchData: Debug> Debug for BsdfMeasurementPoint<PerPatchData> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BsdfMeasurementPoint")
            .field("patch", &self.patch)
            .field("data", &self.data)
            .finish()
    }
}

impl<PerPatchData: Clone> Clone for BsdfMeasurementPoint<PerPatchData> {
    fn clone(&self) -> Self {
        Self {
            patch: self.patch.clone(),
            data: self.data.clone(),
        }
    }
}

/// Measurement of the BSDF (bidirectional scattering distribution function) of
/// a microfacet surface.
#[cfg(feature = "embree")]
pub fn measure_bsdf_embree_rt(
    mut desc: BsdfMeasurement,
    cache: &Cache,
    surfaces: &[Handle<MicroSurface>],
) {
    let msurfs = cache.micro_surfaces(surfaces).unwrap();
    let meshes = cache.micro_surface_meshes_by_surfaces(surfaces).unwrap();
    let emitter_samples = desc.emitter.generate_samples();
    let collector_patches = desc.collector.generate_patches();

    for (surf, mesh) in msurfs.iter().zip(meshes.iter()) {
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Measure surface {}",
            surf.path.as_ref().unwrap().display()
        );
        let t = std::time::Instant::now();
        crate::measure::rtc::embr::measure_bsdf(
            &desc,
            mesh,
            cache,
            &emitter_samples,
            &collector_patches,
        );
        println!(
            "        {BRIGHT_CYAN}✓{RESET} Done in {:?} s",
            t.elapsed().as_secs_f32()
        );
    }
}

/// Brdf measurement of a microfacet surface using the grid ray tracing.
pub fn measure_bsdf_grid_rt(
    desc: BsdfMeasurement,
    cache: &Cache,
    surfaces: &[Handle<MicroSurface>],
) {
    let msurfs = cache.micro_surfaces(surfaces).unwrap();
    let meshes = cache.micro_surface_meshes_by_surfaces(surfaces).unwrap();

    for (surf, mesh) in msurfs.iter().zip(meshes.iter()) {
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Measure surface {}",
            surf.path.as_ref().unwrap().display()
        );
        let t = std::time::Instant::now();
        crate::measure::rtc::grid::measure_bsdf(&desc, surf, mesh, cache);
        println!(
            "        {BRIGHT_CYAN}✓{RESET} Done in {:?} s",
            t.elapsed().as_secs_f32()
        );
    }
}

/// Brdf measurement of a microfacet surface using the OptiX ray tracing.
#[cfg(feature = "optix")]
pub fn measure_bsdf_optix_rt(
    _desc: BsdfMeasurement,
    _cache: &Cache,
    _surfaces: &[Handle<MicroSurface>],
) {
    todo!()
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
    range: RangeByStepSize<Nanometres>,
    num_samples: usize,
}

impl From<RangeByStepSize<Nanometres>> for SpectrumSampler {
    fn from(range: RangeByStepSize<Nanometres>) -> Self {
        let num_samples = ((range.stop - range.start) / range.step_size) as usize + 1;
        Self { range, num_samples }
    }
}

impl SpectrumSampler {
    /// Returns the nth wavelength of the spectrum.
    pub fn nth_sample(&self, n: usize) -> Nanometres {
        self.range.start + self.range.step_size * n as f32
    }

    /// Returns the spectrum's whole wavelength range.
    pub fn samples(&self) -> Vec<Nanometres> {
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
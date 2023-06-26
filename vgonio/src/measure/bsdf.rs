//! Measurement of the BSDF (bidirectional scattering distribution function) of
//! micro-surfaces.

#[cfg(feature = "embree")]
use crate::measure::rtc::embr;
use crate::{
    app::{
        cache::{Cache, Handle},
        cli::{BRIGHT_CYAN, BRIGHT_YELLOW, RESET},
    },
    measure::{
        collector::{BounceAndEnergy, CollectorPatches, PerPatchData},
        emitter::EmitterSamples,
        measurement::{MeasuredData, MeasurementData, MeasurementDataSource, SimulationKind},
        rtc::RayTrajectory,
        RtcMethod,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Deref, DerefMut},
};
use vgcore::math::Vec3;
use vgsurf::{MicroSurface, MicroSurfaceMesh};

use super::measurement::BsdfMeasurementParams;

/// BSDF measurement data.
///
/// Number of emitted rays, wavelengths, and bounces are invariant over
/// emitter's position.
///
/// At each emitter's position, each emitted ray carries an initial energy
/// equals to 1.
#[derive(Debug, Clone)]
pub struct MeasuredBsdfData {
    /// Parameters of the measurement.
    pub params: BsdfMeasurementParams,
    /// The BSDF data per emitter's position (incident direction). The first
    /// index is the azimuthal angle, and the second index is the zenith angle.
    /// The order of the data point follows the order of collector's patches.
    pub samples: Vec<BsdfMeasurementDataPoint<BounceAndEnergy>>,
}

/// Type of the BSDF to be measured.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BsdfKind {
    /// Bidirectional reflectance distribution function.
    Brdf = 0x00,

    /// Bidirectional transmittance distribution function.
    Btdf = 0x01,

    /// Bidirectional scattering-surface distribution function.
    Bssdf = 0x02,

    /// Bidirectional scattering-surface reflectance distribution function.
    Bssrdf = 0x03,

    /// Bidirectional scattering-surface transmittance distribution function.
    Bsstdf = 0x04,
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

impl From<u8> for BsdfKind {
    fn from(value: u8) -> Self {
        match value {
            0x00 => BsdfKind::Brdf,
            0x01 => BsdfKind::Btdf,
            0x02 => BsdfKind::Bssdf,
            0x03 => BsdfKind::Bssrdf,
            0x04 => BsdfKind::Bsstdf,
            _ => panic!("Invalid BSDF kind: {}", value),
        }
    }
}

/// Stores the data per wavelength for a spectrum.
#[derive(Debug)]
pub struct PerWavelength<T>(pub Vec<T>);

impl<T> Clone for PerWavelength<T>
where
    T: Clone,
{
    fn clone(&self) -> Self { Self(self.0.clone()) }
}

impl<T> PerWavelength<T> {
    /// Creates a new empty `PerWavelength`.
    pub fn empty() -> Self { Self(Vec::new()) }
}

impl<T> Deref for PerWavelength<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T> DerefMut for PerWavelength<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T: PartialEq> PartialEq for PerWavelength<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
    }
}

/// BSDF measurement statistics for a single emitter's position.
#[derive(Clone)]
pub struct BsdfMeasurementStatsPoint {
    /// Number of emitted rays that hit the surface; invariant over wavelength.
    pub n_received: u32,
    /// Number of emitted rays that hit the surface and were absorbed;
    pub n_absorbed: PerWavelength<u32>,
    /// Number of emitted rays that hit the surface and were reflected.
    pub n_reflected: PerWavelength<u32>,
    /// Number of emitted rays captured by the collector.
    pub n_captured: PerWavelength<u32>,
    /// Energy captured by the collector; variant over wavelength.
    pub e_captured: PerWavelength<f32>,
    /// Histogram of reflected rays by number of bounces, variant over
    /// wavelength.
    pub num_rays_per_bounce: PerWavelength<Vec<u32>>,
    /// Histogram of energy of reflected rays by number of bounces, variant
    /// over wavelength.
    pub energy_per_bounce: PerWavelength<Vec<f32>>,
}

impl PartialEq for BsdfMeasurementStatsPoint {
    fn eq(&self, other: &Self) -> bool {
        self.n_received == other.n_received
            && self.n_absorbed == other.n_absorbed
            && self.n_reflected == other.n_reflected
            && self.n_captured == other.n_captured
            && self.e_captured == other.e_captured
            && self.num_rays_per_bounce == other.num_rays_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}

impl BsdfMeasurementStatsPoint {
    /// Creates an empty `BsdfMeasurementStatsPoint`.
    ///
    /// # Arguments
    /// * `n_wavelengths`: Number of wavelengths.
    /// * `max_bounces`: Maximum number of bounces.
    pub fn new(n_wavelengths: usize, max_bounces: usize) -> Self {
        Self {
            n_received: 0,
            n_absorbed: PerWavelength(vec![0; n_wavelengths]),
            n_reflected: PerWavelength(vec![0; n_wavelengths]),
            n_captured: PerWavelength(vec![0; n_wavelengths]),
            e_captured: PerWavelength(vec![0.0; n_wavelengths]),
            num_rays_per_bounce: PerWavelength(vec![vec![0; max_bounces]; n_wavelengths]),
            energy_per_bounce: PerWavelength(vec![vec![0.0; max_bounces]; n_wavelengths]),
        }
    }

    /// Calculates the size in bytes of the `BsdfMeasurementStatsPoint` for the
    /// given number of wavelengths and maximum number of bounces.
    pub fn calc_size_in_bytes(n_wavelength: usize, bounces: usize) -> usize {
        4 + (n_wavelength * 4) * 4 + (n_wavelength * bounces * 4) * 2
    }
}

impl Default for BsdfMeasurementStatsPoint {
    fn default() -> Self { Self::new(0, 0) }
}

impl Debug for BsdfMeasurementStatsPoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"BsdfMeasurementPointStats:
    - n_received: {},
    - n_absorbed: {:?},
    - n_reflected: {:?},
    - n_captured: {:?},
    - total_energy_captured: {:?},
    - num_rays_per_bounce: {:?},
    - energy_per_bounce: {:?},
"#,
            self.n_received,
            self.n_absorbed,
            self.n_reflected,
            self.n_captured,
            self.e_captured,
            self.num_rays_per_bounce,
            self.energy_per_bounce
        )
    }
}

/// Measurement data for a single emitter's position.
///
/// It contains the statistics of the measurement and the data collected
/// for all the patches of the collector.
#[derive(Clone)]
pub struct BsdfMeasurementDataPoint<PatchData>
where
    PatchData: PerPatchData,
{
    /// Statistics of the measurement at the point.
    pub stats: BsdfMeasurementStatsPoint,
    /// A list of data collected for each patch of the collector.
    /// Per patch data. This is used to either store the measured data for each
    /// patch in case the collector is partitioned or for the collector's
    /// region at different places in the scene. You need to check the collector
    /// to know which one is the case and how to interpret the data.
    pub data: Vec<PerWavelength<PatchData>>,
    /// Extra ray trajectory data for debugging purposes.
    #[cfg(debug_assertions)]
    pub trajectories: Vec<RayTrajectory>,
    /// Hit points on the collector.
    #[cfg(debug_assertions)]
    pub hit_points: Vec<Vec3>,
}

impl<PatchData: Debug + PerPatchData> Debug for BsdfMeasurementDataPoint<PatchData> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BsdfMeasurementPoint")
            .field("stats", &self.stats)
            .field("data", &self.data)
            .finish()
    }
}

impl<PatchData: PerPatchData + PartialEq> PartialEq for BsdfMeasurementDataPoint<PatchData> {
    fn eq(&self, other: &Self) -> bool { self.stats == other.stats && self.data == other.data }
}

/// Measures the BSDF of a surface using geometric ray tracing methods.
pub fn measure_bsdf_rt(
    params: BsdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    sim_kind: SimulationKind,
    cache: &Cache,
) -> Vec<MeasurementData> {
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = cache.get_micro_surfaces(handles);
    let samples = params.emitter.generate_unit_samples();
    let patches = params.collector.generate_patches();

    match sim_kind {
        SimulationKind::GeomOptics(method) => {
            println!(
                "    {BRIGHT_YELLOW}>{RESET} Measuring {} with geometric optics...",
                params.kind
            );
            match method {
                #[cfg(feature = "embree")]
                RtcMethod::Embree => measure_bsdf_embree_rt(
                    params, handles, &surfaces, &meshes, samples, patches, cache,
                ),
                #[cfg(feature = "optix")]
                RtcMethod::Optix => {
                    measure_bsdf_optix_rt(params, &surfaces, &meshes, samples, patches, cache)
                }
                RtcMethod::Grid => {
                    measure_bsdf_grid_rt(params, &surfaces, &meshes, samples, patches, cache)
                }
            }
        }
        SimulationKind::WaveOptics => {
            println!(
                "    {BRIGHT_YELLOW}>{RESET} Measuring {} with wave optics...",
                params.kind
            );
            todo!("Wave optics simulation is not yet implemented")
        }
    }
}

/// Measurement of the BSDF (bidirectional scattering distribution function) of
/// a microfacet surface.
#[cfg(feature = "embree")]
fn measure_bsdf_embree_rt(
    params: BsdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    surfaces: &[Option<&MicroSurface>],
    meshes: &[Option<&MicroSurfaceMesh>],
    samples: EmitterSamples,
    patches: CollectorPatches,
    cache: &Cache,
) -> Vec<MeasurementData> {
    handles
        .iter()
        .zip(surfaces.iter())
        .zip(meshes)
        .filter_map(|((hdl, surf), mesh)| {
            if surf.is_none() || mesh.is_none() {
                log::debug!("Skipping surface {:?} and its mesh {:?}", surf, mesh);
                return None;
            }
            let surface = surf.unwrap();
            let mesh = mesh.unwrap();
            log::info!(
                "Measuring surface {}",
                surface.path.as_ref().unwrap().display()
            );
            Some(MeasurementData {
                name: surface.file_stem().unwrap().to_owned(),
                source: MeasurementDataSource::Measured(*hdl),
                measured: MeasuredData::Bsdf(embr::measure_full_bsdf(
                    &params, mesh, &samples, &patches, cache,
                )),
            })
        })
        .collect()
}

/// Brdf measurement of a microfacet surface using the grid ray tracing.
fn measure_bsdf_grid_rt(
    params: BsdfMeasurementParams,
    surfaces: &[Option<&MicroSurface>],
    meshes: &[Option<&MicroSurfaceMesh>],
    samples: EmitterSamples,
    patches: CollectorPatches,
    cache: &Cache,
) -> Vec<MeasurementData> {
    for (surf, mesh) in surfaces.iter().zip(meshes.iter()) {
        if surf.is_none() || mesh.is_none() {
            log::debug!("Skipping surface {:?} and its mesh {:?}", surf, mesh);
            continue;
        }
        let surf = surf.unwrap();
        let mesh = mesh.unwrap();
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Measure surface {}",
            surf.path.as_ref().unwrap().display()
        );
        let t = std::time::Instant::now();
        crate::measure::rtc::grid::measure_bsdf(&params, surf, mesh, &samples, &patches, cache);
        println!(
            "        {BRIGHT_CYAN}✓{RESET} Done in {:?} s",
            t.elapsed().as_secs_f32()
        );
    }
    todo!()
}

/// Brdf measurement of a microfacet surface using the OptiX ray tracing.
#[cfg(feature = "optix")]
fn measure_bsdf_optix_rt(
    _params: BsdfMeasurementParams,
    _surfaces: &[Option<&MicroSurface>],
    _meshes: &[Option<&MicroSurfaceMesh>],
    _samples: EmitterSamples,
    _patches: CollectorPatches,
    _cache: &Cache,
) -> Vec<MeasurementData> {
    todo!()
}

// pub fn measure_in_plane_brdf_grid(
//     desc: &MeasurementDesc,
//     ior_db: &RefractiveIndexDatabase,
//     surfaces: &[Heightfield],
// ) { let collector: Collector = desc.collector.into(); let emitter: Emitter =
//   desc.emitter.into(); log::debug!("Emitter generated {} patches.",
//   emitter.patches.len());
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

// fn trace_one_ray_grid_tracing(
//     ray: Ray,
//     rt_grid: &GridRayTracing,
//     ior_t: RefractiveIndex,
//     record: Option<RayTraceRecord>,
// ) -> Option<RayTraceRecord> { if let Some(isect) = rt_grid.trace_ray(ray) {
//   if let Some(Scattering { reflected, .. }) = scattering_air_conductor(ray,
//   isect.hit_point, isect.normal,
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

#[test]
fn test_bsdf_measurement_stats_point() {
    assert_eq!(
        BsdfMeasurementStatsPoint::calc_size_in_bytes(4, 10),
        388,
        "Size of stats point is incorrect."
    );
    assert_eq!(
        BsdfMeasurementStatsPoint::calc_size_in_bytes(4, 100),
        3268,
        "Size of stats point is incorrect."
    );
}

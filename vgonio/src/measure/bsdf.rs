//! Measurement of the BSDF (bidirectional scattering distribution function) of
//! micro-surfaces.

#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::embr;
use crate::{
    app::{
        cache::{Handle, InnerCache},
        cli::{BRIGHT_CYAN, BRIGHT_YELLOW, RESET},
    },
    error::RuntimeError::Image,
    measure::{
        bsdf::{
            detector::{CollectedData, Detector, PerPatchData},
            emitter::Emitter,
            rtc::{RayTrajectory, RtcMethod},
        },
        data::{MeasuredData, MeasurementData, MeasurementDataSource},
        params::SimulationKind,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Deref, DerefMut, Index, IndexMut},
    path::Path,
};
use vgcore::{
    math::{Sph2, Vec3},
    units::rad,
};
use vgsurf::{MicroSurface, MicroSurfaceMesh};

use super::params::BsdfMeasurementParams;

pub mod detector;
pub mod emitter;
pub(crate) mod params;
pub mod rtc;

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
    /// Snapshot of the BSDF at each incident direction of the emitter.
    /// See [`BsdfSnapshot`] for more details.
    pub snapshots: Vec<BsdfSnapshot>,
}

impl MeasuredBsdfData {
    /// Writes the BSDF data to images in exr format. TODO: error handling
    pub fn write_to_images(&self, name: &str, path: &Path) {
        use exr::prelude::*;
        log::info!("Saving BSDF images to {}", path.display());
        const WIDTH: usize = 512;
        const HEIGHT: usize = 512;

        let wavelengths = self.params.emitter.spectrum.values().collect::<Vec<_>>();
        let mut bsdf_samples_per_wavelength = vec![vec![0.0; WIDTH * HEIGHT]; wavelengths.len()];
        let patches = self.params.detector.generate_patches();
        // Pre-compute the patch index for each pixel.
        let mut patch_indices = vec![0i32; WIDTH * HEIGHT];
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                let x = ((2 * i) as f32 / WIDTH as f32 - 1.0) * std::f32::consts::SQRT_2;
                // Flip the y-axis to match the BSDF coordinate system.
                let y = -((2 * j) as f32 / HEIGHT as f32 - 1.0) * std::f32::consts::SQRT_2;
                let r_disc = (x * x + y * y).sqrt();
                let theta = 2.0 * (r_disc / 2.0).asin();
                let phi = {
                    let phi = (y).atan2(x);
                    if phi < 0.0 {
                        phi + std::f32::consts::TAU
                    } else {
                        phi
                    }
                };
                patch_indices[i + j * WIDTH] =
                    match patches.contains(Sph2::new(rad!(theta), rad!(phi))) {
                        None => -1,
                        Some(idx) => idx as i32,
                    }
            }
        }

        let date_string = chrono::Local::now().format("%Y%m%dT%H%M%S").to_string();
        let mut layer_attrib = LayerAttributes {
            owner: Text::new_or_none("vgonio"),
            capture_date: Text::new_or_none(&date_string),
            software_name: Text::new_or_none("vgonio"),
            other: self.params.to_exr_extra_info(),
            ..LayerAttributes::default()
        };

        // Each snapshot is saved as a separate layer of the image.
        let layers = self
            .snapshots
            .iter()
            .map(|snapshot| {
                layer_attrib.layer_name = Text::new_or_none(format!(
                    "th{:4.2}_ph{:4.2}",
                    snapshot.w_i.theta.in_degrees().as_f32(),
                    snapshot.w_i.phi.in_degrees().as_f32()
                ));
                for i in 0..WIDTH {
                    for j in 0..HEIGHT {
                        let idx = patch_indices[i + j * WIDTH];
                        if idx < 0 {
                            continue;
                        }
                        for (wavelength_idx, bsdf) in
                            bsdf_samples_per_wavelength.iter_mut().enumerate()
                        {
                            bsdf[i + j * WIDTH] = snapshot.samples[idx as usize][wavelength_idx];
                        }
                    }
                }
                let channels = wavelengths
                    .iter()
                    .enumerate()
                    .map(|(i, wavelength)| {
                        let name = Text::new_or_panic(format!("{}", wavelength));
                        AnyChannel::new(
                            name,
                            FlatSamples::F32(bsdf_samples_per_wavelength[i].clone()),
                        )
                    })
                    .collect::<Vec<_>>();
                Layer::new(
                    (WIDTH, HEIGHT),
                    layer_attrib.clone(),
                    Encoding::FAST_LOSSLESS,
                    AnyChannels {
                        list: SmallVec::from(channels),
                    },
                )
            })
            .collect::<Vec<_>>();

        let img_attrib = ImageAttributes::new(IntegerBounds::new((0, 0), (WIDTH, HEIGHT)));
        let image = Image::from_layers(img_attrib, layers);
        let filename = format!("bsdf_{}_{name}.exr", date_string,);
        image.write().to_file(path.join(filename)).unwrap();
    }

    #[cfg(feature = "visu-dbg")]
    /// Returns the trajectories of the rays for each BSDF snapshot.
    pub fn trajectories(&self) -> Vec<Vec<RayTrajectory>> {
        self.snapshots
            .iter()
            .map(|snapshot| snapshot.trajectories.clone())
            .collect()
    }

    #[cfg(feature = "visu-dbg")]
    /// Returns the hit points on the collector for each BSDF snapshot.
    pub fn hit_points(&self) -> Vec<Vec<Vec3>> {
        self.snapshots
            .iter()
            .map(|snapshot| snapshot.hit_points.clone())
            .collect()
    }
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
#[derive(Debug, Default)]
pub struct PerWavelength<T>(Vec<T>);

impl<T> PerWavelength<T> {
    /// Creates a new empty `PerWavelength`.
    pub fn new() -> Self { Self(Vec::new()) }

    /// Creates a new `PerWavelength` with the given value for each wavelength.
    pub fn splat(val: T, len: usize) -> Self
    where
        T: Clone,
    {
        Self(vec![val; len])
    }

    /// Creates a new `PerWavelength` from the given vector.
    pub fn from_vec(vec: Vec<T>) -> Self { Self(vec) }
}

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

impl<T> Index<usize> for PerWavelength<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}

impl<T> IndexMut<usize> for PerWavelength<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
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

/// A snapshot of the BSDF during the measurement.
///  
/// This is the data collected at a single incident direction of the emitter.
///
/// It contains the statistics of the measurement and the data collected
/// for all the patches of the collector at the incident direction.
#[derive(Clone)]
pub struct BsdfSnapshotRaw<D>
where
    D: PerPatchData,
{
    /// Incident direction in the unit spherical coordinates.
    pub w_i: Sph2,
    /// Statistics of the measurement at the point.
    pub stats: BsdfMeasurementStatsPoint,
    /// A list of data collected for each patch of the collector.
    pub records: Vec<PerWavelength<D>>,
    /// Extra ray trajectory data for debugging purposes.
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    pub trajectories: Vec<RayTrajectory>,
    /// Hit points on the collector.
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    pub hit_points: Vec<Vec3>,
}

impl<D: Debug + PerPatchData> Debug for BsdfSnapshotRaw<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BsdfMeasurementPoint")
            .field("stats", &self.stats)
            .field("data", &self.records)
            .finish()
    }
}

impl<D: PerPatchData + PartialEq> PartialEq for BsdfSnapshotRaw<D> {
    fn eq(&self, other: &Self) -> bool {
        self.stats == other.stats && self.records == other.records
    }
}

/// A snapshot of the measured BSDF.
#[derive(Debug, Clone)]
pub struct BsdfSnapshot {
    /// Incident direction in the unit spherical coordinates.
    pub w_i: Sph2,
    /// BSDF values for each patch of the collector.
    pub samples: Vec<PerWavelength<f32>>,
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    /// Extra ray trajectory data for debugging purposes.
    pub trajectories: Vec<RayTrajectory>,
    #[cfg(any(feature = "visu-dbg", debug_assertions))]
    /// Hit points on the collector for debugging purposes.
    pub hit_points: Vec<Vec3>,
}

/// Ray tracing simulation result for a single incident direction of a surface.
pub struct SimulationResultPoint {
    /// Incident direction in the unit spherical coordinates.
    pub w_i: Sph2,
    /// Trajectories of the rays.
    pub trajectories: Vec<RayTrajectory>,
}

/// Ray tracing simulation result for all possible incident directions of a
/// surface.
pub struct SimulationResult {
    /// Surface of the interest.
    pub surface: Handle<MicroSurface>,
    /// Simulation results for each incident direction.
    pub outputs: Vec<SimulationResultPoint>,
}

/// Measures the BSDF of a surface using geometric ray tracing methods.
pub fn measure_bsdf_rt(
    params: BsdfMeasurementParams,
    handles: &[Handle<MicroSurface>],
    sim_kind: SimulationKind,
    cache: &InnerCache,
    single_point: Option<Sph2>,
) -> Vec<MeasurementData> {
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = cache.get_micro_surfaces(handles);
    let emitter = Emitter::new(&params.emitter);
    let detector = Detector::new(&params.detector, &params, cache);

    let mut measurements = Vec::new();
    for (surf, mesh) in surfaces.iter().zip(meshes) {
        if surf.is_none() || mesh.is_none() {
            log::debug!("Skipping surface {:?} and its mesh {:?}", surf, mesh);
            continue;
        }

        let surf = surf.unwrap();
        let mesh = mesh.unwrap();

        log::info!(
            "Measuring surface {}",
            surf.path.as_ref().unwrap().display()
        );

        let sim_result_points = match sim_kind {
            SimulationKind::GeomOptics(method) => {
                println!(
                    "    {BRIGHT_YELLOW}>{RESET} Measuring {} with geometric optics...",
                    params.kind
                );
                match method {
                    #[cfg(feature = "embree")]
                    RtcMethod::Embree => {
                        embr::simulate_bsdf_measurement(&emitter, mesh, single_point)
                    }
                    #[cfg(feature = "optix")]
                    RtcMethod::Optix => rtc_simulation_optix(&params, mesh, &emitter, cache),
                    RtcMethod::Grid => rtc_simulation_grid(&params, surf, mesh, &emitter, cache),
                }
            }
            SimulationKind::WaveOptics => {
                println!(
                    "    {BRIGHT_YELLOW}>{RESET} Measuring {} with wave optics...",
                    params.kind
                );
                todo!("Wave optics simulation is not yet implemented")
            }
        };

        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        log::trace!("Estimated orbit radius: {}", orbit_radius);
        let mut collected = CollectedData::empty(Handle::with_id(surf.uuid), &detector.patches);
        for sim_result_point in sim_result_points {
            log::debug!("Collecting BSDF snapshot at {}", sim_result_point.w_i);

            #[cfg(feature = "bench")]
            let t = std::time::Instant::now();

            detector.collect(&sim_result_point, &mut collected, orbit_radius);

            #[cfg(feature = "bench")]
            {
                let elapsed = t.elapsed();
                log::debug!(
                    "bsdf measurement data collection (one snapshot) took {} secs.",
                    elapsed.as_secs_f64()
                );
            }
        }

        let snapshots = collected.compute_bsdf(&params);
        measurements.push(MeasurementData {
            name: surf.file_stem().unwrap().to_owned(),
            source: MeasurementDataSource::Measured(collected.surface),
            measured: MeasuredData::Bsdf(MeasuredBsdfData { params, snapshots }),
        })
    }

    measurements
}

/// Brdf measurement of a microfacet surface using the grid ray tracing.
fn rtc_simulation_grid(
    _params: &BsdfMeasurementParams,
    _surf: &MicroSurface,
    _mesh: &MicroSurfaceMesh,
    _emitter: &Emitter,
    _cache: &InnerCache,
) -> Box<dyn Iterator<Item = SimulationResultPoint>> {
    // for (surf, mesh) in surfaces.iter().zip(meshes.iter()) {
    //     if surf.is_none() || mesh.is_none() {
    //         log::debug!("Skipping surface {:?} and its mesh {:?}", surf,
    // mesh);         continue;
    //     }
    //     let surf = surf.unwrap();
    //     let _mesh = mesh.unwrap();
    //     println!(
    //         "      {BRIGHT_YELLOW}>{RESET} Measure surface {}",
    //         surf.path.as_ref().unwrap().display()
    //     );
    //     // let t = std::time::Instant::now();
    //     // crate::measure::bsdf::rtc::grid::measure_bsdf(
    //     //     &params, surf, mesh, &emitter, cache,
    //     // );
    //     // println!(
    //     //     "        {BRIGHT_CYAN}✓{RESET} Done in {:?} s",
    //     //     t.elapsed().as_secs_f32()
    //     // );
    // }
    todo!("Grid ray tracing is not yet implemented");
}

/// Brdf measurement of a microfacet surface using the OptiX ray tracing.
#[cfg(feature = "optix")]
fn rtc_simulation_optix(
    _params: &BsdfMeasurementParams,
    _surf: &MicroSurfaceMesh,
    _emitter: &Emitter,
    _cache: &InnerCache,
) -> Box<dyn Iterator<Item = SimulationResultPoint>> {
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

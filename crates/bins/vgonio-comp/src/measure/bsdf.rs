//! Measurement of the BSDF (Bidirectional Scattering Distribution Function) of
//! micro-surfaces.

#[cfg(feature = "embree")]
use crate::measure::bsdf::rtc::embr;
#[cfg(feature = "vdbg")]
use crate::measure::bsdf::rtc::RayTrajectory;
use crate::{
    app::{cache::RawCache, cli::ansi},
    measure::{
        bsdf::{
            receiver::{BounceAndEnergy, Receiver},
            rtc::RtcMethod,
        },
        AnyMeasured,
    },
};
use chrono::{DateTime, Local};
use jabr::array::DyArr;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    path::Path,
};
use surf::{MicroSurface, MicroSurfaceMesh};
use vgonio_bxdf::brdf::measured::{
    ClausenBrdf, ClausenBrdfParametrisation, VgonioBrdf, VgonioBrdfParameterisation,
};
use vgonio_core::{
    bxdf::{MeasuredBrdfKind, Origin},
    error::VgonioError,
    math::{rcp_f64, Sph2, Vec3},
    res::{Handle, RawDataStore},
    units::{Degs, Nanometres, Radians, Rads},
    utils::{medium::Medium, partition::SphericalPartition},
    AnyMeasuredBrdf, BrdfLevel, MeasurementKind,
};
use vgonio_meas::{
    bsdf::{
        emitter::Emitter,
        params::{BsdfMeasurementParams, SimulationKind},
        BsdfMeasurement, RawBsdfMeasurement, SingleBsdfMeasurementStats,
    },
    Measurement, MeasurementSource,
};

pub mod receiver;
pub mod rtc;

// TODO: data retrieval and processing
/// Ray tracing simulation result for a single incident direction of a surface.
pub struct SingleSimResult {
    /// Incident direction in the unit spherical coordinates.
    pub wi: Sph2,
    /// Trajectories of the rays.
    #[cfg(feature = "vdbg")]
    pub trajectories: Box<[RayTrajectory]>,
    /// Number of bounces of the rays.
    #[cfg(not(feature = "vdbg"))]
    pub bounces: Box<[u32]>,
    /// Final directions of the rays.
    #[cfg(not(feature = "vdbg"))]
    pub dirs: Box<[Vec3]>,
    /// Energy of the rays per wavelength.
    #[cfg(not(feature = "vdbg"))]
    pub energy: DyArr<f32, 2>,
}

/// Iterator over the rays in the simulation result.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRays<'a> {
    idx: usize,
    result: &'a SingleSimResult,
}

/// Single ray information in the simulation result.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRay<'a> {
    /// Bounce of the ray.
    pub bounce: &'a u32,
    /// Direction of the ray.
    pub dir: &'a Vec3,
    /// Energy of the ray per wavelength.
    pub energy: &'a [f32],
}

#[cfg(not(feature = "vdbg"))]
impl SingleSimResult {
    /// Returns an iterator over the rays in the simulation result.
    pub fn iter_rays(&self) -> SingleSimResultRays {
        debug_assert_eq!(self.bounces.len(), self.dirs.len(), "Length mismatch");
        debug_assert_eq!(
            self.bounces.len() * self.energy.shape()[1],
            self.energy.len(),
            "Length mismatch"
        );

        SingleSimResultRays {
            idx: 0,
            result: self,
        }
    }

    /// Returns an iterator over the rays in the simulation result in chunks.
    pub fn iter_ray_chunks(&self, chunk_size: usize) -> SingleSimResultRayChunks {
        debug_assert_eq!(self.bounces.len(), self.dirs.len(), "Length mismatch");
        debug_assert_eq!(
            self.bounces.len() * self.energy.shape()[1],
            self.energy.len(),
            "Length mismatch"
        );

        SingleSimResultRayChunks {
            chunk_idx: 0,
            chunk_size,
            chunk_count: (self.bounces.len() + chunk_size - 1) / chunk_size,
            result: self,
        }
    }
}

#[cfg(not(feature = "vdbg"))]
impl<'a> Iterator for SingleSimResultRays<'a> {
    type Item = SingleSimResultRay<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.result.bounces.len() {
            let bounce = &self.result.bounces[self.idx];
            let dir = &self.result.dirs[self.idx];
            let n_spectrum = self.result.energy.shape()[1];
            let energy =
                &self.result.energy.as_slice()[self.idx * n_spectrum..(self.idx + 1) * n_spectrum];
            self.idx += 1;
            Some(SingleSimResultRay {
                bounce,
                dir,
                energy,
            })
        } else {
            None
        }
    }
}

#[cfg(not(feature = "vdbg"))]
impl<'a> ExactSizeIterator for SingleSimResultRays<'a> {
    fn len(&self) -> usize { self.result.bounces.len() }
}

/// Chunks of rays in the simulation result.
///
/// This is useful for processing the rays in parallel.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRayChunks<'a> {
    chunk_idx: usize,
    chunk_size: usize,
    chunk_count: usize,
    result: &'a SingleSimResult,
}

#[cfg(not(feature = "vdbg"))]
impl<'a> Iterator for SingleSimResultRayChunks<'a> {
    type Item = SingleSimResultRayChunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let n_rays = self.result.bounces.len();
        if self.chunk_idx < self.chunk_count {
            let start = self.chunk_idx * self.chunk_size;
            let end = usize::min((self.chunk_idx + 1) * self.chunk_size, n_rays);
            let size = self.result.bounces.len().min(end) - start;
            let bounces = &self.result.bounces[start..start + size];
            let dirs = &self.result.dirs[start..start + size];
            let n_spectrum = self.result.energy.shape()[1];
            let energy = &self.result.energy.as_slice()[start * n_spectrum..end * n_spectrum];
            self.chunk_idx += 1;
            Some(SingleSimResultRayChunk {
                size,
                n_spectrum,
                bounces,
                dirs,
                energy,
                curr: 0,
            })
        } else {
            None
        }
    }
}

/// A chunk of rays in the simulation result.
#[cfg(not(feature = "vdbg"))]
pub struct SingleSimResultRayChunk<'a> {
    /// Number of rays in the chunk.
    pub size: usize,
    /// Number of wavelengths.
    pub n_spectrum: usize,
    /// Bounces of the rays.
    pub bounces: &'a [u32],
    /// Directions of the rays.
    pub dirs: &'a [Vec3],
    /// Energy of the rays per wavelength.
    pub energy: &'a [f32],
    /// Current index of the iterator.
    pub curr: usize,
}

#[cfg(not(feature = "vdbg"))]
impl<'a> Iterator for SingleSimResultRayChunk<'a> {
    type Item = SingleSimResultRay<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.size {
            let idx = self.curr;
            let bounce = &self.bounces[idx];
            let dir = &self.dirs[idx];
            let energy = &self.energy[idx * self.n_spectrum..(idx + 1) * self.n_spectrum];
            self.curr += 1;
            Some(SingleSimResultRay {
                bounce,
                dir,
                energy,
            })
        } else {
            None
        }
    }
}

#[cfg(not(feature = "vdbg"))]
impl<'a> ExactSizeIterator for SingleSimResultRayChunk<'a> {
    fn len(&self) -> usize { self.size }
}

// /// Measures the BSDF of a surface using geometric ray tracing methods.
// pub fn measure_bsdf_rt(
//     params: BsdfMeasurementParams,
//     handles: &[Handle<MicroSurface>],
//     cache: &RawCache,
// ) -> Box<[Measurement]> {
//     let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
//     let surfaces = cache.get_micro_surfaces(handles);
//     let emitter = Emitter::new(&params.emitter);
//     let receiver = Receiver::new(&params.receivers, &params, cache);
//     let n_wi = emitter.measpts.len();
//     let n_wo = receiver.patches.n_patches();
//     let n_spectrum = params.emitter.spectrum.step_count();
//     let spectrum = DyArr::from_iterator([-1],
// params.emitter.spectrum.values());     #[cfg(not(feature = "visu-dbg"))]
//     let iors_i = cache
//         .iors
//         .ior_of_spectrum(params.incident_medium, spectrum.as_ref())
//         .unwrap();
//     #[cfg(not(feature = "visu-dbg"))]
//     let iors_t = cache
//         .iors
//         .ior_of_spectrum(params.transmitted_medium, spectrum.as_ref())
//         .unwrap();
//
//     log::debug!(
//         "Measuring BSDF of {} surfaces from {} measurement points with {}
// rays",         surfaces.len(),
//         emitter.measpts.len(),
//         emitter.params.num_rays,
//     );
//
//     let mut measurements = Vec::with_capacity(surfaces.len());
//     for (surf, mesh) in surfaces.iter().zip(meshes) {
//         if surf.is_none() || mesh.is_none() {
//             log::debug!("Skipping surface {:?} and its mesh {:?}", surf,
// mesh);             continue;
//         }
//
//         let surf = surf.unwrap();
//         let mesh = mesh.unwrap();
//
//         log::info!(
//             "Measuring surface {}",
//             surf.path.as_ref().unwrap().display()
//         );
//
//         let sim_results = match &params.sim_kind {
//             SimulationKind::GeomOptics(method) => {
//                 println!(
//                     "    {} Measuring {} with geometric optics...",
//                     ansi::YELLOW_GT,
//                     params.kind
//                 );
//                 match method {
//                     #[cfg(feature = "embree")]
//                     RtcMethod::Embree => embr::simulate_bsdf_measurement(
//                         #[cfg(not(feature = "visu-dbg"))]
//                         &params,
//                         &emitter,
//                         mesh,
//                         #[cfg(not(feature = "visu-dbg"))]
//                         &iors_i,
//                         #[cfg(not(feature = "visu-dbg"))]
//                         &iors_t,
//                     ),
//                     #[cfg(feature = "optix")]
//                     RtcMethod::Optix => rtc_simulation_optix(&params, mesh,
// &emitter, cache),                     RtcMethod::Grid =>
// rtc_simulation_grid(&params, surf, mesh, &emitter, cache),                 }
//             }
//             SimulationKind::WaveOptics => {
//                 println!(
//                     "    {} Measuring {} with wave optics...",
//                     ansi::YELLOW_GT,
//                     params.kind
//                 );
//                 todo!("Wave optics simulation is not yet implemented")
//             }
//         };
//
//         let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
//         log::trace!("Estimated orbit radius: {}", orbit_radius);
//
//         let incoming = DyArr::<Sph2>::from_slice([n_wi], &emitter.measpts);
//
//         #[cfg(feature = "visu-dbg")]
//         let mut trajectories: Box<[MaybeUninit<Box<[RayTrajectory]>>]> =
//             Box::new_uninit_slice(n_wi);
//         #[cfg(feature = "visu-dbg")]
//         let mut hit_points: Box<[MaybeUninit<Vec<Vec3>>]> =
// Box::new_uninit_slice(n_wi);
//
//         let mut records = DyArr::splat(Option::<BounceAndEnergy>::None,
// [n_wi, n_wo, n_spectrum]);         let mut stats:
// Box<[MaybeUninit<SingleBsdfMeasurementStats>]> = Box::new_uninit_slice(n_wi);
//
//         for (i, sim) in sim_results.into_iter().enumerate() {
//             #[cfg(feature = "visu-dbg")]
//             let trjs = trajectories[i].as_mut_ptr();
//             #[cfg(feature = "visu-dbg")]
//             let hpts = hit_points[i].as_mut_ptr();
//             let recs =
//                 &mut records.as_mut_slice()[i * n_wo * n_spectrum..(i + 1) *
// n_wo * n_spectrum];
//
//             #[cfg(feature = "bench")]
//             let t = std::time::Instant::now();
//
//             println!(
//                 "        {} Collecting BSDF snapshot {}{}/{}{}...",
//                 ansi::YELLOW_GT,
//                 ansi::BRIGHT_CYAN,
//                 i + 1,
//                 n_wi,
//                 ansi::RESET
//             );
//
//             // Collect the tracing data into raw bsdf snapshots.
//             receiver.collect(
//                 sim,
//                 stats[i].as_mut_ptr(),
//                 recs,
//                 #[cfg(feature = "visu-dbg")]
//                 orbit_radius,
//                 #[cfg(feature = "visu-dbg")]
//                 params.fresnel,
//                 #[cfg(feature = "visu-dbg")]
//                 trjs,
//                 #[cfg(feature = "visu-dbg")]
//                 hpts,
//             );
//
//             #[cfg(feature = "bench")]
//             {
//                 let elapsed = t.elapsed();
//                 log::debug!(
//                     "bsdf measurement data collection (one snapshot) took {}
// secs.",                     elapsed.as_secs_f64()
//                 );
//             }
//         }
//
//         let raw = RawMeasuredBsdfData {
//             n_zenith_in: emitter.params.zenith.step_count_wrapped(),
//             spectrum: spectrum.clone(),
//             incoming,
//             outgoing: receiver.patches.clone(),
//             records,
//             stats: DyArr::from_boxed_slice([n_wi], unsafe {
// stats.assume_init() }),             #[cfg(feature = "visu-dbg")]
//             trajectories: unsafe { trajectories.assume_init() },
//             #[cfg(feature = "visu-dbg")]
//             hit_points: unsafe { hit_points.assume_init() },
//         };
//
//         let bsdfs = raw.compute_bsdfs(params.incident_medium,
// params.transmitted_medium);
//
//         measurements.push(Measurement {
//             name: surf.file_stem().unwrap().to_owned(),
//             source: MeasurementSource::Measured(Handle::with_id(surf.uuid)),
//             timestamp: chrono::Local::now(),
//             measured: Box::new(MeasuredBsdfData { params, raw, bsdfs }),
//         });
//     }
//
//     measurements.into_boxed_slice()
// }

/// Measures the BSDF of a surface using geometric ray tracing methods.
pub fn measure_bsdf_rt(
    params: BsdfMeasurementParams,
    handles: &[Handle],
    cache: &RawCache,
) -> Box<[Measurement]> {
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let surfaces = cache.get_micro_surfaces(handles);
    let emitter = Emitter::new(&params.emitter);
    let n_wi = emitter.measpts.len();
    let n_spectrum = params.emitter.spectrum.step_count();
    let spectrum = DyArr::from_iterator([-1], params.emitter.spectrum.values());
    #[cfg(not(feature = "vdbg"))]
    let iors_i = cache
        .iors
        .ior_of_spectrum(params.incident_medium, spectrum.as_ref())
        .unwrap();
    #[cfg(not(feature = "vdbg"))]
    let iors_t = cache
        .iors
        .ior_of_spectrum(params.transmitted_medium, spectrum.as_ref())
        .unwrap();

    log::debug!(
        "Measuring BSDF of {} surfaces from {} measurement points with {} rays",
        surfaces.len(),
        emitter.measpts.len(),
        emitter.params.num_rays,
    );

    let incoming = DyArr::<Sph2>::from_slice([n_wi], &emitter.measpts);
    let mut measurements = Vec::with_capacity(surfaces.len() * params.receivers.len());

    for (surf, mesh) in surfaces.iter().zip(meshes) {
        #[cfg(feature = "vdbg")]
        let mut trajectories: Vec<Vec<RayTrajectory>> =
            vec![Vec::with_capacity(emitter.params.num_rays as usize); n_wi];
        #[cfg(feature = "vdbg")]
        let mut hit_points: Vec<Vec<Vec3>> = vec![Vec::new(); n_wi];

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

        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        log::trace!("Estimated orbit radius: {}", orbit_radius);

        // Receiver with its records & stats
        let mut receivers = params
            .receivers
            .iter()
            .map(|rparams| {
                let r = Receiver::new(rparams, &params, cache);
                let records = DyArr::splat(
                    Option::<BounceAndEnergy>::None,
                    [n_wi, r.n_wo(), n_spectrum],
                );
                let stats: Box<[Option<SingleBsdfMeasurementStats>]> =
                    vec![None; n_wi].into_boxed_slice();
                (r, records, stats, *rparams)
            })
            .collect::<Box<_>>();

        match &params.sim_kind {
            SimulationKind::GeomOptics(method) => {
                #[cfg(feature = "embree")]
                let (_, scene, geometry) = embr::create_resources(mesh);
                for sector in emitter.circular_sectors() {
                    for (i, wi) in sector.measpts.iter().enumerate() {
                        #[cfg(feature = "bench")]
                        let t = std::time::Instant::now();
                        let single_result = match method {
                            #[cfg(feature = "embree")]
                            RtcMethod::Embree => embr::simulate_bsdf_measurement_single_point(
                                *wi,
                                &sector,
                                mesh,
                                geometry.clone(),
                                &scene,
                                #[cfg(not(feature = "vdbg"))]
                                params.fresnel,
                                #[cfg(not(feature = "vdbg"))]
                                &iors_i,
                                #[cfg(not(feature = "vdbg"))]
                                &iors_t,
                            ),
                            _ => unimplemented!("Temporarily deactivated"),
                        };
                        #[cfg(feature = "bench")]
                        {
                            let elapsed = t.elapsed();
                            log::debug!(
                                "bsdf measurement simulation (one snapshot) took {} secs.",
                                elapsed.as_secs_f64()
                            );
                        }

                        for (j, (receiver, records, stats, _)) in receivers.iter_mut().enumerate() {
                            let n_wo = receiver.n_wo();
                            let recs = &mut records.as_mut_slice()
                                [i * n_wo * n_spectrum..(i + 1) * n_wo * n_spectrum];

                            #[cfg(feature = "bench")]
                            let t = std::time::Instant::now();

                            println!(
                                "        {} Collecting BSDF snapshot {}{}/{}{} to receiver #{}...",
                                ansi::YELLOW_GT,
                                ansi::BRIGHT_CYAN,
                                i + 1,
                                n_wi,
                                ansi::RESET,
                                j
                            );

                            // Print receiver number of patches
                            println!(
                                "Receiver number of patches: {}",
                                receiver.patches.n_patches()
                            );

                            // Collect the tracing data into raw bsdf snapshots.
                            receiver.collect(
                                &single_result,
                                &mut stats[i],
                                recs,
                                #[cfg(feature = "vdbg")]
                                orbit_radius,
                                #[cfg(feature = "vdbg")]
                                params.fresnel,
                                #[cfg(feature = "vdbg")]
                                &mut trajectories[i],
                                #[cfg(feature = "vdbg")]
                                &mut hit_points[i],
                            );

                            #[cfg(feature = "bench")]
                            {
                                let elapsed = t.elapsed();
                                log::debug!(
                                    "bsdf measurement data collection (one snapshot) took {} secs.",
                                    elapsed.as_secs_f64()
                                );
                            }
                        }
                    }
                }

                #[cfg(feature = "vdbg")]
                let trajectories = trajectories
                    .into_iter()
                    .map(|t| t.into_boxed_slice())
                    .collect::<Box<_>>();
                #[cfg(feature = "vdbg")]
                let hit_points = hit_points.into_boxed_slice();

                for (receiver, records, stats, rparams) in receivers {
                    let stats = unsafe {
                        std::mem::transmute::<
                            Box<[Option<SingleBsdfMeasurementStats>]>,
                            Box<[SingleBsdfMeasurementStats]>,
                        >(stats)
                    };
                    let raw = RawBsdfMeasurement {
                        n_zenith_in: emitter.params.zenith.step_count_wrapped(),
                        spectrum: spectrum.clone(),
                        incoming: incoming.clone(),
                        outgoing: receiver.patches,
                        records,
                        stats: DyArr::from_slice([n_wi], &stats),
                        #[cfg(feature = "vdbg")]
                        trajectories: trajectories.clone(),
                        #[cfg(feature = "vdbg")]
                        hit_points: hit_points.clone(),
                    };
                    let bsdfs =
                        raw.compute_bsdfs(params.incident_medium, params.transmitted_medium);
                    let params = BsdfMeasurementParams {
                        kind: params.kind,
                        sim_kind: params.sim_kind,
                        incident_medium: params.incident_medium,
                        transmitted_medium: params.transmitted_medium,
                        emitter: params.emitter,
                        receivers: vec![rparams],
                        fresnel: params.fresnel,
                    };
                    measurements.push(Measurement {
                        name: surf.file_stem().unwrap().to_owned(),
                        source: MeasurementSource::Measured(Handle::with_id::<MicroSurface>(
                            surf.uuid,
                        )),
                        timestamp: Local::now(),
                        measured: Box::new(BsdfMeasurement { params, raw, bsdfs }),
                    });
                }
            },
            SimulationKind::WaveOptics => {
                println!(
                    "    {} Measuring {} with wave optics...",
                    ansi::YELLOW_GT,
                    params.kind
                );
                todo!("Wave optics simulation is not yet implemented")
            },
        }
    }

    measurements.into_boxed_slice()
}

/// Brdf measurement of a microfacet surface using the grid ray tracing.
fn rtc_simulation_grid<'a>(
    _params: &'a BsdfMeasurementParams,
    _surf: &'a MicroSurface,
    _mesh: &'a MicroSurfaceMesh,
    _emitter: &'a Emitter,
    _cache: &'a RawDataStore,
) -> Box<dyn Iterator<Item = SingleSimResult>> {
    // Temporary deactivated
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
    todo!()
}

/// Brdf measurement of a microfacet surface using the OptiX ray tracing.
#[cfg(feature = "optix")]
fn rtc_simulation_optix<'a>(
    _params: &'a BsdfMeasurementParams,
    _surf: &'a MicroSurfaceMesh,
    _emitter: &'a Emitter,
    _cache: &'a RawDataStore,
) -> Box<dyn Iterator<Item = SingleSimResult>> {
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

use crate::{
    app::cache::{Handle, InnerCache},
    measure::{
        bsdf::{
            BsdfMeasurementStatsPoint, BsdfSnapshot, BsdfSnapshotRaw, PerWavelength,
            SimulationResult,
        },
        params::BsdfMeasurementParams,
    },
    optics::fresnel,
    SphericalDomain,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vgcore::{
    math,
    math::{rcp_f32, Sph2, Vec2, Vec3, Vec3A},
    units::{rad, Radians, SolidAngle},
};
use vgsurf::MicroSurface;

/// Description of a detector collecting the data.
///
/// The virtual goniophotometer's sensors are represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
///
/// A detector is defined by its domain, the precision of the
/// measurements and the partitioning scheme.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectorParams {
    /// Domain of the collector.
    pub domain: SphericalDomain,
    /// Zenith angle step size (in radians).
    pub precision: Radians,
    /// Partitioning scheme of the collector.
    pub scheme: DetectorScheme,
}

/// Scheme of the partitioning of the detector.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetectorScheme {
    /// Partition scheme based on "A general rule for disk and hemisphere
    /// partition into equal-area cells" by Benoit Beckers et Pierre Beckers.
    Beckers = 0x00,
    /// Partition scheme based on "Subdivision of the sky hemisphere for
    /// luminance measurements" by P.R. Tregenza.
    Tregenza = 0x01,
}

/// Partitioned patches of the collector.
#[derive(Debug, Clone)]
pub enum DetectorPatches {
    /// Beckers partitioning scheme.
    Beckers {
        /// The annuli of the collector.
        rings: Vec<Ring>,
        /// The patches of the collector.
        patches: Vec<Patch>,
    },
    // TODO: implement Tregenza
    /// Tregenza partitioning scheme.
    Tregenza,
}

impl DetectorPatches {
    /// Returns the number of patches.
    pub fn len(&self) -> usize {
        match self {
            Self::Beckers { patches, .. } => patches.len(),
            Self::Tregenza => todo!("Tregenza partitioning scheme is not implemented yet"),
        }
    }

    /// Returns the iterator over the patches.
    pub fn patches_iter(&self) -> impl Iterator<Item = &Patch> {
        match self {
            Self::Beckers { patches, .. } => patches.iter(),
            Self::Tregenza => todo!("Tregenza partitioning scheme is not implemented yet"),
        }
    }

    /// Returns the iterator over the rings.
    pub fn rings_iter(&self) -> impl Iterator<Item = &Ring> {
        match self {
            Self::Beckers { rings, .. } => rings.iter(),
            Self::Tregenza => todo!("Tregenza partitioning scheme is not implemented yet"),
        }
    }

    /// Returns the index of the patch containing the direction.
    pub fn contains(&self, sph: Sph2) -> Option<usize> {
        match self {
            Self::Beckers { rings, patches } => {
                for ring in rings.iter() {
                    if ring.theta_inner <= sph.theta.as_f32()
                        && sph.theta.as_f32() <= ring.theta_outer
                    {
                        for (i, patch) in patches.iter().skip(ring.base_index).enumerate() {
                            if patch.min.phi <= sph.phi && sph.phi <= patch.max.phi {
                                return Some(ring.base_index + i);
                            }
                        }
                    }
                }
                None
            }
            Self::Tregenza => todo!("Tregenza partitioning scheme is not implemented yet"),
        }
    }

    pub fn write_to_image(&self) {
        const WIDTH: usize = 1024;
        const HEIGHT: usize = 1024;
        let mut pixels = vec![0; WIDTH * HEIGHT];
        let mut vertices = Vec::new();
        match self {
            DetectorPatches::Beckers { rings, .. } => {
                // Generate from rings.
                let (disc_xs, disc_ys): (Vec<_>, Vec<_>) = (0..360)
                    .map(|i| {
                        let a = i as f32 * std::f32::consts::PI / 180.0;
                        (a.cos(), a.sin())
                    })
                    .unzip();
                let mut max_x = f32::MIN;
                for ring in rings.iter() {
                    let inner_radius = (ring.theta_inner * 0.5).sin();
                    let outer_radius = (ring.theta_outer * 0.5).sin();
                    // Generate the outer border of the ring
                    for (j, (x, y)) in disc_xs.iter().zip(disc_ys.iter()).enumerate() {
                        vertices.push(Vec2::new(*x * outer_radius, *y * outer_radius));
                        vertices.push(Vec2::new(
                            disc_xs[(j + 1) % 360] * outer_radius,
                            disc_ys[(j + 1) % 360] * outer_radius,
                        ));
                    }
                    let step = std::f32::consts::TAU / ring.patch_count as f32;
                    // Generate the cells
                    if ring.patch_count == 1 {
                        continue;
                    } else {
                        for k in 0..ring.patch_count {
                            let x = (step * k as f32).cos();
                            let y = (step * k as f32).sin();
                            let inner_pnt = Vec2::new(x * inner_radius, y * inner_radius);
                            let outer_pnt = Vec2::new(x * outer_radius, y * outer_radius);
                            let dx = outer_pnt.x - inner_pnt.x;
                            let dy = outer_pnt.y - inner_pnt.y;
                            for l in 0..10 {
                                let pnt = Vec2::new(
                                    inner_pnt.x + dx * l as f32 / 10.0,
                                    inner_pnt.y + dy * l as f32 / 10.0,
                                );
                                vertices.push(pnt);
                                vertices.push(Vec2::new(pnt.x + dx / 10.0, pnt.y + dy / 10.0));
                            }
                        }
                    }
                }
                let half_w = WIDTH as f32 / 2.0;
                let half_h = HEIGHT as f32 / 2.0;
                for vtx in vertices {
                    let x = (vtx.x * f32::sqrt(2.0) * half_w + half_w).min(WIDTH as f32 - 1.0);
                    let y = (vtx.y * f32::sqrt(2.0) * half_h + half_h).min(HEIGHT as f32 - 1.0);
                    println!("x: {}, y: {}", x, y);
                    pixels[y as usize * WIDTH + x as usize] = 255;
                }
                println!("max_x: {}", max_x);
                let image =
                    image::GrayImage::from_raw(WIDTH as u32, HEIGHT as u32, pixels).unwrap();
                image.save("test.png").unwrap();
            }
            DetectorPatches::Tregenza => {}
        }
    }
}

#[test]
fn write_patches_to_image() {
    let params = DetectorParams {
        domain: SphericalDomain::Upper,
        precision: deg!(5.0).to_radians(),
        scheme: DetectorScheme::Beckers,
    };
    let patches = params.generate_patches();
    patches.write_to_image();
}

/// A patch of the detector.
#[derive(Debug, Copy, Clone)]
pub struct Patch {
    /// Minimum zenith (theta) and azimuth (phi) angles of the patch.
    pub min: Sph2,
    /// Maximum zenith (theta) and azimuth (phi) angles of the patch.
    pub max: Sph2,
}

impl Patch {
    /// Creates a new patch.
    pub fn new(theta_min: Radians, theta_max: Radians, phi_min: Radians, phi_max: Radians) -> Self {
        Self {
            min: Sph2::new(theta_min, phi_min),
            max: Sph2::new(theta_max, phi_max),
        }
    }

    /// Returns the center of the patch.
    pub fn center(&self) -> Sph2 {
        Sph2::new(
            (self.min.theta + self.max.theta) / 2.0,
            (self.min.phi + self.max.phi) / 2.0,
        )
    }

    pub fn solid_angle(&self) -> SolidAngle {
        let d_theta = self.max.theta - self.min.theta;
        let d_phi = self.max.phi - self.min.phi;
        let sin_theta = ((self.min.theta + self.max.theta) / 2.0).sin();
        SolidAngle::new(d_theta.as_f32() * d_phi.as_f32() * sin_theta)
    }

    /// Checks if the patch contains the direction.
    pub fn contains(&self, dir: Vec3) -> bool {
        let sph = Sph2::from_cartesian(dir);
        self.min.theta <= sph.theta
            && sph.theta <= self.max.theta
            && self.min.phi <= sph.phi
            && sph.phi <= self.max.phi
    }
}

/// A segment in form of an annulus of the collector.
#[derive(Debug, Copy, Clone)]
pub struct Ring {
    /// Minimum theta angle of the annulus.
    pub theta_inner: f32,
    /// Maximum theta angle of the annulus.
    pub theta_outer: f32,
    /// Step size of the phi angle inside the annulus.
    pub phi_step: f32,
    /// Number of patches in the annulus: 2 * pi / phi_step == patch_count.
    pub patch_count: usize,
    /// Base index of the annulus in the patches buffer.
    pub base_index: usize,
}

mod beckers {
    use vgcore::math::sqr;

    /// Computes the number of cells inside the external circle of the ring.
    pub fn compute_ks(k0: u32, num_rings: u32) -> Vec<u32> {
        let mut ks = vec![0; num_rings as usize];
        ks[0] = k0;
        let sqrt_pi = std::f32::consts::PI.sqrt();
        for i in 1..num_rings as usize {
            ks[i] = sqr(f32::sqrt(ks[i - 1] as f32) + sqrt_pi).round() as u32;
        }
        ks
    }

    /// Computes the radius of the rings.
    pub fn compute_rs(ks: &[u32], num_rings: u32, radius: f32) -> Vec<f32> {
        let mut rs = vec![0.0; num_rings as usize];
        rs[0] = radius * f32::sqrt(ks[0] as f32 / ks[num_rings as usize - 1] as f32);
        for i in 1..num_rings as usize {
            rs[i] = (ks[i] as f32 / ks[i - 1] as f32).sqrt() * rs[i - 1]
        }
        rs
    }

    /// Computes the zenith angle of the rings on the hemisphere.
    pub fn compute_ts(rs: &[f32]) -> Vec<f32> {
        rs.iter().map(|r| 2.0 * (r / 2.0).asin()).collect()
    }
}

impl DetectorParams {
    /// Returns the number of patches of the collector.
    pub fn patches_count(&self) -> usize {
        match self.scheme {
            DetectorScheme::Beckers => {
                let num_rings = (Radians::HALF_PI / self.precision).round() as u32;
                let ks = beckers::compute_ks(1, num_rings);
                ks[num_rings as usize - 1] as usize
            }
            DetectorScheme::Tregenza => 0,
        }
    }

    /// Generates the patches of the collector.
    ///
    /// The patches are generated based on the scheme of the collector. They are
    /// used to collect the data. The patches are generated in the order of
    /// the azimuth angle first, then the zenith angle.
    pub fn generate_patches(&self) -> DetectorPatches {
        match self.scheme {
            DetectorScheme::Beckers => {
                let num_rings = (Radians::HALF_PI / self.precision).round() as u32;
                let ks = beckers::compute_ks(1, num_rings);
                let rs = beckers::compute_rs(&ks, num_rings, f32::sqrt(2.0));
                let ts = beckers::compute_ts(&rs);
                let mut patches = Vec::with_capacity(ks[num_rings as usize - 1] as usize);
                let mut rings = Vec::with_capacity(num_rings as usize);
                // Patches are generated in the order of rings.
                for (i, (t, k)) in ts.iter().zip(ks.iter()).enumerate() {
                    log::trace!("Ring {}: t = {}, k = {}", i, t.to_degrees(), k);
                    let k_prev = if i == 0 { 0 } else { ks[i - 1] };
                    let n = k - k_prev;
                    let t_prev = if i == 0 { 0.0 } else { ts[i - 1] };
                    let phi_step = Radians::TWO_PI / n as f32;
                    rings.push(Ring {
                        theta_inner: t_prev,
                        theta_outer: *t,
                        phi_step: phi_step.as_f32(),
                        patch_count: n as usize,
                        base_index: patches.len(),
                    });
                    for j in 0..n {
                        let phi_min = phi_step * j as f32;
                        let phi_max = phi_step * (j + 1) as f32;
                        patches.push(Patch::new(
                            t_prev.into(),
                            rad!(*t),
                            phi_min.into(),
                            phi_max.into(),
                        ));
                    }
                }
                DetectorPatches::Beckers { rings, patches }
            }
            DetectorScheme::Tregenza => {
                todo!("Tregenza partitioning scheme is not implemented yet")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Detector {
    params: DetectorParams,
    patches: DetectorPatches,
}

/// Outgoing ray of a trajectory [`RayTrajectory`].
///
/// Used during the data collection process.
#[derive(Debug, Copy, Clone)]
struct OutgoingRayDir {
    /// Index of the ray in the trajectory.
    pub idx: usize,
    /// The final direction of the ray.
    pub dir: Vec3A,
    /// The number of bounces of the ray.
    pub bounce: usize,
}

impl Detector {
    /// Creates a new detector.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the detector.
    pub fn new(params: &DetectorParams) -> Self {
        Self {
            params: *params,
            patches: params.generate_patches(),
        }
    }

    /// Collects the ray-tracing data.
    ///
    /// Returns the collected data and the statistics of the BSDF.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the BSDF measurement.
    /// * `sim_res` - The simulation results.
    /// * `cache` - The cache where all the data are stored.
    ///
    /// # Returns
    ///
    /// The collected data for each simulation result which is a vector of
    /// [`BsdfSnapshotRaw`].
    pub fn collect(
        &self,
        params: &BsdfMeasurementParams,
        sim_res: &[SimulationResult],
        cache: &InnerCache,
    ) -> Vec<CollectedData> {
        let spectrum = params.emitter.spectrum.values().collect::<Vec<_>>();
        let spectrum_len = spectrum.len();
        log::debug!("[Collector] spectrum samples: {:?}", spectrum);
        // Retrieve the incident medium's refractive indices for each wavelength.
        let iors_i = cache
            .iors
            .ior_of_spectrum(params.incident_medium, &spectrum)
            .expect("incident medium IOR not found");
        // Retrieve the transmitted medium's refractive indices for each wavelength.
        let iors_t = cache
            .iors
            .ior_of_spectrum(params.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");
        let max_bounces = params.emitter.max_bounces as usize;

        sim_res
            .iter()
            .map(|res| {
                log::info!("Collecting BSDF data of surface {}", res.surface.id());
                let orbit_radius = crate::measure::estimate_orbit_radius(
                    cache
                        .get_micro_surface_mesh_by_surface_id(res.surface)
                        .unwrap(),
                );
                log::trace!("[Collector] Estimated orbit radius: {}", orbit_radius);

                let n_received = std::sync::atomic::AtomicU32::new(0);

                let data = res
                    .outputs
                    .iter()
                    .map(|output| {
                        log::debug!("[Collector] collecting data at {}", output.w_i);

                        #[cfg(feature = "bench")]
                        let start = std::time::Instant::now();

                        let mut stats = BsdfMeasurementStatsPoint::new(spectrum_len, max_bounces);
                        // Convert the last rays of the trajectories into a vector located
                        // at the collector's center and pointing to the intersection point
                        // of the last ray with the collector's surface.
                        // Each element of the vector is a tuple containing the index of the
                        // trajectory, the intersection point and the number of bounces.
                        let dirs = output
                            .trajectories
                            .par_iter()
                            .enumerate()
                            .filter_map(|(i, trajectory)| {
                                match trajectory.last() {
                                    None => None,
                                    Some(last) => {
                                        n_received
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        // 1. Calculate the intersection point of the last ray with
                                        // the collector's surface. Ray-Sphere intersection.
                                        // Collect's center is at (0, 0, 0).
                                        let a = last.dir.dot(last.dir);
                                        let b = 2.0 * last.dir.dot(last.org);
                                        let c = last.org.dot(last.org) - math::sqr(orbit_radius);
                                        let p = match math::solve_quadratic(a, b, c) {
                                            math::QuadraticSolution::None
                                            | math::QuadraticSolution::One(_) => {
                                                unreachable!(
                                                    "Ray starting inside the collector's surface, \
                                                     it should have more than one intersection \
                                                     point."
                                                )
                                            }
                                            math::QuadraticSolution::Two(_, t) => {
                                                last.org + last.dir * t
                                            }
                                        };
                                        // Returns the index of the ray, the unit vector pointing to
                                        // the collector's surface, and the number of bounces.
                                        Some(OutgoingRayDir {
                                            idx: i,
                                            dir: p.normalize(),
                                            bounce: trajectory.len() - 1,
                                        })
                                    }
                                }
                            })
                            .collect::<Vec<_>>();
                        stats.n_received = n_received.load(std::sync::atomic::Ordering::Relaxed);

                        #[cfg(feature = "bench")]
                        let dirs_time = start.elapsed().as_millis();
                        #[cfg(feature = "bench")]
                        log::info!("Collector::collect: dirs: {} ms", dirs_time);

                        // Calculate the energy of each rays per wavelength.
                        let ray_energy_per_wavelength = dirs
                            .iter()
                            .map(|w_o| {
                                let trajectory = &output.trajectories[w_o.idx];
                                let mut energy = vec![Energy::Reflected(1.0); spectrum_len];
                                for node in trajectory.iter().take(trajectory.len() - 1) {
                                    for i in 0..spectrum_len {
                                        match energy[i] {
                                            Energy::Absorbed => continue,
                                            Energy::Reflected(ref mut e) => {
                                                *e *= fresnel::reflectance(
                                                    node.cos.unwrap_or(1.0),
                                                    iors_i[i],
                                                    iors_t[i],
                                                );
                                                if *e <= 0.0 {
                                                    energy[i] = Energy::Absorbed;
                                                }
                                            }
                                        }
                                    }
                                }
                                (w_o.idx, energy)
                            })
                            .collect::<Vec<_>>();
                        #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                        log::debug!("ray_energy_per_wavelength: {:?}", ray_energy_per_wavelength);

                        #[cfg(feature = "bench")]
                        let ray_energy_time = start.elapsed().as_millis();
                        #[cfg(feature = "bench")]
                        log::info!(
                            "Collector::collect: ray_energy_per_wavelength: {} ms",
                            ray_energy_time - dirs_time
                        );

                        // Calculate the number of absorbed and reflected rays per wavelength.
                        for (_, energies) in &ray_energy_per_wavelength {
                            // per ray
                            for (idx_wavelength, energy) in energies.iter().enumerate() {
                                // per wavelength
                                match energy {
                                    Energy::Absorbed => stats.n_absorbed[idx_wavelength] += 1,
                                    Energy::Reflected(_) => stats.n_reflected[idx_wavelength] += 1,
                                }
                            }
                        }
                        log::debug!("stats.n_absorbed: {:?}", stats.n_absorbed);
                        log::debug!("stats.n_reflected: {:?}", stats.n_reflected);
                        log::debug!("stats.n_received: {:?}", stats.n_received);

                        // Collect the outgoing rays per patch.
                        let outgoing_dirs_per_patch = self
                            .patches
                            .patches_iter()
                            .map(|patch| {
                                // Retrieve the ray indices (of trajectories) that intersect the
                                // patch.
                                dirs.iter()
                                    .filter_map(|w_o| {
                                        patch
                                            .contains(w_o.dir.into())
                                            .then_some((w_o.idx, w_o.bounce))
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();
                        #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                        log::debug!("outgoing_dirs_per_patch: {:?}", outgoing_dirs_per_patch);

                        #[cfg(feature = "bench")]
                        let outgoing_dirs_time = start.elapsed().as_millis();
                        #[cfg(feature = "bench")]
                        log::info!(
                            "Collector::collect: outgoing_dirs_per_patch: {} ms",
                            outgoing_dirs_time - ray_energy_time
                        );

                        let data = outgoing_dirs_per_patch
                            .iter()
                            .map(|dirs| {
                                let mut data_per_patch = PerWavelength(vec![
                                    BounceAndEnergy::empty(
                                        params.emitter.max_bounces as usize
                                    );
                                    spectrum_len
                                ]);
                                for (i, bounce) in dirs.iter() {
                                    let ray_energy = &ray_energy_per_wavelength
                                        .iter()
                                        .find(|(j, _)| *j == *i)
                                        .unwrap()
                                        .1;
                                    for (lambda_idx, energy) in ray_energy.iter().enumerate() {
                                        match energy {
                                            Energy::Absorbed => continue,
                                            Energy::Reflected(e) => {
                                                stats.n_captured[lambda_idx] += 1;
                                                data_per_patch[lambda_idx].total_energy += e;
                                                data_per_patch[lambda_idx].total_rays += 1;
                                                data_per_patch[lambda_idx].energy_per_bounce
                                                    [*bounce - 1] += e;
                                                data_per_patch[lambda_idx].num_rays_per_bounce
                                                    [*bounce - 1] += 1;
                                                stats.num_rays_per_bounce[lambda_idx]
                                                    [*bounce - 1] += 1;
                                                stats.energy_per_bounce[lambda_idx][*bounce - 1] +=
                                                    e;
                                            }
                                        }
                                        stats.e_captured[lambda_idx] += energy.energy();
                                    }
                                }
                                data_per_patch
                            })
                            .collect::<Vec<_>>();
                        #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                        log::debug!("data per patch: {:?}", data);
                        #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                        log::debug!("stats: {:?}", stats);

                        #[cfg(feature = "bench")]
                        let data_time = start.elapsed().as_millis();
                        #[cfg(feature = "bench")]
                        log::info!(
                            "Collector::collect: data: {} ms",
                            data_time - outgoing_dirs_time
                        );

                        // Compute the vertex positions of the outgoing rays.
                        #[cfg(any(feature = "visu-dbg", debug_assertions))]
                        let outgoing_vertex_positions: Vec<Vec3> = dirs
                            .iter()
                            .map(|w_o| (w_o.dir * orbit_radius).into())
                            .collect::<Vec<_>>();

                        BsdfSnapshotRaw {
                            w_i: output.w_i,
                            records: data,
                            stats,
                            #[cfg(any(feature = "visu-dbg", debug_assertions))]
                            trajectories: output.trajectories.to_vec(),
                            #[cfg(any(feature = "visu-dbg", debug_assertions))]
                            hit_points: outgoing_vertex_positions,
                        }
                    })
                    .collect();
                CollectedData {
                    surface: res.surface,
                    patches: self.patches.clone(),
                    snapshots: data,
                }
            })
            .collect()
    }
}

/// Collected data by the detector for a specific micro-surface.
///
/// The data are collected for patches of the detector and for each
/// wavelength of the incident spectrum.
#[derive(Debug, Clone)]
pub struct CollectedData {
    /// The micro-surface where the data were collected.
    pub surface: Handle<MicroSurface>,
    /// The partitioned patches of the detector.
    pub patches: DetectorPatches,
    /// The collected data.
    pub snapshots: Vec<BsdfSnapshotRaw<BounceAndEnergy>>,
}

impl CollectedData {
    /// Computes the BSDF according to the collected data.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters of the BSDF measurement.
    ///
    /// # Returns
    ///
    /// The BSDF computed from the collected data for each wavelength of the
    /// incident spectrum. The output vector is of size equal to the number of
    /// measurement points of the emitter times the number of patches of the
    /// detector.
    pub fn compute_bsdf(&self, params: &BsdfMeasurementParams) -> Vec<BsdfSnapshot> {
        // For each snapshot (w_i), compute the BSDF.
        log::info!(
            "Computing BSDF... with {} patches",
            params.detector.patches_count()
        );
        self.snapshots
            .iter()
            .map(|snapshot| {
                let mut samples =
                    vec![
                        PerWavelength::splat(0.0, params.emitter.spectrum.values().len());
                        snapshot.records.len()
                    ];
                let cos_i = snapshot.w_i.theta.cos();
                let l_i = snapshot.stats.n_received as f32 * cos_i;
                for (i, patch_data) in snapshot.records.iter().enumerate() {
                    // Per wavelength
                    for (j, stats) in patch_data.iter().enumerate() {
                        let patch = self.patches.patches_iter().nth(i).unwrap();
                        let cos_o = patch.center().theta.cos();
                        if cos_o == 0.0 {
                            samples[i][j] = 0.0;
                        } else {
                            let d_omega = patch.solid_angle();
                            let l_o = stats.total_energy * rcp_f32(cos_o * d_omega.as_f32());
                            samples[i][j] = l_o * rcp_f32(l_i);
                            #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                            log::debug!(
                                "energy of patch {i}: {}, λ[{j}] --  L_i: {}, L_o[{i}]: {} -- \
                                 brdf: {}",
                                stats.total_energy,
                                l_i,
                                l_o,
                                samples[i][j],
                            );
                        }
                    }
                }
                #[cfg(all(debug_assertions, feature = "verbose-dbg"))]
                log::debug!("snapshot.samples, w_i = {:?}: {:?}", snapshot.w_i, samples);
                BsdfSnapshot {
                    w_i: snapshot.w_i,
                    samples,
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    trajectories: snapshot.trajectories.clone(),
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    hit_points: snapshot.hit_points.clone(),
                }
            })
            .collect::<Vec<_>>()
    }
}

/// Energy after a ray is reflected by the micro-surface.
///
/// Used during the data collection process.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
enum Energy {
    /// The ray of certain wavelength is absorbed by the micro-surface.
    Absorbed,
    /// The ray of a specific wavelength is reflected by the micro-surface.
    Reflected(f32),
}

impl Energy {
    /// Returns the energy of the ray.
    fn energy(&self) -> f32 {
        match self {
            Self::Absorbed => 0.0,
            Self::Reflected(energy) => *energy,
        }
    }
}

/// Represents the data that a patch can carry.
pub trait PerPatchData: Sized + Clone + Send + Sync + 'static {}

/// Bounce and energy of a patch.
///
/// Length of `num_rays_per_bounce` and `energy_per_bounce` is equal to the
/// maximum number of bounces.
#[derive(Debug, Clone)]
pub struct BounceAndEnergy {
    /// Total number of rays that hit the patch.
    pub total_rays: u32,
    /// Total energy of rays that hit the patch.
    pub total_energy: f32,
    /// Number of rays hitting the patch at the given bounce.
    pub num_rays_per_bounce: Vec<u32>,
    /// Total energy of rays hitting the patch at the given bounce.
    pub energy_per_bounce: Vec<f32>,
}

impl BounceAndEnergy {
    pub fn empty(bounces: usize) -> Self {
        Self {
            num_rays_per_bounce: vec![0; bounces],
            energy_per_bounce: vec![0.0; bounces],
            total_energy: 0.0,
            total_rays: 0,
        }
    }

    pub const fn calc_size_in_bytes(bounces: usize) -> usize { 8 * bounces + 8 }
}

impl PartialEq for BounceAndEnergy {
    fn eq(&self, other: &Self) -> bool {
        self.total_rays == other.total_rays
            && self.total_energy == other.total_energy
            && self.num_rays_per_bounce == other.num_rays_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}

impl PerPatchData for BounceAndEnergy {}

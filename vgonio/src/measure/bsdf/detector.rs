use crate::{
    app::{
        cache,
        cache::{Handle, InnerCache},
    },
    measure::{
        bsdf::{
            rtc::RayTrajectory, BsdfMeasurementDataPoint, BsdfMeasurementStatsPoint,
            MeasuredBsdfData, PerWavelength, SimulationResult, SimulationResultPoint,
        },
        params::BsdfMeasurementParams,
    },
    optics::fresnel,
    SphericalDomain,
};
use serde::{Deserialize, Serialize};
use vgcore::{
    math,
    math::{rcp_f32, Sph2, UVec2, Vec2, Vec3, Vec3A},
    units::{deg, rad, Radians, SolidAngle},
};
use vgsurf::{MicroSurface, MicroSurfaceMesh};

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

impl DetectorScheme {
    pub fn patches_count(&self) -> usize {
        match self {
            Self::Beckers => {
                let num_rings = (Radians::HALF_PI / rad!(0.1)).round() as u32;
                let ks = beckers::compute_ks(1, num_rings);
                ks[num_rings as usize - 1] as usize
            }
            Self::Tregenza => 0,
        }
    }
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
                log::trace!("ks: {:?}", ks);
                log::trace!("rs: {:?}", rs);
                log::trace!("ts: {:?}", ts);
                let mut patches = Vec::with_capacity(ks[num_rings as usize - 1] as usize);
                let mut rings = Vec::with_capacity(num_rings as usize);
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
    /// [`BsdfMeasurementDataPoint`].
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
        log::debug!("[Collector] incident medium IORs: {:?}", iors_i);
        // Retrieve the transmitted medium's refractive indices for each wavelength.
        let iors_t = cache
            .iors
            .ior_of_spectrum(params.transmitted_medium, &spectrum)
            .expect("transmitted medium IOR not found");
        log::debug!("[Collector] transmitted medium IORs: {:?}", iors_t);
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
                log::trace!("[Collector] Estimated orbit radius: {}", orbit_radius,);

                let data = res
                    .outputs
                    .iter()
                    .map(|output| {
                        log::debug!("[Collector] collecting data at {}", output.w_i);
                        let mut stats = BsdfMeasurementStatsPoint::new(spectrum_len, max_bounces);
                        // Convert the last rays of the trajectories into a vector located
                        // at the collector's center and pointing to the intersection point
                        // of the last ray with the collector's surface.
                        // Each element of the vector is a tuple containing the index of the
                        // trajectory, the intersection point and the number of bounces.
                        let dirs = output
                            .trajectories
                            .iter()
                            .enumerate()
                            .filter_map(|(i, trajectory)| {
                                match trajectory.last() {
                                    None => None,
                                    Some(last) => {
                                        stats.n_received += 1;
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
                                        // the collector's
                                        // surface, and the number of bounces.
                                        Some(OutgoingRayDir {
                                            idx: i,
                                            dir: p.normalize(),
                                            bounce: trajectory.len() - 1,
                                        })
                                    }
                                }
                            })
                            .collect::<Vec<_>>();

                        // Calculate the energy of the rays (with wavelength) that intersected the
                        // collector's surface.
                        // Outer index: ray index, inner index: wavelength index.
                        let ray_energy_per_wavelength = dirs
                            .iter()
                            .map(|w_o| {
                                let trajectory = &output.trajectories[w_o.idx];
                                let mut energy = vec![Energy::Reflected(1.0); spectrum_len];
                                for node in trajectory.iter().take(trajectory.len() - 1) {
                                    for i in 0..spectrum.len() {
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

                        // For each patch, collect the rays that intersect it using the
                        // outgoing_dirs vector.
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

                        let data = outgoing_dirs_per_patch
                            .iter()
                            .map(|dirs| {
                                let mut data_per_patch = PerWavelength(vec![
                                    BounceAndEnergy::empty(
                                        params.emitter.max_bounces as usize
                                    );
                                    spectrum.len()
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

                        // Compute the vertex positions of the outgoing rays.
                        #[cfg(debug_assertions)]
                        let outgoing_vertex_positions: Vec<Vec3> = dirs
                            .iter()
                            .map(|w_o| (w_o.dir * orbit_radius).into())
                            .collect::<Vec<_>>();

                        BsdfMeasurementDataPoint {
                            w_i: output.w_i,
                            per_patch_data: data,
                            stats,
                            #[cfg(debug_assertions)]
                            trajectories: output.trajectories.to_vec(),
                            #[cfg(debug_assertions)]
                            hit_points: outgoing_vertex_positions,
                        }
                    })
                    .collect();
                CollectedData {
                    surface: res.surface,
                    patches: self.patches.clone(),
                    data,
                }
            })
            .collect()
    }
}

/// Collected data by the detector for one micro-surface.
#[derive(Debug, Clone)]
pub struct CollectedData {
    /// The micro-surface where the data were collected.
    pub surface: Handle<MicroSurface>,
    /// The partitioned patches of the detector.
    pub patches: DetectorPatches,
    /// The collected data.
    pub data: Vec<BsdfMeasurementDataPoint<BounceAndEnergy>>,
}

impl CollectedData {
    pub fn compute_bsdf(&self, params: &BsdfMeasurementParams) -> Vec<Vec<PerWavelength<f32>>> {
        self.data
            .iter()
            .map(|data_point| {
                let mut bsdf = vec![
                    PerWavelength(vec![0.0; params.emitter.spectrum.values().len()]);
                    data_point.per_patch_data.len()
                ];
                let cos_i = data_point.w_i.theta.cos();
                let incident_irradiance = data_point.stats.n_received as f32 * cos_i;
                for (i, patch_data) in data_point.per_patch_data.iter().enumerate() {
                    for (j, stats) in patch_data.0.iter().enumerate() {
                        let patch = self.patches.patches_iter().skip(i).next().unwrap();
                        let cos_o = patch.center().theta.cos();
                        if cos_o == 0.0 {
                            bsdf[i][j] = 0.0;
                        } else {
                            let d_omega = patch.solid_angle();
                            let outgoing_radiance =
                                stats.total_energy * rcp_f32(cos_o * d_omega.as_f32());
                            bsdf[i][j] = outgoing_radiance * rcp_f32(incident_irradiance);
                        }
                    }
                }
                bsdf
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

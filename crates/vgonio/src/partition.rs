use crate::{RangeByStepSizeInclusive, SphericalDomain};
use base::{
    math::{Sph2, Vec3},
    units::{rad, Radians, SolidAngle},
};
use serde::{Deserialize, Serialize};

/// Scheme of the spherical partition.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PartitionScheme {
    /// Partition scheme based on "A general rule for disk and hemisphere
    /// partition into equal-area cells" by Benoit Beckers et Pierre Beckers.
    Beckers = 0x00,
    /// Simple partition scheme which divides uniformly the hemisphere into
    /// equal angle patches.
    EqualAngle = 0x01,
}

impl PartitionScheme {
    /// Returns the number of patches.
    pub fn num_patches(&self, precision: Sph2) -> usize {
        match self {
            PartitionScheme::Beckers => {
                let num_rings = (Radians::HALF_PI / precision.theta).round() as u32;
                let ks = beckers::compute_ks(1, num_rings);
                ks[num_rings as usize - 1] as usize
            }
            PartitionScheme::EqualAngle => {
                debug_assert!(
                    precision.theta > Radians::ZERO && precision.phi > Radians::ZERO,
                    "theta/phi precision must be greater than zero"
                );
                let azimuth =
                    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TWO_PI, precision.phi);
                let zenith =
                    RangeByStepSizeInclusive::new(Radians::ZERO, Radians::HALF_PI, precision.theta);
                azimuth.step_count_wrapped() * zenith.step_count_wrapped()
            }
        }
    }
}

/// Partitioned patches of the collector.
#[derive(Debug, Clone)]
pub struct SphericalPartition {
    /// Precision of the partitioning scheme.
    pub precision: Sph2,
    /// The partitioning scheme of the collector.
    pub scheme: PartitionScheme,
    /// The domain of the receiver.
    pub domain: SphericalDomain,
    /// The annuli of the collector.
    pub rings: Box<[Ring]>,
    /// The patches of the collector.
    pub patches: Box<[Patch]>,
}

impl SphericalPartition {
    /// Creates a new spherical partition.
    pub fn new(scheme: PartitionScheme, domain: SphericalDomain, precision: Sph2) -> Self {
        match scheme {
            PartitionScheme::Beckers => Self::new_beckers(domain, precision),
            PartitionScheme::EqualAngle => Self::new_equal_angle(domain, precision),
        }
    }

    /// Creates a new partition based on the Beckers partitioning scheme.
    pub fn new_beckers(domain: SphericalDomain, precision: Sph2) -> Self {
        debug_assert!(
            precision.theta > Radians::ZERO,
            "theta precision must be greater than zero"
        );
        let num_rings = (Radians::HALF_PI / precision.theta).round() as u32;
        let ks = beckers::compute_ks(1, num_rings);
        let rs = beckers::compute_rs(&ks, num_rings, std::f32::consts::SQRT_2);
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
                patches.push(Patch::new(t_prev.into(), rad!(*t), phi_min, phi_max));
            }
        }

        // Mirror the patches of the upper hemisphere to the lower hemisphere.
        if domain.is_lower_hemisphere() {
            Self::mirror_partition(&mut patches, &mut rings);
        }

        // Append the patches of the lower hemisphere to the upper hemisphere.
        if domain.is_full_sphere() {
            Self::mirror_then_extend_partition(&mut patches, &mut rings);
        }

        Self {
            precision,
            scheme: PartitionScheme::Beckers,
            domain,
            rings: rings.into_boxed_slice(),
            patches: patches.into_boxed_slice(),
        }
    }

    /// Creates a new partition based on the EqualAngle partitioning scheme.
    pub fn new_equal_angle(domain: SphericalDomain, precision: Sph2) -> Self {
        debug_assert!(
            precision.theta > Radians::ZERO && precision.phi > Radians::ZERO,
            "theta/phi precision must be greater than zero"
        );
        // Put the measurement point at the center of the angle intervals.
        let num_rings = (Radians::HALF_PI / precision.theta).round() as usize + 1;
        let num_patches_per_ring =
            RangeByStepSizeInclusive::new(Radians::ZERO, Radians::TWO_PI, precision.phi)
                .step_count_wrapped();
        let mut rings = Vec::with_capacity(num_rings);
        let mut patches = Vec::with_capacity(num_rings * num_patches_per_ring);
        let half_theta = precision.theta * 0.5;
        for i in 0..num_rings {
            let theta_min = (i as f32 * precision.theta - half_theta).max(Radians::ZERO);
            let theta_max = (i as f32 * precision.theta + half_theta).min(Radians::HALF_PI);
            for j in 0..num_patches_per_ring {
                let phi_min = j as f32 * precision.phi;
                let phi_max = phi_min + precision.phi;
                patches.push(Patch::new(theta_min, theta_max, phi_min, phi_max));
            }
            rings.push(Ring {
                theta_inner: theta_min.as_f32(),
                theta_outer: theta_max.as_f32(),
                phi_step: precision.phi.as_f32(),
                patch_count: num_patches_per_ring,
                base_index: i * num_patches_per_ring,
            });
        }

        // Mirror the patches of the upper hemisphere to the lower hemisphere.
        if domain.is_lower_hemisphere() {
            Self::mirror_partition(&mut patches, &mut rings);
        }

        // Append the patches of the lower hemisphere to the upper hemisphere.
        if domain.is_full_sphere() {
            Self::mirror_then_extend_partition(&mut patches, &mut rings);
        }

        Self {
            precision,
            scheme: PartitionScheme::EqualAngle,
            domain,
            rings: rings.into_boxed_slice(),
            patches: patches.into_boxed_slice(),
        }
    }

    /// Returns the number of patches.
    pub fn num_patches(&self) -> usize { self.patches.len() }

    /// Returns the number of rings.
    pub fn num_rings(&self) -> usize { self.rings.len() }

    /// Returns the index of the patch containing the direction.
    pub fn contains(&self, sph: Sph2) -> Option<usize> {
        for ring in self.rings.iter() {
            if ring.theta_inner <= sph.theta.as_f32() && sph.theta.as_f32() <= ring.theta_outer {
                for (i, patch) in self.patches.iter().skip(ring.base_index).enumerate() {
                    // The patch starts at the minus phi.
                    if patch.min.phi > patch.max.phi {
                        if sph.phi >= Radians::ZERO
                            && sph.phi <= patch.min.phi
                            && sph.phi >= patch.max.phi
                            && sph.phi <= Radians::TAU
                        {
                            return Some(ring.base_index + i);
                        }
                    } else if patch.min.phi <= sph.phi && sph.phi <= patch.max.phi {
                        return Some(ring.base_index + i);
                    }
                }
            }
        }
        None
    }

    /// Mirror the patches and rings of the upper hemisphere to the lower
    /// hemisphere.
    fn mirror_partition(patches: &mut Vec<Patch>, rings: &mut Vec<Ring>) {
        for ring in rings.iter_mut() {
            ring.theta_inner = std::f32::consts::PI - ring.theta_inner;
            ring.theta_outer = std::f32::consts::PI - ring.theta_outer;
        }
        for patch in patches.iter_mut() {
            patch.min.theta = Radians::PI - patch.min.theta;
            patch.max.theta = Radians::PI - patch.max.theta;
        }
    }

    /// Mirror the patches and rings of the upper hemisphere to the lower
    /// hemisphere and append them to the upper hemisphere.
    fn mirror_then_extend_partition(patches: &mut Vec<Patch>, rings: &mut Vec<Ring>) {
        let mut patches_lower = patches.clone();
        let mut rings_lower = rings.clone();
        Self::mirror_partition(&mut patches_lower, &mut rings_lower);
        patches.extend(patches_lower);
        rings.extend(rings_lower);
    }
}

/// A patch of the receiver.
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
            (self.min.theta + self.max.theta) * 0.5,
            (self.min.phi + self.max.phi) * 0.5,
        )
    }

    /// Returns the solid angle of the patch.
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

/// Beckers partitioning scheme helper functions.
pub mod beckers {
    use base::math::sqr;

    /// Computes the number of cells inside the external circle of the ring.
    pub fn compute_ks(k0: u32, num_rings: u32) -> Box<[u32]> {
        let mut ks = vec![0; num_rings as usize];
        ks[0] = k0;
        let sqrt_pi = std::f32::consts::PI.sqrt();
        for i in 1..num_rings as usize {
            ks[i] = sqr(f32::sqrt(ks[i - 1] as f32) + sqrt_pi).round() as u32;
        }
        ks.into_boxed_slice()
    }

    /// Computes the radius of the rings.
    pub fn compute_rs(ks: &[u32], num_rings: u32, radius: f32) -> Box<[f32]> {
        let mut rs = vec![0.0; num_rings as usize];
        rs[0] = radius * f32::sqrt(ks[0] as f32 / ks[num_rings as usize - 1] as f32);
        for i in 1..num_rings as usize {
            rs[i] = (ks[i] as f32 / ks[i - 1] as f32).sqrt() * rs[i - 1]
        }
        rs.into_boxed_slice()
    }

    /// Computes the zenith angle of the rings on the hemisphere.
    pub fn compute_ts(rs: &[f32]) -> Box<[f32]> {
        let mut ts = vec![0.0; rs.len()];
        for (i, r) in rs.iter().enumerate() {
            ts[i] = 2.0 * (r / 2.0).asin();
        }
        ts.into_boxed_slice()
    }
}

// Methods related to the data collection.
impl SphericalPartition {
    /// Helper function calculating the index of the patch containing the
    /// pixel.
    ///
    /// # Arguments
    /// * `w` - The width of the image.
    /// * `h` - The height of the image.
    /// * `indices` - The buffer to store the patch indices. -1 means no patch.
    pub fn compute_pixel_patch_indices(&self, w: u32, h: u32, indices: &mut [i32]) {
        debug_assert_eq!(
            indices.len(),
            (w * h) as usize,
            "The buffer must match the size of the image."
        );
        match self.scheme {
            PartitionScheme::Beckers => {
                for i in 0..w {
                    // x, width, column
                    for j in 0..h {
                        // y, height, row
                        let x = ((2 * i) as f32 / w as f32 - 1.0) * std::f32::consts::SQRT_2;
                        // Flip the y-axis to match the BSDF coordinate system.
                        let y = -((2 * j) as f32 / h as f32 - 1.0) * std::f32::consts::SQRT_2;
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
                        indices[(i + j * w) as usize] =
                            match self.contains(Sph2::new(rad!(theta), rad!(phi))) {
                                None => -1,
                                Some(idx) => idx as i32,
                            }
                    }
                }
            }
            PartitionScheme::EqualAngle => {
                // Calculate the patch index for each pixel.
                let half_phi_bin_width = self.precision.phi * 0.5;
                for i in 0..w {
                    // x, width, column
                    for j in 0..h {
                        // y, height, row
                        let x = ((2 * i) as f32 / w as f32 - 1.0) * std::f32::consts::SQRT_2;
                        // Flip the y-axis to match the measurement coordinate system.
                        let y = -((2 * j) as f32 / h as f32 - 1.0) * std::f32::consts::SQRT_2;
                        let r_disc = (x * x + y * y).sqrt();
                        let theta = 2.0 * (r_disc / 2.0).asin();
                        let phi = (rad!((y).atan2(x)) + half_phi_bin_width).wrap_to_tau();
                        indices[(i + j * w) as usize] =
                            match self.contains(Sph2::new(rad!(theta), phi)) {
                                None => -1,
                                Some(idx) => idx as i32,
                            }
                    }
                }
            }
        }
    }
}

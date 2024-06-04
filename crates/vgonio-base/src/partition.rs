//! Spherical partitioning.
use crate::{
    math::{Sph2, Vec3},
    range::RangeByStepSizeInclusive,
    units::{rad, Radians, Rads, SolidAngle},
    utils,
};
use num_traits::{Euclid, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, fmt::Display};

use crate::error::VgonioError;
use exr::prelude::Text;
#[cfg(feature = "io")]
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// The domain of the spherical coordinate.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SphericalDomain {
    /// Simulation happens only on the upper part of the sphere.
    #[default]
    #[serde(rename = "upper_hemisphere")]
    Upper = 0x01,

    /// Simulation happens only on the lower part of the sphere.
    #[serde(rename = "lower_hemisphere")]
    Lower = 0x02,

    /// Simulation happens on the whole sphere.
    #[serde(rename = "whole_sphere")]
    Whole = 0x00,
}

impl Display for SphericalDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upper => write!(f, "upper hemisphere"),
            Self::Lower => write!(f, "lower hemisphere"),
            Self::Whole => write!(f, "whole sphere"),
        }
    }
}

impl SphericalDomain {
    /// Range of the zenith angle in radians of the upper hemisphere.
    pub const ZENITH_RANGE_UPPER_DOMAIN: (Radians, Radians) = (Radians::ZERO, Radians::HALF_PI);
    /// Range of the zenith angle in radians of the lower hemisphere.
    pub const ZENITH_RANGE_LOWER_DOMAIN: (Radians, Radians) = (Radians::HALF_PI, Radians::PI);
    /// Range of the zenith angle in radians of the whole sphere.
    pub const ZENITH_RANGE_WHOLE_DOMAIN: (Radians, Radians) = (Radians::ZERO, Radians::TWO_PI);

    /// Returns the zenith range of the domain.
    pub const fn zenith_range(&self) -> (Radians, Radians) {
        match self {
            Self::Upper => Self::ZENITH_RANGE_UPPER_DOMAIN,
            Self::Lower => Self::ZENITH_RANGE_LOWER_DOMAIN,
            Self::Whole => Self::ZENITH_RANGE_WHOLE_DOMAIN,
        }
    }

    /// Returns the azimuth range of the domain.
    pub const fn azimuth_range(&self) -> (Radians, Radians) { (Radians::ZERO, Radians::TWO_PI) }

    /// Returns the zenith angle difference between the maximum and minimum
    /// zenith angle.
    pub const fn zenith_angle_diff(&self) -> Radians {
        match self {
            SphericalDomain::Upper | SphericalDomain::Lower => Radians::HALF_PI,
            SphericalDomain::Whole => Radians::PI,
        }
    }

    /// Clamp the given azimuthal and zenith angle to shape's boundaries.
    ///
    /// # Arguments
    ///
    /// `zenith` - zenith angle in radians.
    /// `azimuth` - azimuthal angle in radians.
    ///
    /// # Returns
    ///
    /// `(zenith, azimuth)` - clamped zenith and azimuth angles in radians.
    #[inline]
    pub fn clamp(&self, zenith: Radians, azimuth: Radians) -> (Radians, Radians) {
        (self.clamp_zenith(zenith), self.clamp_azimuth(azimuth))
    }

    /// Clamps the given zenith angle to shape's boundaries.
    #[inline]
    pub fn clamp_zenith(&self, zenith: Radians) -> Radians {
        let (zenith_min, zenith_max) = self.zenith_range();
        zenith.clamp(zenith_min, zenith_max)
    }

    /// Clamps the given azimuthal angle to shape's boundaries.
    #[inline]
    pub fn clamp_azimuth(&self, azimuth: Radians) -> Radians {
        azimuth.clamp(Radians::ZERO, Radians::TWO_PI)
    }

    /// Returns true if the domain is the upper hemisphere.
    pub fn is_upper_hemisphere(&self) -> bool { matches!(self, SphericalDomain::Upper) }

    /// Returns true if the domain is the lower hemisphere.
    pub fn is_lower_hemisphere(&self) -> bool { matches!(self, SphericalDomain::Lower) }

    /// Returns true if the domain is the full sphere.
    pub fn is_full_sphere(&self) -> bool { matches!(self, SphericalDomain::Whole) }
}

/// Scheme of the spherical partition.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PartitionScheme {
    /// Partition scheme based on "A general rule for disk and hemisphere
    /// partition into equal-area cells" by Benoit Beckers et Pierre Beckers.
    Beckers = 0x00,
    /// Simple partition scheme which uniformly divides the hemisphere into
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Create a new spherical partition.
    ///
    /// The partition is based on the given partitioning scheme, domain, and
    /// precision. In the case of the Beckers partitioning scheme, only the
    /// theta precision is used.
    pub fn new(scheme: PartitionScheme, domain: SphericalDomain, precision: Sph2) -> Self {
        match scheme {
            PartitionScheme::Beckers => Self::new_beckers(domain, precision.theta),
            PartitionScheme::EqualAngle => Self::new_equal_angle(domain, precision),
        }
    }

    /// Creates a new partition based on the Beckers partitioning scheme.
    pub fn new_beckers(domain: SphericalDomain, theta_precision: Radians) -> Self {
        debug_assert!(
            theta_precision > Radians::ZERO,
            "theta precision must be greater than zero"
        );
        let num_rings = (Radians::HALF_PI / theta_precision).round() as u32;
        let ks = beckers::compute_ks(1, num_rings);
        let rs = beckers::compute_rs(&ks, num_rings, std::f32::consts::SQRT_2);
        let ts = beckers::compute_ts(&rs);
        let mut patches = Vec::with_capacity(ks[num_rings as usize - 1] as usize);
        let mut rings = Vec::with_capacity(num_rings as usize);
        // Patches are generated in the order of rings.
        for (i, (t, k)) in ts.iter().zip(ks.iter()).enumerate() {
            // log::trace!("Ring {}: t = {}, k = {}", i, t.to_degrees(), k);
            let k_prev = if i == 0 { 0 } else { ks[i - 1] };
            let n = k - k_prev;
            let t_prev = if i == 0 { 0.0 } else { ts[i - 1] };
            let phi_step = Radians::TWO_PI / n as f32;
            rings.push(Ring {
                theta_min: t_prev,
                theta_max: *t,
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
            precision: Sph2::new(theta_precision, Radians::ZERO),
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
                theta_min: theta_min.as_f32(),
                theta_max: theta_max.as_f32(),
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
    pub fn n_patches(&self) -> usize { self.patches.len() }

    /// Returns the number of rings.
    pub fn n_rings(&self) -> usize { self.rings.len() }

    /// Returns the index of the patch containing the direction.
    pub fn contains(&self, sph: Sph2) -> Option<usize> {
        for ring in self.rings.iter() {
            if ring.theta_min <= sph.theta.as_f32() && sph.theta.as_f32() <= ring.theta_max {
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

    /// Find the index of the ring containing the direction and the index of the
    /// nearest ring to the direction, which could be the next or the previous.
    ///
    /// # Arguments
    ///
    /// * `sph` - The direction in spherical coordinates.
    ///
    /// # Returns
    ///
    /// A tuple of the upper bound ring index and the lower bound ring index.
    pub fn find_rings(&self, sph: Sph2) -> (usize, usize) {
        let mut ring_idx = 0;
        let mut min_dist = Rads::TAU;
        for (i, ring) in self.rings.iter().enumerate() {
            let dist = (ring.zenith_center() - sph.theta).abs();
            if dist < min_dist {
                min_dist = dist;
                ring_idx = i;
            }
        }
        let dist = (sph.theta.as_f32() - self.rings[ring_idx].theta_min)
            / (self.rings[ring_idx].theta_max - self.rings[ring_idx].theta_min);

        if dist < 0.5 {
            if ring_idx == 0 {
                (ring_idx, ring_idx)
            } else {
                (ring_idx - 1, ring_idx)
            }
        } else {
            if ring_idx == self.rings.len() - 1 {
                (ring_idx, ring_idx)
            } else {
                (ring_idx, ring_idx + 1)
            }
        }
    }

    /// Mirror the patches and rings of the upper hemisphere to the lower
    /// hemisphere.
    fn mirror_partition(patches: &mut Vec<Patch>, rings: &mut Vec<Ring>) {
        for ring in rings.iter_mut() {
            ring.theta_min = std::f32::consts::PI - ring.theta_min;
            ring.theta_max = std::f32::consts::PI - ring.theta_max;
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

#[cfg(feature = "io")]
impl SphericalPartition {
    /// The size of the buffer required to read or write the parameters.
    pub const fn total_required_size(&self) -> usize { 24 + self.rings.len() * Ring::REQUIRED_SIZE }

    /// Serialize the partition to a buffer for writing to a vgmo file.
    ///
    /// # Arguments
    ///
    /// * `buf` - The buffer to write the partition to.
    /// * `write_domain_scheme` - If true, the domain and scheme are written to
    ///   the buffer.
    #[track_caller]
    pub fn write_to_buf(&self, buf: &mut [u8]) {
        let size_required = self.total_required_size();
        assert_eq!(buf.len(), size_required, "Buffer size mismatch.");
        buf[0..4].copy_from_slice(&(self.domain as u32).to_le_bytes());
        buf[4..8].copy_from_slice(&(self.scheme as u32).to_le_bytes());
        buf[8..12].copy_from_slice(&self.precision.theta.value().to_le_bytes());
        buf[12..16].copy_from_slice(&self.precision.phi.value().to_le_bytes());
        buf[16..20].copy_from_slice(&(self.n_rings() as u32).to_le_bytes());
        buf[20..24].copy_from_slice(&(self.n_patches() as u32).to_le_bytes());
        let mut offset = 24;
        self.rings.iter().for_each(|ring| {
            ring.write_to_buf(&mut buf[offset..offset + Ring::REQUIRED_SIZE]);
            offset += Ring::REQUIRED_SIZE;
        });
    }

    /// Deserialize the partition from a buffer read from a vgmo file without
    /// reading the ring information.
    fn read_from_buf_without_rings(
        buf: &[u8],
    ) -> (SphericalDomain, PartitionScheme, Sph2, usize, usize) {
        assert!(buf.len() >= 24, "Buffer size mismatch.");
        let domain = match u32::from_le_bytes(buf[0..4].try_into().unwrap()) {
            0 => SphericalDomain::Whole,
            1 => SphericalDomain::Upper,
            2 => SphericalDomain::Lower,
            _ => panic!("Invalid domain kind"),
        };
        let scheme = match u32::from_le_bytes(buf[4..8].try_into().unwrap()) {
            0 => PartitionScheme::Beckers,
            1 => PartitionScheme::EqualAngle,
            _ => panic!("Invalid scheme kind"),
        };
        let precision_theta = rad!(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
        let precision_phi = rad!(f32::from_le_bytes(buf[12..16].try_into().unwrap()));
        let n_rings = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let n_patches = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        (
            domain,
            scheme,
            Sph2::new(precision_theta, precision_phi),
            n_rings as usize,
            n_patches as usize,
        )
    }

    /// As the partition can be reconstructed from the parameters, ring and
    /// patch information is skipped during deserialization.
    pub fn read_skipping_rings<R: Read + Seek>(
        reader: &mut BufReader<R>,
    ) -> (SphericalDomain, PartitionScheme, Sph2, usize, usize) {
        let mut buf = [0u8; 24];
        reader.read_exact(&mut buf).unwrap();
        let (domain, scheme, precision, n_rings, n_patches) =
            Self::read_from_buf_without_rings(&buf);
        // Skip the ring information.
        reader
            .seek_relative((n_rings * Ring::REQUIRED_SIZE) as i64)
            .unwrap();
        (domain, scheme, precision, n_rings, n_patches)
    }
}

/// A patch of the receiver.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Patch {
    /// Minimum zenith (theta) and azimuth (phi) angles of the patch.
    pub min: Sph2,
    /// Maximum zenith (theta) and azimuth (phi) angles of the patch.
    pub max: Sph2,
}

impl Patch {
    /// Create a new patch.
    pub fn new(theta_min: Radians, theta_max: Radians, phi_min: Radians, phi_max: Radians) -> Self {
        Self {
            min: Sph2::new(theta_min, phi_min),
            max: Sph2::new(theta_max, phi_max),
        }
    }

    /// Return the center of the patch.
    pub fn center(&self) -> Sph2 {
        // For the patch at the top of the hemisphere.
        if self.min.phi.as_f32() <= 1e-6 && (self.max.phi - Rads::TAU).as_f32().abs() <= 1e-6 {
            // If the patch covers the entire hemisphere, then it must be the top patch.
            Sph2::new(Rads::ZERO, Rads::ZERO)
        } else {
            Sph2::new(
                (self.min.theta + self.max.theta) * 0.5,
                (self.min.phi + self.max.phi) * 0.5,
            )
        }
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

/// A segment in the form of an annulus on the collector hemisphere.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Ring {
    /// Minimum theta angle of the annulus.
    pub theta_min: f32,
    /// Maximum theta angle of the annulus.
    pub theta_max: f32,
    /// Step size of the phi angle inside the annulus.
    pub phi_step: f32,
    /// Number of patches in the annulus: 2 * pi / phi_step.
    pub patch_count: usize,
    /// Base index of the annulus in the patches buffer.
    pub base_index: usize,
}

impl Ring {
    /// Given a phi angle, returns the indices of the patches where the phi
    /// angle falls in between (the patch where the phi resides and the closest
    /// adjacent patch) in the ring.
    pub fn find_patch_indices(&self, phi: Rads) -> (usize, usize) {
        let phi = phi.wrap_to_tau();
        let (q, r) = phi.as_f32().div_rem_euclid(&self.phi_step);
        let a = q as isize;
        if r >= 0.5 * self.phi_step {
            (a as usize, (a + 1) as usize % self.patch_count)
        } else {
            let prev = a - 1;
            if prev < 0 {
                (self.patch_count - 1, 0)
            } else {
                (prev as usize, a as usize)
            }
        }
    }

    /// Return the center of the ring in terms of zenith angle.
    pub fn zenith_center(&self) -> Radians {
        if self.theta_min == 0.0 {
            // The ring at the top of the hemisphere.
            Radians::ZERO
        } else {
            Radians::from(self.theta_min + self.theta_max) * 0.5
        }
    }
}

/// Beckers partitioning scheme helper functions.
pub mod beckers {
    use crate::math::sqr;

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
                            let phi = y.atan2(x);
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
                        let phi = (rad!(y.atan2(x)) + half_phi_bin_width).wrap_to_tau();
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

#[cfg(feature = "io")]
impl Ring {
    /// The size of the buffer required to read or write the parameters.
    pub const REQUIRED_SIZE: usize = 20;

    /// Writes the ring to the given buffer.
    pub fn write_to_buf(&self, buf: &mut [u8]) {
        debug_assert!(
            buf.len() >= Self::REQUIRED_SIZE,
            "Ring needs at least 20 bytes of space"
        );
        buf[0..4].copy_from_slice(&self.theta_min.to_le_bytes());
        buf[4..8].copy_from_slice(&self.theta_max.to_le_bytes());
        buf[8..12].copy_from_slice(&self.phi_step.to_le_bytes());
        buf[12..16].copy_from_slice(&(self.patch_count as u32).to_le_bytes());
        buf[16..20].copy_from_slice(&(self.base_index as u32).to_le_bytes());
    }

    /// Reads the ring from the given buffer.
    pub fn read_from_buf(buf: &[u8]) -> Self {
        debug_assert!(
            buf.len() >= Self::REQUIRED_SIZE,
            "Ring needs at least 20 bytes of space"
        );
        let theta_inner = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        let theta_outer = f32::from_le_bytes(buf[4..8].try_into().unwrap());
        let phi_step = f32::from_le_bytes(buf[8..12].try_into().unwrap());
        let patch_count = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let base_index = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
        Self {
            theta_min: theta_inner,
            theta_max: theta_outer,
            phi_step,
            patch_count,
            base_index,
        }
    }
}

#[cfg(feature = "io")]
/// A writer for the data carried on the hemisphere.
pub struct DataCarriedOnHemisphereImageWriter<'a> {
    /// The partition of the hemisphere.
    partition: &'a SphericalPartition,
    /// Resolution of the output image.
    resolution: u32,
    /// Patch indices for each pixel.
    indices: Box<[i32]>,
}

#[cfg(feature = "io")]
impl<'a> DataCarriedOnHemisphereImageWriter<'a> {
    /// Creates a new writer for the data carried on the hemisphere.
    pub fn new(partition: &'a SphericalPartition, resolution: u32) -> Self {
        // Calculate the patch indices for each pixel.
        let mut indices = vec![0i32; (resolution * resolution) as usize].into_boxed_slice();
        partition.compute_pixel_patch_indices(resolution, resolution, &mut indices);
        Self {
            partition,
            resolution,
            indices,
        }
    }

    /// Writes the data as an EXR file.
    pub fn write_as_exr<L, C>(
        &self,
        data: &'a [f32],
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        layer_name: L,
        channel_name: C,
    ) -> Result<(), VgonioError>
    where
        L: FnOnce(usize) -> Option<Text>,
        C: FnOnce(usize) -> Text,
    {
        if data.len() != self.partition.n_patches() {
            return Err(VgonioError::new(
                "Data carried on hemisphere length mismatches the partition",
                None,
            ));
        }

        use exr::prelude::*;
        let (w, h) = (self.resolution, self.resolution);
        let mut samples = vec![0.0f32; w as usize * h as usize];
        for i in 0..w {
            for j in 0..h {
                let idx = self.indices[(i + j * w) as usize];
                if idx >= 0 {
                    samples[(i + j * w) as usize] = data[idx as usize];
                } else {
                    samples[(i + j * w) as usize] = 0.0;
                }
            }
        }

        let layer = Layer::new(
            (w as usize, h as usize),
            LayerAttributes {
                layer_name: layer_name(0),
                capture_date: Text::new_or_none(utils::iso_timestamp_from_datetime(timestamp)),
                ..LayerAttributes::default()
            },
            Encoding::FAST_LOSSLESS,
            AnyChannels {
                list: SmallVec::from_vec(vec![AnyChannel::new(
                    channel_name(0),
                    FlatSamples::F32(Cow::Borrowed(&samples)),
                )]),
            },
        );
        let image = Image::from_layer(layer);
        image
            .write()
            .to_file(filepath)
            .map_err(|err| VgonioError::new("Failed to write EXR file.", Some(Box::new(err))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::deg;

    #[test]
    fn test_spherical_domain_clamp() {
        let domain = SphericalDomain::Upper;
        let angle = deg!(91.0);
        let clamped = domain.clamp_zenith(angle.into());
        assert_eq!(clamped, deg!(90.0));

        let domain = SphericalDomain::Lower;
        let angle = deg!(191.0);
        let clamped = domain.clamp_zenith(angle.into());
        assert_eq!(clamped, deg!(180.0));
    }

    #[test]
    fn test_ring_find_patches_single_patch() {
        let ring = Ring {
            theta_min: 0.0,
            theta_max: 3.0f32.to_radians(),
            phi_step: std::f32::consts::TAU,
            patch_count: 1,
            base_index: 0,
        };

        {
            let (a, b) = ring.find_patch_indices(Rads::ZERO);
            assert_eq!(a, b);
            assert_eq!(a, 0);
        }
        {
            let (a, b) = ring.find_patch_indices(Rads::PI);
            assert_eq!(a, b);
            assert_eq!(a, 0);
        }
    }

    #[test]
    fn test_ring_find_patches() {
        let ring = Ring {
            theta_min: 0.0,
            theta_max: 3.0f32.to_radians(),
            phi_step: std::f32::consts::FRAC_PI_6,
            patch_count: 12,
            base_index: 0,
        };

        for i in 0..360 {
            let phi = Rads::from_degrees(i as f32);
            let (a, b) = ring.find_patch_indices(phi);
            println!("i = {}, phi = {}, a = {}, b = {}", i, phi, a, b);
        }
    }
}

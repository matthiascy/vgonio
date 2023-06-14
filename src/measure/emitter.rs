use crate::{
    measure::{measurement::Radius, rtc::Ray},
    units::{radians, steradians, Nanometres, Radians, SolidAngle},
    Handedness, RangeByStepSizeInclusive, SphericalCoord,
};
use glam::Vec3;
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Light emitter of the virtual gonio-photometer.
///
/// The light source is represented by a region over the spherical domain
/// defined by the `radius`. The shape of the region is defined by the
/// [`RegionShape`].
///
/// Note: need to update the radius for each surface before the measurement to
/// make sure that the surface is covered by the patch.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    /// Number of emitted rays.
    pub num_rays: u32,

    /// Max allowed bounces for each ray.
    pub max_bounces: u32,

    /// Distance from the emitter's center to the specimen's center.
    pub(crate) radius: Radius,

    /// Inclination angle (polar angle) of emitter's possible positions (center
    /// of the emitter) in spherical coordinates.
    pub zenith: RangeByStepSizeInclusive<Radians>,

    /// Azimuthal angle range of emitter's possible positions (center of the
    /// emitter) in spherical coordinates.
    pub azimuth: RangeByStepSizeInclusive<Radians>,

    /// Shape of the emitter.
    /// The shape is defined by the region over the spherical domain.
    pub shape: RegionShape,

    /// Light source's spectrum.
    pub spectrum: RangeByStepSizeInclusive<Nanometres>,

    /// Solid angle subtended by emitter's region.
    #[serde(skip)]
    pub(crate) solid_angle: SolidAngle,
}

/// Represents the shape of a region on the surface of a sphere.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum RegionShape {
    /// A patch has a disk shape on the surface of the sphere.
    #[serde(rename = "cap")]
    SphericalCap {
        /// Maximum zenith angle of the spherical cap.
        zenith: Radians,
    } = 0x00,
    /// A patch has a rectangular shape on the surface of the sphere.
    /// TODO: refactor
    #[serde(rename = "rect")]
    SphericalRect {
        /// Polar angle range of the patch (in radiansf).
        zenith: (Radians, Radians),

        /// Azimuthal angle range of the patch (in radians).
        azimuth: (Radians, Radians),
    } = 0x01,
    /// A patch has a disk shape, but the disk is not on the surface of the
    /// sphere. The radius of the disk is calculated according to the size
    /// of the surface.
    Disk {
        /// Radius of the disk.
        #[serde(skip)]
        radius: Radius,
    } = 0x02,
}

impl RegionShape {
    pub fn default_spherical_cap() -> Self {
        Self::SphericalCap {
            zenith: Radians::from_degrees(5.0),
        }
    }

    pub fn default_spherical_rect() -> Self {
        Self::SphericalRect {
            zenith: (Radians::from_degrees(5.0), Radians::from_degrees(10.0)),
            azimuth: (Radians::from_degrees(0.0), Radians::from_degrees(360.0)),
        }
    }

    /// Create a new spherical cap emitter.
    pub fn spherical_cap(zenith: Radians) -> Self { Self::SphericalCap { zenith } }

    /// Create a new spherical rectangle emitter.
    pub fn spherical_rect(zenith: (Radians, Radians), azimuth: (Radians, Radians)) -> Self {
        Self::SphericalRect { zenith, azimuth }
    }

    pub fn zenith(&self) -> Option<(Radians, Radians)> {
        match self {
            Self::SphericalCap { zenith } => Some((radians!(0.0), *zenith)),
            Self::SphericalRect { zenith, .. } => Some(*zenith),
            Self::Disk { .. } => None,
        }
    }

    /// Returns the mutable zenith angle of the spherical cap. Used for UI.
    pub fn cap_zenith_mut(&mut self) -> Option<&mut Radians> {
        match self {
            Self::SphericalCap { zenith } => Some(zenith),
            Self::SphericalRect { .. } => None,
            Self::Disk { .. } => None,
        }
    }

    /// Returns the mutable azimuth angle of the spherical cap. Used for UI.
    pub fn rect_zenith_mut(&mut self) -> Option<(&mut Radians, &mut Radians)> {
        match self {
            Self::SphericalCap { .. } => None,
            Self::SphericalRect { zenith, .. } => Some((&mut zenith.0, &mut zenith.1)),
            Self::Disk { .. } => None,
        }
    }

    pub fn rect_azimuth_mut(&mut self) -> Option<(&mut Radians, &mut Radians)> {
        match self {
            Self::SphericalCap { .. } => None,
            Self::SphericalRect { azimuth, .. } => Some((&mut azimuth.0, &mut azimuth.1)),
            Self::Disk { .. } => None,
        }
    }

    pub fn azimuth(&self) -> Option<(Radians, Radians)> {
        match self {
            Self::SphericalCap { .. } => {
                Some((radians!(0.0), radians!(2.0 * std::f32::consts::PI)))
            }
            Self::SphericalRect { azimuth, .. } => Some(*azimuth),
            Self::Disk { .. } => None,
        }
    }

    pub fn solid_angle(&self) -> SolidAngle {
        match self {
            Self::SphericalCap { zenith } => {
                let solid_angle: f32 = (zenith.cos() - 0.0) * (2.0 * std::f32::consts::PI);
                steradians!(solid_angle)
            }
            Self::SphericalRect { zenith, azimuth } => {
                todo!("calculate solid angle of the rectangular patch")
            }
            RegionShape::Disk { .. } => {
                todo!("calculate solid angle of the disk")
            }
        }
    }

    pub fn is_spherical_cap(&self) -> bool { matches!(self, Self::SphericalCap { .. }) }

    pub fn is_spherical_rect(&self) -> bool { matches!(self, Self::SphericalRect { .. }) }

    pub fn is_disk(&self) -> bool { matches!(self, Self::Disk { .. }) }
}

#[derive(Debug, Clone)]
pub struct EmitterSamples(Vec<Vec3>);

impl Deref for EmitterSamples {
    type Target = Vec<Vec3>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DerefMut for EmitterSamples {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl EmitterSamples {
    pub fn into_iter(self) -> impl Iterator<Item = Vec3> { self.0.into_iter() }
}

impl Emitter {
    /// Accesses the radius of the emitter.
    pub fn radius(&self) -> Radius { self.radius }

    /// Updates the radius of the emitter.
    pub fn set_radius(&mut self, radius: Radius) { self.radius = radius; }

    /// Returns the number of measured BSDF samples. Each sample is a
    /// measurement of the BSDF at a specific position of the emitter; it
    /// contains the measured value for each wavelength for each collector
    /// position.
    pub fn bsdf_data_samples_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// All possible measurement positions of the emitter.
    pub fn meas_points(&self) -> Vec<SphericalCoord> {
        log::trace!(
            "azimuth: {:?}",
            self.azimuth.values_wrapped().collect::<Vec<_>>()
        );
        log::trace!(
            "zenith: {:?}, {}",
            self.zenith.values().collect::<Vec<_>>(),
            self.zenith.step_count()
        );
        self.azimuth
            .values_wrapped()
            .flat_map(|azimuth| {
                self.zenith.values().map(move |zenith| SphericalCoord {
                    radius: 1.0,
                    zenith,
                    azimuth,
                })
            })
            .collect()
    }

    /// Generated samples inside emitter's region.
    pub fn generate_unit_samples(&self) -> EmitterSamples {
        let num_samples = self.num_rays as usize;

        let samples = if self.shape.is_disk() {
            uniform_sampling_on_unit_disk(num_samples, Handedness::RightHandedYUp)
        } else {
            // Generates uniformly distributed samples using rejection sampling.
            let (zenith_start, zenith_stop, azimuth_start, azimuth_stop) = {
                match self.shape {
                    RegionShape::SphericalCap { zenith } => (
                        radians!(0.0),
                        zenith,
                        radians!(0.0),
                        radians!(2.0 * std::f32::consts::PI),
                    ),
                    RegionShape::SphericalRect { zenith, azimuth } => {
                        (zenith.0, zenith.1, azimuth.0, azimuth.1)
                    }
                    _ => {
                        unreachable!("Only spherical cap and rectangular patch are supported")
                    }
                }
            };

            log::info!(
                "  - Generating {} samples in region θ: {}° ~ {}° φ: {}° ~ {}°",
                num_samples,
                zenith_start.in_degrees().value,
                zenith_stop.in_degrees().value,
                azimuth_start.in_degrees().value,
                azimuth_stop.in_degrees().value
            );

            uniform_sampling_on_unit_sphere(
                num_samples,
                zenith_start,
                zenith_stop,
                azimuth_start,
                azimuth_stop,
                Handedness::RightHandedYUp,
            )
        };

        EmitterSamples(samples)
    }

    /// Emits rays from the patch located at `pos`.
    pub fn emit_rays_with_radius(
        &self,
        samples: &EmitterSamples,
        pos: SphericalCoord,
        radius: f32,
    ) -> Vec<Ray> {
        log::trace!("Emitting rays from {} with radius = {}", pos, radius);
        let mat = glam::Mat3::from_axis_angle(glam::Vec3::Y, pos.zenith.value)
            * glam::Mat3::from_axis_angle(glam::Vec3::Z, pos.azimuth.value);
        let dir = -pos.to_cartesian(Handedness::RightHandedYUp);

        let rays = samples
            .par_iter()
            .map(|sample| {
                let origin = mat * *sample * radius;
                Ray::new(origin, dir)
            })
            .collect();
        rays
    }
}

/// Generates uniformly distributed samples on the unit sphere.
///
/// Depending on the handedness of the coordinate system, the samples are
/// generated differently.
///
/// 1. Right-handed Z-up coordinate system:
///
/// x = cos phi * sin theta
/// y = sin phi * sin theta
/// z = cos theta
///
/// 2. Right-handed Y-up coordinate system:
///
/// x = cos phi * sin theta
/// y = cos theta
/// z = sin phi * sin theta
pub fn uniform_sampling_on_unit_sphere(
    num: usize,
    theta_start: Radians,
    theta_stop: Radians,
    phi_start: Radians,
    phi_stop: Radians,
    handedness: Handedness,
) -> Vec<Vec3> {
    use std::f32::consts::PI;

    const SEED: u64 = 0;

    let range = Uniform::new(0.0, 1.0);
    let mut samples = Vec::with_capacity(num);
    samples.resize(num, glam::Vec3::ZERO);
    log::trace!("  - Generating samples following {:?}", handedness);

    match handedness {
        Handedness::RightHandedZUp => {
            samples
                .par_chunks_mut(8192)
                .enumerate()
                .for_each(|(i, chunks)| {
                    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
                    rng.set_stream(i as u64);

                    let mut j = 0;
                    while j < chunks.len() {
                        let phi = radians!(range.sample(&mut rng) * PI * 2.0);
                        let theta = radians!((1.0 - 2.0 * range.sample(&mut rng)).acos());
                        if (theta_start..theta_stop).contains(&theta)
                            && (phi_start..phi_stop).contains(&phi)
                        {
                            chunks[j] = glam::Vec3::new(
                                theta.sin() * phi.cos(),
                                theta.sin() * phi.sin(),
                                theta.cos(),
                            );
                            j += 1;
                        }
                    }
                });
        }
        Handedness::RightHandedYUp => {
            samples
                .par_chunks_mut(8192)
                .enumerate()
                .for_each(|(i, chunks)| {
                    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
                    rng.set_stream(i as u64);

                    let mut j = 0;
                    while j < chunks.len() {
                        let phi = radians!(range.sample(&mut rng) * PI * 2.0);
                        let theta = radians!((1.0 - 2.0 * range.sample(&mut rng)).acos());
                        if (theta_start..theta_stop).contains(&theta)
                            && (phi_start..phi_stop).contains(&phi)
                        {
                            chunks[j] = glam::Vec3::new(
                                theta.sin() * phi.cos(),
                                theta.cos(),
                                theta.sin() * phi.sin(),
                            );
                            j += 1;
                        }
                    }
                })
        }
    }

    samples
}

pub fn uniform_sampling_on_unit_disk(num: usize, handedness: Handedness) -> Vec<Vec3> {
    const SEED: u64 = 0;

    let range: Uniform<f32> = Uniform::new(0.0, 1.0);
    let mut samples = Vec::with_capacity(num);
    samples.resize(num, glam::Vec3::ZERO);

    match handedness {
        Handedness::RightHandedZUp => {
            todo!("uniform_sampling_on_unit_disk for RightHandedZUp")
        }
        Handedness::RightHandedYUp => {
            samples
                .par_chunks_mut(8192)
                .enumerate()
                .for_each(|(i, chunks)| {
                    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
                    rng.set_stream(i as u64);

                    chunks.iter_mut().for_each(|v| {
                        let r = range.sample(&mut rng).sqrt();
                        let a = range.sample(&mut rng) * std::f32::consts::TAU;
                        v.x = r * a.cos();
                        v.z = r * a.sin();
                    });
                })
        }
    }

    samples
}

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
    #[serde(rename = "rect")]
    SphericalRect {
        /// Polar angle range of the patch (in radians).
        zenith: (Radians, Radians),

        /// Azimuthal angle range of the patch (in radians).
        azimuth: (Radians, Radians),
    } = 0x01,
}

impl RegionShape {
    /// Create a new spherical cap emitter.
    pub fn spherical_cap(zenith: Radians) -> Self { Self::SphericalCap { zenith } }

    /// Create a new spherical rectangle emitter.
    pub fn spherical_rect(zenith: (Radians, Radians), azimuth: (Radians, Radians)) -> Self {
        Self::SphericalRect { zenith, azimuth }
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
        }
    }
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

impl Emitter {
    /// Accesses the radius of the emitter.
    pub fn radius(&self) -> Radius { self.radius }

    /// Updates the radius of the emitter.
    pub fn set_radius(&mut self, radius: Radius) { self.radius = radius; }

    /// All possible measurement positions of the emitter.
    pub fn meas_points(&self) -> Vec<SphericalCoord> {
        self.azimuth
            .values()
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
    ///
    /// Samples are generated uniformly distributed on the surface of the
    /// sphere.
    pub fn generate_samples(&self) -> EmitterSamples {
        let num_samples = self.num_rays as usize;
        // Generates uniformly distributed samples using rejection sampling.
        let (zenith_start, zenith_stop, azimuth_start, azimuth_stop) = {
            match self.shape {
                RegionShape::SphericalCap { zenith } => {
                    let zenith_range = 0.0..zenith.value;
                    let azimuth_range = 0.0..(2.0 * std::f32::consts::PI);
                    (
                        radians!(0.0),
                        zenith,
                        radians!(0.0),
                        radians!(2.0 * std::f32::consts::PI),
                    )
                }
                RegionShape::SphericalRect { zenith, azimuth } => {
                    (zenith.0, zenith.1, azimuth.0, azimuth.1)
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

        let t = std::time::Instant::now();
        let samples = uniform_sampling_on_unit_sphere(
            num_samples,
            zenith_start,
            zenith_stop,
            azimuth_start,
            azimuth_stop,
            Handedness::RightHandedYUp,
        );
        log::trace!("  - Samples: {:?}", samples);
        log::info!("  - Done in {} ms", t.elapsed().as_millis());
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
    num_samples: usize,
    theta_start: Radians,
    theta_stop: Radians,
    phi_start: Radians,
    phi_stop: Radians,
    handedness: Handedness,
) -> Vec<glam::Vec3> {
    use std::f32::consts::PI;

    const SEED: u64 = 0;

    let range = Uniform::new(0.0, 1.0);
    let mut samples = Vec::with_capacity(num_samples);
    samples.resize(num_samples, glam::Vec3::ZERO);
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

use crate::{
    measure::{measurement::Radius, rtc::Ray, SphericalTransform},
    RangeByStepSizeInclusive,
};
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};
use vgcore::{
    math::{Mat3, Sph3, Vec3},
    units::{radians, steradians, Nanometres, Radians, SolidAngle},
};
use vgsurf::MicroSurfaceMesh;

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
        /// Maximum zenith angle of the spherical cap. This is half of the
        /// opening angle of the cap.
        zenith: Radians,
    } = 0x00,
    /// A patch has a rectangular shape on the surface of the sphere.
    /// TODO: refactor
    #[serde(rename = "rect")]
    SphericalRect {
        /// Polar angle range of the patch (in radians).
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

    // TODO: implement DISK
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
                log::error!("TODO: calculate solid angle of the disk");
                steradians!(1.0)
            }
        }
    }

    pub fn is_spherical_cap(&self) -> bool { matches!(self, Self::SphericalCap { .. }) }

    pub fn is_spherical_rect(&self) -> bool { matches!(self, Self::SphericalRect { .. }) }

    pub fn is_disk(&self) -> bool { matches!(self, Self::Disk { .. }) }

    pub fn disk_radius(&self) -> Option<Radius> {
        match self {
            Self::Disk { radius } => Some(*radius),
            _ => None,
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

impl EmitterSamples {
    /// Consumes the samples and returns an interator over the samples.
    pub fn into_iter(self) -> impl Iterator<Item = Vec3> { self.0.into_iter() }
}

impl Emitter {
    /// Accesses the radius of the emitter.
    pub fn orbit_radius(&self) -> Radius { self.radius }

    /// Updates the radius of the emitter.
    pub fn set_orbit_radius(&mut self, radius: Radius) { self.radius = radius; }

    /// Returns the number of measured BSDF samples. Each sample is a
    /// measurement of the BSDF at a specific position of the emitter; it
    /// contains the measured value for each wavelength for each collector
    /// position.
    pub fn samples_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// All possible measurement positions of the emitter.
    pub fn measurement_points(&self) -> Vec<Sph3> {
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
                self.zenith
                    .values()
                    .map(move |zenith| Sph3::new(1.0, zenith, azimuth))
            })
            .collect()
    }

    /// Generated samples inside emitter's region.
    ///
    /// The samples are generated in the local coordinate system of the emitter.
    pub fn generate_unit_samples(&self) -> EmitterSamples {
        let num_samples = self.num_rays as usize;

        let samples = if self.shape.is_disk() {
            crate::measure::uniform_sampling_on_unit_disk(num_samples)
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
                zenith_start.in_degrees().value(),
                zenith_stop.in_degrees().value(),
                azimuth_start.in_degrees().value(),
                azimuth_stop.in_degrees().value()
            );

            crate::measure::uniform_sampling_on_unit_sphere(
                num_samples,
                zenith_start,
                zenith_stop,
                azimuth_start,
                azimuth_stop,
            )
        };

        EmitterSamples(samples)
    }

    /// Estimates the distance from the emitter to the specimen surface.
    pub fn estimate_orbit_radius(&self, mesh: &MicroSurfaceMesh) -> f32 {
        self.radius.estimate(mesh)
    }

    /// Estimates the radius of the emitter's disk if it is a disk.
    pub fn estimate_disk_radius(&self, mesh: &MicroSurfaceMesh) -> Option<f32> {
        match self.shape {
            RegionShape::SphericalCap { .. } | RegionShape::SphericalRect { .. } => None,
            RegionShape::Disk { radius } => Some(radius.estimate_disk_radius(mesh)),
        }
    }

    /// Applies the transformation matrix for the emitter's samples at the
    /// given position in spherical coordinates to the samples.
    pub fn transform_samples(
        sph_pos: Sph3,
        orbit_radius: f32,
        shape_radius: Option<f32>,
        samples: &EmitterSamples,
    ) -> Vec<Vec3> {
        match shape_radius {
            None => {
                let transform = SphericalTransform::transform_cap(sph_pos, orbit_radius);
                samples.par_iter().map(move |s| transform * *s).collect()
            }
            Some(disk_radius) => {
                let transform =
                    SphericalTransform::transform_disk(sph_pos, disk_radius, orbit_radius);
                samples
                    .par_iter()
                    .map(move |s| transform * Vec3::new(s.x * sph_pos.theta.cos(), s.y, s.z))
                    .collect()
            }
        }
    }

    /// Emits rays from the patch located at `pos` with `orbit_radius`.
    /// If the emitter is a disk, `shape_radius` is used to generate rays on the
    /// disk.
    ///
    /// # Arguments
    ///
    /// * `samples` - The samples on the emitter's surface.
    /// * `sph_pos` - Current position of the emitter in spherical coordinates.
    /// * `orbit_radius` - The radius of the orbit.
    /// * `shape_radius` - The radius of the emitter's disk if it is a disk.
    pub fn emit_rays(
        samples: &EmitterSamples,
        sph_pos: Sph3,
        orbit_radius: f32,
        shape_radius: Option<f32>,
    ) -> Vec<Ray> {
        log::trace!(
            "[Emitter] emitting rays from {} with orbit radius = {}, disk radius = {:?}",
            sph_pos,
            orbit_radius,
            shape_radius
        );
        let transform = SphericalTransform::transform_to(sph_pos);
        let dir = -sph_pos.to_cartesian();
        log::trace!("[Emitter] emitting rays with dir = {:?}", dir);

        Self::transform_samples(sph_pos, orbit_radius, shape_radius, samples)
            .into_iter()
            .map(|origin| Ray::new(origin, dir))
            .collect()
    }
}

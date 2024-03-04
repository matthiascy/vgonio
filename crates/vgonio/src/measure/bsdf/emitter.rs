//! Light source of the measurement system.

use crate::{
    measure::{bsdf::rtc::Ray, SphericalTransform},
    RangeByStepSizeInclusive,
};
use base::{
    math::{Sph2, Vec3},
    units::{deg, nm, rad, Nanometres, Radians},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use surf::MicroSurfaceMesh;

/// Parameters for the emitter.
///
/// The light source is represented by a disk tangent to the hemisphere around
/// which the emitter is rotating. The orbit radius of the emitter and the disk
/// radius are estimated according to the size of the surface to be measured.
/// See [`crate::measure::estimate_orbit_radius`] and
/// [`crate::measure::estimate_shape_radius`].
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmitterParams {
    /// Number of emitted rays.
    pub num_rays: u32,

    /// Max allowed bounces for each ray.
    pub max_bounces: u32,

    /// Inclination angle (polar angle) of emitter's possible positions (center
    /// of the emitter) in spherical coordinates.
    pub zenith: RangeByStepSizeInclusive<Radians>,

    /// Azimuthal angle range of emitter's possible positions (center of the
    /// emitter) in spherical coordinates.
    pub azimuth: RangeByStepSizeInclusive<Radians>,

    /// Light source's spectrum.
    pub spectrum: RangeByStepSizeInclusive<Nanometres>,
}

impl Default for EmitterParams {
    fn default() -> Self {
        EmitterParams {
            num_rays: 2 ^ 20,
            max_bounces: 8,
            zenith: RangeByStepSizeInclusive::new(
                rad!(0.0),
                Radians::HALF_PI,
                deg!(2.0).to_radians(),
            ),
            azimuth: RangeByStepSizeInclusive::new(
                rad!(0.0),
                Radians::TWO_PI,
                deg!(5.0).to_radians(),
            ),
            spectrum: RangeByStepSizeInclusive::new(nm!(400.0), nm!(700.0), nm!(100.0)),
        }
    }
}

impl EmitterParams {
    /// Emitter's possible positions in spherical coordinates.
    pub fn generate_measurement_points(&self) -> MeasurementPoints {
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
                    .map(move |zenith| Sph2::new(zenith, azimuth))
            })
            .collect()
    }

    /// Generated samples inside emitter's region.
    ///
    /// The samples are generated in the local coordinate system of the emitter.
    pub fn generate_unit_samples(&self) -> EmitterSamples {
        EmitterSamples(crate::measure::uniform_sampling_on_unit_disk(
            self.num_rays as usize,
        ))
    }

    /// Returns the number of measurement positions.
    pub fn measurement_points_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// Transforms the samples from the sampling space to the desired position
    /// in the world coordinate system.
    pub(crate) fn transform_samples(
        samples: &EmitterSamples,
        dest: Sph2,
        orbit_radius: f32,
        disc_radius: f32,
    ) -> Vec<Vec3> {
        let transform = SphericalTransform::transform_disc(dest, disc_radius, orbit_radius);
        samples
            .par_iter()
            .map(move |s| transform * Vec3::new(s.x * dest.theta.cos(), s.y, s.z))
            .collect()
    }

    /// Emits rays from the samples
    pub(crate) fn emit_rays(
        samples: &EmitterSamples,
        dest: Sph2,
        orbit_radius: f32,
        disc_radius: f32,
    ) -> Vec<Ray> {
        let dir = -dest.to_cartesian();
        let transform = SphericalTransform::transform_disc(dest, disc_radius, orbit_radius);
        samples
            .par_iter()
            .map(move |s| Ray::new(transform * Vec3::new(s.x * dest.theta.cos(), s.y, s.z), dir))
            .collect()
    }
}

/// Emitter's samples in the sampling space.
#[derive(Debug, Clone)]
pub struct EmitterSamples(Vec<Vec3>);

impl Deref for EmitterSamples {
    type Target = Vec<Vec3>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl FromIterator<Vec3> for EmitterSamples {
    fn from_iter<T: IntoIterator<Item = Vec3>>(iter: T) -> Self {
        EmitterSamples(iter.into_iter().collect())
    }
}

/// Emitter's possible positions in spherical coordinates.
#[derive(Debug, Clone)]
pub struct MeasurementPoints(Box<[Sph2]>);

impl Deref for MeasurementPoints {
    type Target = [Sph2];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl FromIterator<Sph2> for MeasurementPoints {
    fn from_iter<T: IntoIterator<Item = Sph2>>(iter: T) -> Self {
        MeasurementPoints(iter.into_iter().collect::<Vec<_>>().into_boxed_slice())
    }
}

impl MeasurementPoints {
    /// Returns an iterator over the measurement points.
    pub fn iter(&self) -> impl Iterator<Item = &Sph2> { self.0.iter() }

    /// Returns the number of measurement points.
    pub fn len(&self) -> usize { self.0.len() }

    /// Returns `true` if the measurement points is empty.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

impl IntoIterator for MeasurementPoints {
    type Item = Sph2;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter { Vec::from(self.0).into_iter() }
}

/// Emitter constructed from the parameters.
#[derive(Debug, Clone)]
pub struct Emitter {
    /// Parameters of the emitter.
    pub params: EmitterParams,
    /// Emitter's possible positions in spherical coordinates.
    pub measpts: MeasurementPoints,
    /// Generated samples inside emitter's region.
    pub samples: EmitterSamples,
}

impl Emitter {
    /// Constructs an emitter from the parameters.
    pub fn new(params: &EmitterParams) -> Self {
        let measpts = params.generate_measurement_points();
        let samples = params.generate_unit_samples();
        Emitter {
            params: *params,
            measpts,
            samples,
        }
    }

    /// Transforms the samples from the sampling space to the emitter's local
    /// coordinate system.
    pub fn samples_at(&self, pos: Sph2, mesh: &MicroSurfaceMesh) -> Vec<Vec3> {
        let disk_radius = crate::measure::estimate_disc_radius(mesh);
        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        EmitterParams::transform_samples(&self.samples, pos, orbit_radius, disk_radius)
    }

    /// Emits rays from the emitter at `pos` covering the whole surface.
    pub fn emit_rays(&self, pos: Sph2, mesh: &MicroSurfaceMesh) -> Vec<Ray> {
        let disk_radius = crate::measure::estimate_disc_radius(mesh);
        let orbit_radius = crate::measure::estimate_orbit_radius(mesh);
        log::trace!(
            "[Emitter] emitting rays from {} with orbit radius = {}, disk radius = {:?}",
            pos,
            orbit_radius,
            disk_radius
        );
        let dir = -pos.to_cartesian();
        log::trace!("[Emitter] emitting rays with dir = {:?}", dir);

        self.samples_at(pos, mesh)
            .into_iter()
            .map(|origin| Ray::new(origin, dir))
            .collect()
    }
}

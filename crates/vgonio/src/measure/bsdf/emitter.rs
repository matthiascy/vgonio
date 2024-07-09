//! Light source of the measurement system.

use crate::{
    app::cli::ansi,
    measure::{bsdf::rtc::Ray, SphericalTransform},
};
use base::{
    math::{Sph2, Vec3},
    range::RangeByStepSizeInclusive,
    units::{deg, nm, rad, Nanometres, Radians, Rads},
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{env::temp_dir, fs::File, ops::Deref, rc::Rc};
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
    pub num_rays: u64,

    /// Number of circular sectors on the unit sample disk.
    /// Not saved in the vgonio file.
    pub num_sectors: u32,

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
            num_sectors: 1,
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
    pub fn generate_unit_samples(
        &self,
        tstart: Rads,
        tstop: Rads,
        num_rays: usize,
    ) -> EmitterSamples {
        log::debug!(
            "[Emitter] generating {} samples in the range [{}, {}]",
            num_rays,
            tstart.prettified(),
            tstop.prettified()
        );
        let mut samples = vec![Vec3::ZERO; num_rays].into_boxed_slice();
        crate::measure::uniform_sampling_on_unit_disk(&mut samples, tstart, tstop);
        EmitterSamples(samples)
    }

    /// Returns the number of measurement positions.
    pub fn measurement_points_count(&self) -> usize {
        self.azimuth.step_count_wrapped() * self.zenith.step_count_wrapped()
    }

    /// Returns the number of measurement points in the zenith direction.
    pub fn measurement_points_zenith_count(&self) -> usize { self.zenith.step_count_wrapped() }

    /// Returns the number of measurement points in the azimuth direction.
    pub fn measurement_points_azimuth_count(&self) -> usize { self.azimuth.step_count_wrapped() }

    /// Transforms the samples from the sampling space to the desired position
    /// in the world coordinate system.
    pub(crate) fn transform_samples(
        samples: &[Vec3],
        dest: Sph2,
        orbit_radius: f32,
        disc_radius: f32,
    ) -> Vec<Vec3> {
        let transform = SphericalTransform::transform_disc(dest, disc_radius, orbit_radius);
        let mut transformed = samples
            .par_iter()
            .map(move |s| transform * Vec3::new(s.x * dest.theta.cos(), s.y, s.z))
            .collect::<Vec<_>>();
        transformed.shrink_to_fit();
        transformed
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
pub struct EmitterSamples(Box<[Vec3]>);

impl EmitterSamples {
    /// Returns an empty set of samples.
    pub fn empty() -> Self { EmitterSamples(Box::new([])) }
}

impl Deref for EmitterSamples {
    type Target = [Vec3];

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl FromIterator<Vec3> for EmitterSamples {
    fn from_iter<T: IntoIterator<Item = Vec3>>(iter: T) -> Self {
        EmitterSamples(iter.into_iter().collect::<Vec<_>>().into_boxed_slice())
    }
}

/// Emitter's possible positions in spherical coordinates.
#[derive(Debug, Clone)]
pub struct MeasurementPoints(pub(crate) Box<[Sph2]>);

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

/// Emitter with samples for partial measurements.
pub struct Emitter {
    /// Parameters of the emitter.
    pub params: EmitterParams,
    /// Emitter's possible positions in spherical coordinates.
    pub measpts: MeasurementPoints,
    /// The number of rays per sector.
    pub num_rays_per_sector: u32,
}

/// A single circular sector of the emitter.
pub struct EmitterCircularSector<'a> {
    /// Parameters of the emitter.
    pub params: &'a EmitterParams,
    /// Measurement points inherent to the emitter.
    pub measpts: &'a MeasurementPoints,
    /// Generated samples inside one sector of the emitter's region.
    pub samples: EmitterSamples,
    /// The index of the sector.
    pub idx: u32,
}

/// Circular sectors of the emitter.
pub struct EmitterCircularSectors<'a> {
    /// The emitter where the sectors are generated.
    emitter: &'a Emitter,
    /// The angle of the sector.
    sector_angle: Rads,
    /// The index of the current sector.
    /// The length of the iterator is equal to the number of sectors.
    /// See [`EmitterParams::num_sectors`].
    idx: u32,
}

impl<'a> EmitterCircularSector<'a> {
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

impl Emitter {
    /// Creates a new emitter with the given parameters.
    pub fn new(params: &EmitterParams) -> Self {
        let num_rays_per_sector =
            (params.num_rays as f64 / params.num_sectors as f64).ceil() as u32;
        let measpts = params.generate_measurement_points();
        println!(
            "      {}>{} Dividing the emitter into {} sectors, {} rays per sector",
            ansi::BRIGHT_YELLOW,
            ansi::RESET,
            params.num_sectors,
            num_rays_per_sector
        );
        Self {
            params: *params,
            measpts,
            num_rays_per_sector,
        }
    }

    /// Returns an iterator over the circular sectors of the emitter.
    pub fn circular_sectors(&self) -> EmitterCircularSectors {
        EmitterCircularSectors {
            emitter: self,
            sector_angle: Rads::TAU / self.params.num_sectors as f32,
            idx: 0,
        }
    }
}

impl<'a> Iterator for EmitterCircularSectors<'a> {
    type Item = EmitterCircularSector<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.emitter.params.num_sectors {
            let start = self.idx as f32 * self.sector_angle;
            let stop = (self.idx + 1) as f32 * self.sector_angle;
            let sector_idx = self.idx;
            self.idx += 1;

            log::debug!(
                "[Emitter] sector {} with angle [{}, {}]",
                sector_idx,
                start.prettified(),
                stop.prettified()
            );

            Some(EmitterCircularSector {
                params: &self.emitter.params,
                measpts: &self.emitter.measpts,
                samples: self.emitter.params.generate_unit_samples(
                    start,
                    stop,
                    self.emitter.num_rays_per_sector as usize,
                ),
                idx: sector_idx,
            })
        } else {
            None
        }
    }
}

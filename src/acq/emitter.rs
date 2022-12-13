use crate::acq::{
    measurement::Radius,
    radians, steradians,
    util::{RangeByStepSize, SphericalCoord},
    Nanometres, Radians, Ray, SolidAngle,
};
use serde::{Deserialize, Serialize};

/// Light emitter of the virtual gonio-photometer.
///
/// The light source is represented by a region over the spherical domain
/// defined by the `radius`. The shape of the region is defined by the
/// [`RegionShape`].
///
/// Note: need to update the radius for each surface before the measurement to
/// make sure that the surface is covered by the patch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    /// Number of emitted rays.
    pub num_rays: u32,

    /// Max allowed bounces for each ray.
    pub max_bounces: u32,

    /// Distance from the emitter's center to the specimen's center.
    pub radius: Radius,

    /// Inclination angle (polar angle) of emitter's possible positions in
    /// spherical coordinates.
    pub zenith: RangeByStepSize<Radians>,

    /// Azimuthal angle range of emitter's possible positions in spherical
    /// coordinates.
    pub azimuth: RangeByStepSize<Radians>,

    /// Shape of the emitter.
    /// The shape is defined by the region over the spherical domain.
    pub shape: RegionShape,

    /// Light source's spectrum.
    pub spectrum: RangeByStepSize<Nanometres>,

    /// Solid angle subtended by the emitter.
    #[serde(skip)]
    pub solid_angle: SolidAngle,

    /// Samples generated for the patch.
    #[serde(skip)]
    pub samples: Vec<glam::Vec3>,
}

/// Represents the shape of a region on the surface of a sphere.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionShape {
    /// A patch has a disk shape on the surface of the sphere.
    #[serde(rename = "disk")]
    SphericalCap {
        /// Maximum zenith angle of the spherical cap.
        zenith: Radians,
    },
    /// A patch has a rectangular shape on the surface of the sphere.
    #[serde(rename = "rect")]
    SphericalRect {
        /// Polar angle range of the patch (in radians).
        zenith: (Radians, Radians),

        /// Azimuthal angle range of the patch (in radians).
        azimuth: (Radians, Radians),
    },
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
                todo!()
                // let solid_angle: f32 = (zenith.0.cos() - zenith.1.cos()) *
                // (azimuth.1 - azimuth.0); steradians!
                // (solid_angle)
            }
        }
    }
}

impl Emitter {
    pub fn set_radius(&mut self, radius: Radius) { self.radius = radius; }

    /// All possible positions of the emitter.
    pub fn positions(&self) -> Vec<SphericalCoord> {
        let n_zenith =
            ((self.zenith.stop - self.zenith.start) / self.zenith.step_size).ceil() as usize;
        let n_azimuth =
            ((self.azimuth.stop - self.azimuth.start) / self.azimuth.step_size).ceil() as usize;

        (0..n_zenith)
            .into_iter()
            .flat_map(|i_theta| {
                (0..n_azimuth).into_iter().map(move |i_phi| SphericalCoord {
                    zenith: self.zenith.start + i_theta as f32 * self.zenith.step_size,
                    azimuth: self.azimuth.start + i_phi as f32 * self.azimuth.step_size,
                })
            })
            .collect()
    }

    /// Emits rays from the patch located at `pos`.
    pub fn emit_rays(&self, pos: SphericalCoord) -> Vec<Ray> {
        let r = self.radius.value();
        let mat = glam::Mat3::from_axis_angle(glam::Vec3::Y, pos.zenith.value)
            * glam::Mat3::from_axis_angle(glam::Vec3::Z, pos.azimuth.value);
        let dir = -pos.into_cartesian();

        self.samples
            .iter()
            .map(|sample| {
                let origin = mat * *sample * r.value;
                Ray { o: origin, d: dir }
            })
            .collect()
    }

    pub fn init(&mut self) {
        // TODO
    }
}

// TODO: update
/// Generates uniformly distributed samples on the unit sphere.
/// Right-handed Z-up coordinate system is used.
///
/// x = cos phi * sin theta
/// y = sin phi * sin theta
/// z = cos theta
pub fn uniform_sampling_on_unit_sphere(
    num_samples: u32,
    theta_start: Radians,
    theta_stop: Radians,
    phi_start: Radians,
    phi_stop: Radians,
) -> Vec<glam::Vec3> {
    use rand_distr::Distribution;
    use std::f32::consts::PI;

    let mut rng = rand::thread_rng();
    let uniform = rand_distr::Uniform::new(0.0, 1.0);
    let mut samples = vec![];

    let mut i = 0;
    while i < num_samples {
        let phi = radians!(uniform.sample(&mut rng) * PI * 2.0);
        let theta = radians!((1.0 - 2.0 * uniform.sample(&mut rng)).acos());

        if (theta_start..theta_stop).contains(&theta) && (phi_start..phi_stop).contains(&phi) {
            samples.push(glam::Vec3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ));
            i += 1;
        }
    }

    samples
}

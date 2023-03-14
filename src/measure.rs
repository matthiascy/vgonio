//! Acquisition related.

pub mod bsdf;
mod collector;
pub(crate) mod emitter; // TODO: maybe make private
pub mod measurement;
pub mod microfacet;
pub mod rtc;
pub mod scattering;

pub use collector::{Collector, CollectorScheme, Patch};
pub use emitter::Emitter;

use std::str::FromStr;

use crate::{
    app::gfx::camera::{Projection, ProjectionKind},
    common::Handedness,
    units::Radians,
    Error,
};
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec3A};

/// Representation of a ray.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    /// The origin of the ray.
    pub o: Vec3,

    /// The direction of the ray.
    pub d: Vec3,
}

impl Ray {
    /// Create a new ray (direction will be normalised).
    pub fn new(o: Vec3, d: Vec3) -> Self {
        let d = d.normalize();
        // let inv_dir_z = 1.0 / d.z;
        // let kz = Axis::max_axis(d.abs());
        // let kx = kz.next_axis();
        // let ky = kz.next_axis();
        Self { o, d }
    }
}

/// Enumeration of the different ways to trace rays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RtcMethod {
    /// Ray tracing using Intel's Embree library.
    #[cfg(feature = "embree")]
    Embree,
    /// Ray tracing using Nvidia's OptiX library.
    #[cfg(feature = "optix")]
    Optix,
    /// Customised grid ray tracing method.
    Grid,
}

/// Struct used to record ray path.
#[derive(Debug, Copy, Clone)]
pub struct TrajectoryNode {
    /// The ray of the node.
    pub ray: Ray,

    /// The cosine of the angle between the ray and the normal (always
    /// positive).
    pub cos: f32,
}

/// Ray tracing record.
#[derive(Debug)]
pub struct RtcRecord {
    /// Path of traced ray.
    pub trajectory: Vec<TrajectoryNode>,

    /// Energy of the ray with different wavelengths at each bounce.
    /// Inner vector is the energy of the ray of different wavelengths.
    /// Outer vector is the number of bounces.
    pub energy_each_bounce: Vec<Vec<f32>>,
}

impl RtcRecord {
    /// Returns the bounces of traced ray.
    pub fn bounces(&self) -> usize { self.trajectory.len() - 1 }
}

/// Medium of the surface.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum Medium {
    /// Air.
    Air,
    /// Vacuum.
    Vacuum,
    /// Aluminium.
    Aluminium,
    /// Copper.
    Copper,
}

impl FromStr for Medium {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "air" => Ok(Self::Air),
            "vacuum" => Ok(Self::Vacuum),
            "al" => Ok(Self::Aluminium),
            "cu" => Ok(Self::Copper),
            &_ => Err(Error::Any("unknown medium".to_string())),
        }
    }
}

/// Light source used for acquisition of shadowing and masking function.
pub struct LightSource {
    /// Position of the light source.
    pub pos: Vec3A,
    /// Parameters of projection.
    pub proj: Projection,
    /// Type of projection.
    pub proj_kind: ProjectionKind,
}

/// Light space matrix data uploaded to GPU during generation of depth map.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LightSourceRaw([f32; 16]);

impl LightSource {
    /// Returns the GPU data of the light source.
    pub fn to_raw(&self) -> LightSourceRaw {
        let forward = -self.pos;
        let up = if forward == -Vec3A::Y {
            Vec3A::new(1.0, 1.0, 0.0).normalize()
        } else {
            Vec3A::Y
        };
        let view = glam::Mat4::look_at_rh(self.pos.into(), Vec3::ZERO, up.into());
        let proj = self.proj.matrix(self.proj_kind);
        LightSourceRaw((proj * view).to_cols_array())
    }
}

/// Conversion from spherical coordinate system to cartesian coordinate system.
///
/// # Arguments
///
/// * `r` - radius
/// * `zenith` - polar angle
/// * `azimuth` - azimuthal angle
/// * `handedness` - handedness of the cartesian coordinate system
pub fn spherical_to_cartesian(
    r: f32,
    zenith: Radians,
    azimuth: Radians,
    handedness: Handedness,
) -> Vec3 {
    let a = r * zenith.sin() * azimuth.cos();
    let b = r * zenith.sin() * azimuth.sin();
    let c = r * zenith.cos();

    match handedness {
        Handedness::RightHandedZUp => Vec3::new(a, b, c),
        Handedness::RightHandedYUp => Vec3::new(a, c, b),
    }
}

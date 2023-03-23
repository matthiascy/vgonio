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

/// Struct used to record the trajectory of a ray.
#[derive(Debug, Copy, Clone)]
pub struct TrajectoryNode {
    /// The ray of the node.
    pub ray: Ray,

    /// The cosine of the incident angle between the ray and the normal (always
    /// positive) of the surface where the ray hits.
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

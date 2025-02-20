//! Utility functions and types temporarily stored here.

#[cfg(feature = "winit")]
pub mod input;
pub mod medium;
pub mod partition;
pub mod range;

use crate::{
    math::{Sph2, Sph3},
    units::{rad, Radians},
};
use chrono::{DateTime, Local};
use glam::{Mat3, Vec3};
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    prelude::ParallelSliceMut,
};

/// Returns the current time as an ISO 8601 (RFC 3339) timestamp.
pub fn iso_timestamp() -> String {
    Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

/// Returns the current time as an ISO 8601 (RFC 3339) timestamp without the
/// timezone and the colon in the time field.
pub fn iso_timestamp_short(datetime: DateTime<Local>) -> String {
    datetime.format("%Y-%m-%dT%H-%M-%S").to_string()
}

/// Converts a date time to an ISO 8601 (RFC 3339) timestamp.
pub fn iso_timestamp_from_datetime(dt: &DateTime<Local>) -> String {
    dt.to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

/// Converts a date time to an ISO 8601 (RFC 3339) timestamp without the
/// timezone and with the colon in the time field.
pub fn iso_timestamp_display(dt: &DateTime<Local>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Helper structure dealing with the spherical transform related to the
/// acquisition.
pub struct SphericalTransform;

impl SphericalTransform {
    /// Returns the transformation matrix transforming something from the
    /// local coordinate system to the desired position defined by the
    /// spherical coordinates.
    ///
    /// The local coordinate system is a right-handed system with the z-axis
    /// pointing upwards, the x-axis pointing to the right and the y-axis
    /// pointing forward.
    ///
    /// # Arguments
    ///
    /// * `dest` - The desired position in spherical coordinates; it should
    ///   always be defined in the unit sphere.
    ///
    /// # Returns
    ///
    /// The transformation matrix.
    pub fn transform_to(dest: Sph2) -> Mat3 {
        Mat3::from_axis_angle(Vec3::Z, dest.phi.value())
            * Mat3::from_axis_angle(Vec3::Y, dest.theta.value())
    }

    /// Returns the transformation matrix transforming the spherical cap shape
    /// or samples to the desired position defined by the spherical
    /// coordinates.
    ///
    /// See [`uniform_sampling_on_unit_sphere`] for more information about the
    /// spherical cap samples.
    ///
    /// # Arguments
    ///
    /// * `dest` - The desired position in spherical coordinates; should always
    ///   be defined in the unit sphere.
    /// * `orbit_radius` - The radius of the orbit about which the samples are
    ///   rotating around.
    pub fn transform_cap(dest: Sph2, orbit_radius: f32) -> Mat3 {
        Self::transform_to(dest) * Mat3::from_diagonal(Vec3::splat(orbit_radius))
    }

    /// Returns the transformation matrix transforming the disk shape or samples
    /// to the desired position defined by the spherical coordinates.
    ///
    /// See [`uniform_sampling_on_unit_disk`] for more information about the
    /// disk samples.
    ///
    /// # Arguments
    ///
    /// * `dest` - The desired position in spherical coordinates; should always
    ///   be defined in the unit sphere.
    /// * `disk_radius` - The radius of the disk on which the samples are
    ///   distributed.
    /// * `orbit_radius` - The radius of the orbit about which the samples are
    ///   rotating around.
    pub fn transform_disc(dest: Sph2, disk_radius: f32, orbit_radius: f32) -> Mat3 {
        Self::transform_to(dest)
            * Mat3::from_diagonal(Vec3::new(disk_radius, disk_radius, orbit_radius))
    }
}

/// Generates uniformly distributed samples on the unit disk.
///
/// NOTE: samples are generated on the xy-plane, with the z component set to 1.0
/// in the returned vector. This could simplify the transformation of the
/// samples when rotating the disk.
pub fn uniform_sampling_on_unit_disk(samples: &mut [Vec3], tstart: Radians, tstop: Radians) {
    const SEED: u64 = 0;

    let range: Uniform<f32> = Uniform::new(0.0, 1.0);
    samples
        .par_chunks_mut(8192)
        .enumerate()
        .for_each(|(i, chunks)| {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED);
            rng.set_stream(i as u64);

            chunks.iter_mut().for_each(|v| {
                let r = range.sample(&mut rng).sqrt();
                let a = range.sample(&mut rng) * (tstop - tstart).as_f32() + tstart.as_f32();
                *v = Vec3::new(r * a.cos(), r * a.sin(), 1.0);
            });
        });
}

/// Generates uniformly distributed samples on the unit sphere.
///
/// TODO: sampling with pdf
///
/// The samples are generated on the unit sphere, in the right-handed,
/// Z-up coordinate system.
///
/// x = cos phi * sin theta
/// y = sin phi * sin theta
/// z = cos theta
pub fn uniform_sampling_on_unit_sphere(
    num: usize,
    theta_start: Radians,
    theta_stop: Radians,
    phi_start: Radians,
    phi_stop: Radians,
) -> Box<[Vec3]> {
    const SEED: u64 = 0;
    let range = Uniform::new(0.0, 1.0);
    let mut samples = Box::new_uninit_slice(num);
    log::trace!("  - Generating samples on unit sphere");

    samples
        .par_chunks_mut(8192)
        .enumerate()
        .for_each(|(i, chunks)| {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED);
            rng.set_stream(i as u64);

            let mut j = 0;
            while j < chunks.len() {
                let phi = rad!(range.sample(&mut rng) * std::f32::consts::TAU);
                let theta = rad!((1.0 - 2.0 * range.sample(&mut rng)).acos());
                if (theta_start..theta_stop).contains(&theta)
                    && (phi_start..phi_stop).contains(&phi)
                {
                    chunks[j].write(Sph3::new(1.0, theta, phi).to_cartesian());
                    j += 1;
                }
            }
        });

    unsafe { samples.assume_init() }
}

/// ANSI color codes.
pub mod ansi {
    pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
    pub const BRIGHT_RED: &str = "\u{001b}[31m";
    pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
    pub const RESET: &str = "\u{001b}[0m";

    pub const RED_EXCLAMATION: &str = "\u{001b}[31m!\u{001b}[0m";
    pub const CYAN_CHECK: &str = "\u{001b}[36m✓\u{001b}[0m";

    pub const YELLOW_GT: &str = "\u{001b}[33m>\u{001b}[0m";

    pub const CYAN_MINUS: &str = "\u{001b}[36m-\u{001b}[0m";
}

//! Acquisition related.

pub mod bsdf;
pub mod mfd;
pub mod params;

use crate::{
    app::cli::ansi::YELLOW_GT,
    measure::mfd::{MeasuredNdfData, MeasuredSdfData},
};
use chrono::{DateTime, Local};
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    prelude::ParallelSliceMut,
};
use std::{
    ffi::OsStr,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use surf::{MicroSurface, MicroSurfaceMesh};
use vgonio_bxdf::brdf::measured::{rgl::RglBrdf, ClausenBrdf, MerlBrdf, VgonioBrdf, Yan18Brdf};
use vgonio_core::{
    asset,
    error::VgonioError,
    io::{
        Header, HeaderMeta, ReadFileError, ReadFileErrorKind, WriteFileError, WriteFileErrorKind,
    },
    math::{self, Mat3, Sph2, Sph3, Vec3},
    res::{AssetTypeId, Handle},
    units::{rad, Radians},
    utils::{
        medium::Medium,
        partition::{SphericalDomain, SphericalPartition},
    },
    AnyMeasured, MeasurementKind, Version,
};
use vgonio_meas::{
    io::{
        vgmo::{vgmo_header_ext_from_data, VgmoHeaderExt},
        OutputFileFormatOption,
    },
    mfd::MeasuredNdfData,
};

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

/// Estimates the radius of the sphere or hemisphere enclosing the specimen.
#[inline(always)]
pub fn estimate_orbit_radius(mesh: &MicroSurfaceMesh) -> f32 {
    mesh.bounds.max_extent() * std::f32::consts::SQRT_2
}

/// Estimates the radius of the disk with the area covering the specimen.
#[inline(always)]
pub fn estimate_disc_radius(mesh: &MicroSurfaceMesh) -> f32 { mesh.bounds.max_extent() * 0.7 }

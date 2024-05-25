//! Acquisition related.

pub mod bsdf;
pub mod mfd;
pub mod params;

use crate::{
    app::cache::Handle,
    io::{
        vgmo::{vgmo_header_ext_from_data, VgmoHeaderExt},
        OutputFileFormatOption,
    },
    measure::{
        bsdf::MeasuredBsdfData,
        mfd::{MeasuredMsfData, MeasuredNdfData, MeasuredSdfData},
    },
};
use base::{
    error::VgonioError,
    io::{
        Header, HeaderMeta, ReadFileError, ReadFileErrorKind, WriteFileError, WriteFileErrorKind,
    },
    math::{Mat3, Sph2, Sph3, Vec3},
    units::{rad, Radians},
    Asset, MeasuredData, MeasurementKind, Version,
};
use bxdf::brdf::measured::ClausenBrdf;
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

/// Where the measurement data is loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementSource {
    /// Measurement data is loaded from a file.
    Loaded(PathBuf),
    /// Measurement data is generated from a micro-surface.
    Measured(Handle<MicroSurface>),
}

impl MeasurementSource {
    /// Returns the path to the measurement data if it is loaded from a file.
    pub fn path(&self) -> Option<&Path> {
        match self {
            MeasurementSource::Loaded(p) => Some(p.as_path()),
            MeasurementSource::Measured(_) => None,
        }
    }

    /// Returns the micro-surface handle if the measurement data is generated.
    pub fn micro_surface(&self) -> Option<Handle<MicroSurface>> {
        match self {
            MeasurementSource::Loaded(_) => None,
            MeasurementSource::Measured(ms) => Some(*ms),
        }
    }
}

// TODO: add support for storing data in the memory in a compressed
//       format(maybe LZ4).
/// Structure for storing measurement data in the memory, especially
/// when loading from a file.
#[derive(Debug)]
pub struct Measurement {
    /// Internal tag for displaying the measurement data in the GUI.
    pub name: String,
    /// Origin of the measurement data.
    pub source: MeasurementSource,
    /// Timestamp of the measurement.
    pub timestamp: DateTime<Local>,
    /// Measurement data.
    pub measured: Box<dyn MeasuredData>,
}

unsafe impl Send for Measurement {}
unsafe impl Sync for Measurement {}

impl Asset for Measurement {}

impl PartialEq for Measurement {
    fn eq(&self, other: &Self) -> bool { self.source == other.source }
}

impl Measurement {
    /// Returns the kind of the measurement data.
    pub fn kind(&self) -> MeasurementKind { self.measured.kind() }

    /// Returns the Area Distribution Function data slice for the given
    /// azimuthal angle in radians.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    ///
    /// Azimuthal angle will be wrapped around to the range [0, 2π).
    ///
    /// 2π will be mapped to 0.
    ///
    /// # Arguments
    ///
    /// * `azimuth_m` - Azimuthal angle of the microfacet normal in radians.
    pub fn ndf_data_slice(&self, azimuth_m: Radians) -> Option<(&[f32], Option<&[f32]>)> {
        debug_assert!(self.kind() == MeasurementKind::Ndf);
        self.measured
            .downcast_ref::<MeasuredNdfData>()
            .and_then(|ndf| {
                let (azi, zen) = ndf.measurement_range().unwrap();
                let azimuth_m = azimuth_m.wrap_to_tau();
                let azimuth_m_idx = azi.index_of(azimuth_m);
                let opposite_azimuth_m = azimuth_m.opposite();
                let opposite_index =
                    if azi.start <= opposite_azimuth_m && opposite_azimuth_m <= azi.stop {
                        Some(azi.index_of(opposite_azimuth_m))
                    } else {
                        None
                    };
                let zen_step_count = zen.step_count_wrapped();
                Some((
                    &ndf.samples
                        [azimuth_m_idx * zen_step_count..(azimuth_m_idx + 1) * zen_step_count],
                    opposite_index.map(|index| {
                        &ndf.samples[index * zen_step_count..(index + 1) * zen_step_count]
                    }),
                ))
            })
    }

    /// Returns the Masking Shadowing Function data slice for the given
    /// microfacet normal and azimuthal angle of the incident direction.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    pub fn msf_data_slice(
        &self,
        azimuth_m: Radians,
        zenith_m: Radians,
        azimuth_i: Radians,
    ) -> (&[f32], Option<&[f32]>) {
        debug_assert!(
            self.kind() == MeasurementKind::Msf,
            "measurement data kind should be MicrofacetMaskingShadowing"
        );
        let msf = self.measured.downcast_ref::<MeasuredMsfData>().unwrap();
        msf.slice_at(azimuth_m, zenith_m, azimuth_i)
    }

    /// Writes the measurement data to a file in VGMO format.
    pub fn write_to_file(
        &self,
        filepath: &Path,
        format: &OutputFileFormatOption,
    ) -> Result<(), VgonioError> {
        match format {
            OutputFileFormatOption::Vgmo {
                encoding,
                compression,
            } => {
                let timestamp = {
                    let mut timestamp = [0_u8; 32];
                    timestamp.copy_from_slice(
                        base::utils::iso_timestamp_from_datetime(&self.timestamp).as_bytes(),
                    );
                    timestamp
                };
                let header = Header {
                    meta: HeaderMeta {
                        version: Version::new(0, 1, 0),
                        timestamp,
                        length: 0,
                        sample_size: 4,
                        encoding: *encoding,
                        compression: *compression,
                    },
                    extra: vgmo_header_ext_from_data(&self.measured),
                };
                let filepath = filepath.with_extension("vgmo");
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&filepath)
                    .map_err(|err| {
                        VgonioError::from_io_error(err, "Failed to open measurement file.")
                    })?;
                let mut writer = BufWriter::new(file);
                crate::io::vgmo::write(&mut writer, header, &self.measured).map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: err,
                        },
                        "Failed to write VGMO file.",
                    )
                })?;

                writer.flush().map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: WriteFileErrorKind::Write(err),
                        },
                        "Failed to flush VGMO file.",
                    )
                })?;
            }
            OutputFileFormatOption::Exr { resolution } => {
                let filepath = filepath.with_extension("exr");
                match self.measured.kind() {
                    MeasurementKind::Bsdf => {
                        let bsdf = self.measured.downcast_ref::<MeasuredBsdfData>().unwrap();
                        bsdf.write_as_exr(&filepath, &self.timestamp, *resolution)?
                    }
                    MeasurementKind::Ndf => {
                        let ndf = self.measured.downcast_ref::<MeasuredNdfData>().unwrap();
                        ndf.write_as_exr(&filepath, &self.timestamp, *resolution)?
                    }
                    MeasurementKind::Msf => {
                        todo!("Writing MSF to EXR is not supported yet.");
                    }
                    MeasurementKind::Sdf => {
                        let sdf = self.measured.downcast_ref::<MeasuredSdfData>().unwrap();
                        sdf.write_histogram_as_exr(&filepath, &self.timestamp, *resolution)?;
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Loads the measurement data from a file.
    pub fn read_from_file(filepath: &Path) -> Result<Self, VgonioError> {
        let file = File::open(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to open measurement file: {}", filepath.display()),
            )
        })?;
        let mut reader = BufReader::new(file);

        if let Some(extension) = filepath.extension()
            && extension == OsStr::new("json")
        {
            return ClausenBrdf::load_from_reader(reader).map(|brdf| Measurement {
                name: format!(
                    "clausen_{}",
                    filepath.file_name().unwrap().to_str().unwrap()
                ),
                source: MeasurementSource::Loaded(filepath.to_path_buf()),
                timestamp: Local::now(),
                measured: Box::new(brdf),
            });
        }

        let header = Header::<VgmoHeaderExt>::read(&mut reader).map_err(|err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: ReadFileErrorKind::Read(err),
                },
                "Failed to read VGMO file.",
            )
        })?;
        log::debug!("Read VGMO file of length: {}", header.meta.length);
        log::debug!("Header: {:?}", header);

        let path = filepath.to_path_buf();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("invalid file stem")
            .to_string();

        let measured = crate::io::vgmo::read(&mut reader, &header).map_err(|err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: err,
                },
                "Failed to read VGMO file.",
            )
        })?;
        let timestamp = DateTime::from(
            DateTime::parse_from_rfc3339(std::str::from_utf8(&header.meta.timestamp).map_err(
                |err| VgonioError::from_utf8_error(err, "Failed to parse timestamp from file."),
            )?)
            .map_err(|err| {
                VgonioError::new(
                    format!(
                        "Failed to parse timestamp from file: {}",
                        filepath.display()
                    ),
                    Some(Box::new(err)),
                )
            })?,
        );

        Ok(Measurement {
            name,
            source: MeasurementSource::Loaded(path),
            timestamp,
            measured,
        })
    }
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
    pub fn transform_disc(dest: Sph2, disc_radius: f32, orbit_radius: f32) -> Mat3 {
        Self::transform_to(dest)
            * Mat3::from_diagonal(Vec3::new(disc_radius, disc_radius, orbit_radius))
    }
}

/// Generates uniformly distributed samples on the unit disk.
///
/// NOTE: samples are generated on the xy-plane, with the z component set to 1.0
/// in the returned vector. This could simplify the transformation of the
/// samples when rotating the disk.
pub fn uniform_sampling_on_unit_disk(num: usize) -> Box<[Vec3]> {
    const SEED: u64 = 0;

    let range: Uniform<f32> = Uniform::new(0.0, 1.0);
    let mut samples = Vec::with_capacity(num);
    samples.resize(num, Vec3::Z);
    let mut samples = Box::new_uninit_slice(num);
    samples
        .par_chunks_mut(8192)
        .enumerate()
        .for_each(|(i, chunks)| {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED);
            rng.set_stream(i as u64);

            chunks.iter_mut().for_each(|v| {
                let r = range.sample(&mut rng).sqrt();
                let a = range.sample(&mut rng) * std::f32::consts::TAU;
                v.write(Vec3::new(r * a.cos(), r * a.sin(), 1.0));
            });
        });

    unsafe { samples.assume_init() }
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

// TODO: unify the data sampling and the measured data.
pub struct MeasuredDataSampler<'a> {
    data: &'a dyn MeasuredData,
}
